import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import wraps

import torch
from interp_engine import EagerModel, VLLMModel

from neuronpedia_inference.resilience import is_allocator_oom, reclaim_after_oom
from neuronpedia_inference.sae_cache import sae_cache

logger = logging.getLogger(__name__)

# The backend actually loaded at startup: the eager engine model or the vLLM async backend.
# Helpers that reach into backend attributes (``.tokenizer``, ``.cfg``) take this instead of
# ``object``; the backend-agnostic capture helpers in ``engine_adapter`` keep ``object`` because
# they isinstance-narrow and accept any duck-typed backend.
LoadedModel = VLLMModel | EagerModel

# Timeout for acquiring a request slot (seconds). 0 = no timeout
REQUEST_LOCK_TIMEOUT = float(os.environ.get("REQUEST_LOCK_TIMEOUT", "300"))  # 5 min default


class RequestBusy(Exception):
    """Raised when ``fail_if_busy`` is set and no slot, or no memory, is free right now."""


class RequestTooLarge(Exception):
    """Raised when one request's estimated working set exceeds the ENTIRE budget.

    Waiting would not help -- no amount of other requests finishing makes room -- so the
    endpoint returns this to the caller as a 4xx naming both numbers.
    """

    def __init__(self, needed_bytes: int, budget_bytes: int) -> None:
        self.needed_bytes = needed_bytes
        self.budget_bytes = budget_bytes
        super().__init__(
            f"This request needs an estimated {needed_bytes / 1024**3:.2f} GiB of "
            f"working memory, but only {budget_bytes / 1024**3:.2f} GiB is available for "
            f"all requests combined. Reduce the number of selected sources, the prompt "
            f"length, or the number of results."
        )


class VramBudget:
    """Admit requests against measured free VRAM, not just a flat request count.

    The concurrency limiter alone cannot prevent OOM, because requests are not
    interchangeable: a lens or steer call costs a few hundred MB while an all-layers
    activation search over wide SAEs costs orders of magnitude more. Four of the former fit
    comfortably in the headroom outside the vLLM pool; four of the latter do not.

    So each endpoint declares an estimated byte cost for the request it is about to serve
    (see :mod:`neuronpedia_inference.memory_cost`) and reserves that much here. Requests wait
    for room rather than racing each other into the allocator, which is what lets
    ``max_concurrent`` be raised for the cheap endpoints without gambling on the expensive
    ones.

    A total of 0 disables rationing entirely (non-CUDA device, or too little free memory to
    be worth dividing up), in which case ``reserve`` is a no-op.
    """

    def __init__(self) -> None:
        self._total = 0
        self._available = 0
        self._condition = asyncio.Condition()

    def configure(self, total_bytes: int) -> None:
        self._total = max(0, int(total_bytes))
        self._available = self._total
        if self._total:
            logger.info(
                "[BUDGET] configured with %.2f GiB for concurrent request working sets",
                self._total / 1024**3,
            )
        else:
            logger.info("[BUDGET] disabled (no measured budget to ration)")

    @property
    def total_bytes(self) -> int:
        return self._total

    @property
    def available_bytes(self) -> int:
        return self._available

    @property
    def enabled(self) -> bool:
        return self._total > 0

    async def acquire(
        self,
        nbytes: int,
        *,
        timeout: float = REQUEST_LOCK_TIMEOUT,
        fail_if_busy: bool = False,
    ) -> int:
        """Claim ``nbytes``, waiting for room. Returns the amount actually claimed.

        Callers MUST pass the return value to :meth:`release` when done -- it is 0 when the
        budget is disabled, so releasing it is harmless either way. Prefer :meth:`reserve`
        unless the reservation has to outlive the calling frame (the streaming lens response
        holds one for the lifetime of its generator).

        ``fail_if_busy`` raises :class:`RequestBusy` instead of waiting when there is no room
        right now. A caller that can try another pod wants this: a free slot on a pod with no
        memory to go with it still means "come back later", and the wait for room is the one
        remaining place a fail-fast request can block for minutes.

        Raises :class:`RequestTooLarge` if it can never fit, or ``TimeoutError`` if it did
        not fit in time.
        """
        nbytes = max(0, int(nbytes))
        if not self.enabled or nbytes == 0:
            return 0
        if nbytes > self._total:
            raise RequestTooLarge(nbytes, self._total)

        async with self._condition:
            if fail_if_busy and self._available < nbytes:
                raise RequestBusy()
            has_room = self._condition.wait_for(lambda: self._available >= nbytes)
            if timeout and timeout > 0:
                await asyncio.wait_for(has_room, timeout=timeout)
            else:
                await has_room
            self._available -= nbytes
        logger.debug(
            "[BUDGET] reserved %.3f GiB, %.3f GiB of %.2f GiB left",
            nbytes / 1024**3,
            self._available / 1024**3,
            self._total / 1024**3,
        )
        return nbytes

    async def release(self, nbytes: int) -> None:
        """Return a claim from :meth:`acquire` and wake anything waiting on room."""
        if nbytes <= 0:
            return
        async with self._condition:
            self._available = min(self._total, self._available + nbytes)
            self._condition.notify_all()

    @asynccontextmanager
    async def reserve(self, nbytes: int, *, timeout: float = REQUEST_LOCK_TIMEOUT, fail_if_busy: bool = False):
        """Hold ``nbytes`` of the budget for the duration of the block."""
        claimed = await self.acquire(nbytes, timeout=timeout, fail_if_busy=fail_if_busy)
        try:
            yield
        finally:
            await self.release(claimed)


class ConcurrencyLimiter:
    """Admit requests based on backend concurrency-safety.

    Shared vLLM requests use a bounded semaphore so the engine can batch them. Exclusive
    requests close a writer gate and drain every semaphore permit before running, which
    guarantees that no shared request can overlap them. Non-vLLM backends remain strictly
    one-at-a-time regardless of the requested mode.
    """

    def __init__(self) -> None:
        self._concurrent = False
        self._max = 1
        self._sem = asyncio.Semaphore(1)
        self._exclusive = asyncio.Lock()
        self._single = asyncio.Lock()
        self._active_shared = 0

    def configure(self, *, concurrent: bool, max_concurrent: int) -> None:
        self._concurrent = bool(concurrent)
        self._max = max(1, int(max_concurrent))
        self._sem = asyncio.Semaphore(self._max)
        self._active_shared = 0
        logger.info(
            "[LIMITER] configured concurrent=%s max_concurrent=%d",
            self._concurrent,
            self.max_concurrent,
        )

    @property
    def max_concurrent(self) -> int:
        return self._max if self._concurrent else 1

    def is_busy(self, exclusive: bool = True) -> bool:
        if not self._concurrent:
            return self._single.locked()
        if exclusive:
            return self._exclusive.locked() or self._active_shared > 0
        return self._exclusive.locked() or self._sem.locked()

    async def _acquire_shared(self):  # type: ignore[no-untyped-def]
        # Pass through the writer gate before taking capacity. An exclusive waiter holds
        # this gate while existing readers drain, so new shared work cannot join its batch.
        await self._exclusive.acquire()
        try:
            await self._sem.acquire()
        finally:
            self._exclusive.release()
        self._active_shared += 1
        return _SharedConcurrencyLease(self)

    async def _acquire_exclusive(self):  # type: ignore[no-untyped-def]
        await self._exclusive.acquire()
        acquired = 0
        try:
            for _ in range(self._max):
                await self._sem.acquire()
                acquired += 1
        except BaseException:
            for _ in range(acquired):
                self._sem.release()
            self._exclusive.release()
            raise
        return _ExclusiveConcurrencyLease(self)

    async def acquire(self, *, exclusive: bool = True, timeout: float = REQUEST_LOCK_TIMEOUT):
        """Acquire and return a lease that the caller must release."""
        if not self._concurrent:
            primitive = self._single
            if timeout and timeout > 0:
                await asyncio.wait_for(primitive.acquire(), timeout=timeout)
            else:
                await primitive.acquire()
            return primitive

        acquire = self._acquire_exclusive() if exclusive else self._acquire_shared()
        if timeout and timeout > 0:
            return await asyncio.wait_for(acquire, timeout=timeout)
        return await acquire

    @asynccontextmanager
    async def slot(
        self,
        *,
        exclusive: bool = True,
        fail_if_busy: bool = False,
        timeout: float = REQUEST_LOCK_TIMEOUT,
    ):
        if fail_if_busy and self.is_busy(exclusive):
            raise RequestBusy()
        acquired = await self.acquire(exclusive=exclusive, timeout=timeout)
        try:
            yield
        finally:
            acquired.release()


class _SharedConcurrencyLease:
    """One shared vLLM capacity permit."""

    def __init__(self, limiter: ConcurrencyLimiter) -> None:
        self._limiter = limiter
        self._released = False

    def release(self) -> None:
        if self._released:
            raise RuntimeError("Concurrency lease released twice")
        self._released = True
        self._limiter._active_shared -= 1
        self._limiter._sem.release()


class _ExclusiveConcurrencyLease:
    """Every vLLM capacity permit plus the gate that blocks new shared work."""

    def __init__(self, limiter: ConcurrencyLimiter) -> None:
        self._limiter = limiter
        self._released = False

    def release(self) -> None:
        if self._released:
            raise RuntimeError("Concurrency lease released twice")
        self._released = True
        for _ in range(self._limiter._max):
            self._limiter._sem.release()
        self._limiter._exclusive.release()


# Global limiter (configured at startup via configure_limiter()).
limiter = ConcurrencyLimiter()

# Global VRAM budget (configured at startup via configure_budget()).
budget = VramBudget()


def configure_limiter(*, concurrent: bool, max_concurrent: int) -> None:
    limiter.configure(concurrent=concurrent, max_concurrent=max_concurrent)


def configure_budget(total_bytes: int) -> None:
    budget.configure(total_bytes)


class RecoverableOutOfMemory(Exception):
    """A torch allocator OOM that left the CUDA context intact -- retry, don't restart."""

    def __init__(self) -> None:
        super().__init__(
            "The server ran out of GPU memory serving this request. The CUDA context is "
            "intact, so this is retryable: try again, or reduce the prompt length, the "
            "number of selected sources, or the number of results."
        )


def recover_from_oom(exc: BaseException) -> bool:
    """True if ``exc`` was an allocator OOM, having first reclaimed cached blocks.

    Endpoints that funnel every failure into a 500 call this from their ``except`` block so an
    OOM comes back as a retryable 503 with a clean allocator behind it, instead of a 500 that
    leaves the next request to inherit the fragmentation.
    """
    if not is_allocator_oom(exc):
        return False
    logger.error("[BUDGET] allocation OOM; reclaiming and failing this request only")
    reclaim_after_oom()
    return True


def _handler_request(args, kwargs):  # type: ignore[no-untyped-def]
    return kwargs.get("request", args[0] if args else None)


def _estimate_cost(cost, args, kwargs) -> int:  # type: ignore[no-untyped-def]
    """Run an endpoint's cost estimator over the handler's own arguments.

    Never allowed to fail a request: a broken estimator degrades to "free" (and so to the
    old count-only behaviour) rather than 500-ing a request that would have worked.
    """
    if cost is None:
        return 0
    request = _handler_request(args, kwargs)
    if request is None:
        return 0
    try:
        return int(cost(request))
    except Exception:  # noqa: BLE001 - advisory only, must not break the request
        logger.exception("[BUDGET] cost estimator failed; admitting without a reservation")
        return 0


def _wants_fail_fast(args, kwargs) -> bool:  # type: ignore[no-untyped-def]
    """Whether the handler's own request asked to be refused rather than queued.

    Read off the request model like ``cost`` is, so an endpoint opts in by declaring the field
    and nothing has to be passed at the decorator.
    """
    request = _handler_request(args, kwargs)
    return bool(getattr(request, "fail_if_busy", False))


def _estimate_sae_residency(args, kwargs) -> int:  # type: ignore[no-untyped-def]
    """Bytes of SAE weights the handler will need GPU-resident, when paging is on.

    Applied to every decorated endpoint rather than declared per endpoint: the size comes
    from the sources named in the request and is measured at load time, so there is nothing
    for an endpoint (or a new SAE set) to configure.
    """
    if not sae_cache.enabled:
        return 0
    request = _handler_request(args, kwargs)
    if request is None:
        return 0
    try:
        from neuronpedia_inference.memory_cost import sae_residency_bytes

        return int(sae_residency_bytes(request))
    except Exception:  # noqa: BLE001 - advisory only, must not break the request
        logger.exception("[SAE-CACHE] residency estimate failed; admitting without a reservation")
        return 0


def with_request_lock(exclusive: bool = True, cost=None):  # type: ignore[no-untyped-def]
    """Decorator: admit the handler through the limiter, and optionally the VRAM budget.

    ``exclusive=True`` (default) is the safe choice: it runs alone on every backend,
    draining in-flight shared vLLM work before entering. Concurrency-safe endpoints pass
    ``exclusive=False`` to run concurrently on vLLM.

    ``cost`` is a callable taking the handler's request model and returning the estimated
    peak working-set bytes. Endpoints that pass one are additionally admitted against the
    measured budget (:class:`VramBudget`), so N concurrent requests can only be in flight
    when N of them actually fit. Omitting it keeps the old count-only admission.

    A request carrying ``fail_if_busy`` is refused with :class:`RequestBusy` rather than
    queued, at each of the three gates below. All three, because a client that fails over on
    a refusal needs the refusal to be immediate: one gate left blocking is enough to hold the
    connection for the full timeout and hide the other two.

    When SAE paging is on, every endpoint additionally reserves room for the SAE weights it
    will page in (:mod:`neuronpedia_inference.sae_cache`). That reservation is what makes
    eviction safe: it bounds the weights in use at any moment to the residency budget, so
    the cache always has something evictable that nobody is reading.
    """

    def decorator(func):  # type: ignore
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            fail_if_busy = _wants_fail_fast(args, kwargs)
            try:
                async with (
                    limiter.slot(exclusive=exclusive, fail_if_busy=fail_if_busy),
                    sae_cache.reserve(
                        _estimate_sae_residency(args, kwargs),
                        timeout=REQUEST_LOCK_TIMEOUT,
                        fail_if_busy=fail_if_busy,
                    ),
                    budget.reserve(_estimate_cost(cost, args, kwargs), fail_if_busy=fail_if_busy),
                ):
                    return await func(*args, **kwargs)
            except TimeoutError as exc:
                logger.error("[LIMITER] Timeout waiting for a request slot")
                raise TimeoutError("Request timed out waiting for a slot") from exc
            except Exception as exc:
                # Only for an OOM that escaped the handler; everything else propagates
                # untouched, including the fatal CUDA states resilience.py restarts on.
                if recover_from_oom(exc):
                    raise RecoverableOutOfMemory() from exc
                raise

        return wrapper

    return decorator


class Model:
    _instance: LoadedModel

    @classmethod
    def get_instance(cls) -> LoadedModel:
        if cls._instance is None:
            raise ValueError("Model not initialized")
        return cls._instance

    @classmethod
    def set_instance(cls, model: LoadedModel) -> None:
        cls._instance = model


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
