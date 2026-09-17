"""Neuronpedia vLLM worker extension for request-scoped SAE feature edits."""

from __future__ import annotations

from typing import Any, Literal

import torch
from interp_engine.vllm_capture._demux import _ensure_dev, _ensure_patched, _get_demux, _maybe_unregister, _release_hook
from interp_engine.vllm_capture.requests import _ensure_hook, _write_site
from interp_engine.vllm_plugin import InterpWorkerExtension

from neuronpedia_inference.sae_interventions import SAEFeatureIntervention, sae_intervention_delta
from neuronpedia_inference.saes.saelens import SaeLensSAE

PositionPolicy = Literal["next_token", "each_generated_token"]

# One entry per distinct SAE this worker process has been asked to encode with. Loading an SAE
# from disk into the GPU took ~5 s per request when it was done per request (an 80k x 5120
# SAE is 1.6 GB in bf16), and scale/ablate need the whole encoder for TopK selection, so the
# only way to make those requests cost what an `add` costs is to keep the SAE resident. This
# process serves one model with a handful of SAE sets, so the dict stays small.
_WORKER_SAE_CACHE: dict[tuple[str, str, str, str], object] = {}


def _load_worker_sae(spec: dict[str, Any], device: torch.device, dtype: torch.dtype) -> object:
    """Return the SAE named by ``spec``, loading it on first use in this worker process."""
    dtype_name = str(spec.get("dtype") or str(dtype).removeprefix("torch."))
    key = (str(spec["release"]), str(spec["sae_id"]), str(device), dtype_name)
    cached = _WORKER_SAE_CACHE.get(key)
    if cached is None:
        cached, _hook = SaeLensSAE.load(key[0], key[1], key[2], key[3])
        _WORKER_SAE_CACHE[key] = cached
    return cached


def _features(spec: dict[str, Any]) -> list[SAEFeatureIntervention]:
    return [
        SAEFeatureIntervention(
            feature_index=int(item["feature_index"]),
            operation=str(item["operation"]),  # type: ignore[arg-type]
            value=None if item.get("value") is None else float(item["value"]),
        )
        for item in spec.get("features", [])
    ]


def _make_sae_modifier(
    spec: dict[str, Any],
    *,
    dev: torch.device,
    dt: torch.dtype,
    prompt_len: int,
    diag_store: list[dict[str, Any]],
):
    features = _features(spec)
    raw_vectors = spec.get("decoder_vectors")
    decoder_vectors = None
    additive_only = all(feature.operation == "add" for feature in features)
    if additive_only:
        if raw_vectors is None:
            raise ValueError("additive-only SAE steering requires supplied decoder_vectors")
        decoder_vectors = {
            int(feature_index): torch.tensor(vector, device=dev, dtype=dt)
            for feature_index, vector in raw_vectors.items()
        }
        sae = None
    else:
        sae = _load_worker_sae(spec["sae"], dev, dt)
    policy: PositionPolicy = spec["position_policy"]
    want_diag = bool(spec.get("return_diagnostics", False))
    # Absolute position of the next row this request will show the hook. The hook only sees the
    # rows scheduled this step, so the position has to be tracked here: vLLM may split a prompt
    # across steps (chunked prefill, which happens to a short prompt whenever other requests
    # hold the token budget), and the row to edit is the one at prompt_len - 1 whichever chunk
    # it lands in. Raising here instead would surface inside the model forward and take the
    # engine down, not just this request.
    next_position = 0

    def _modify(full: torch.Tensor) -> torch.Tensor:
        nonlocal next_position
        delta = torch.zeros_like(full)
        if full.dim() != 2:
            raise ValueError(f"SAE source steering supports 2-D residual rows only, got {tuple(full.shape)}")
        num_rows = int(full.shape[0])
        if num_rows > 1 and next_position >= prompt_len:
            # Once decoding has begun a request advances one row per step (speculative decoding
            # is refused at the API), so several rows at once means vLLM preempted the request
            # and is recomputing its prompt from the start.
            next_position = 0
        start = next_position
        next_position += num_rows
        # Rows to edit, in absolute positions: prompt_len - 1 always (it predicts the first
        # generated token); everything after it as well under each_generated_token.
        lo = max(start, prompt_len - 1)
        hi = start + num_rows if policy == "each_generated_token" else min(start + num_rows, prompt_len)
        if lo >= hi:
            return delta
        rows = full[lo - start : hi - start]
        row_offset = lo
        row_delta, diagnostics = sae_intervention_delta(
            rows,
            sae,
            features,
            return_diagnostics=want_diag,
            decoder_vectors_by_feature=decoder_vectors,
        )
        delta[lo - start : hi - start] = row_delta
        if diagnostics is not None:
            diag_store.append(
                {
                    "row_offset": int(row_offset),
                    "num_rows": diagnostics.edited_rows,
                    "edit_count": diagnostics.edit_count,
                    "perturbation_norms": diagnostics.perturbation_norms,
                    "feature_activations": {str(k): v for k, v in diagnostics.feature_activations.items()},
                    "encoded": diagnostics.encoded,
                    "active": diagnostics.active,
                }
            )
        return delta

    return _modify


def worker_register_sae_feature_steering(
    worker: object,
    req_id: str,
    specs: list[dict[str, Any]],
    prompt_len: int,
) -> None:
    demux = _get_demux(worker)
    _ensure_patched(worker, demux)
    _ensure_dev(worker, demux)
    demux.registered.add(req_id)
    mods = demux.steer_mods.setdefault(req_id, {})
    diag_by_req = getattr(worker, "_np_sae_steering_diagnostics", None)
    if diag_by_req is None:
        diag_by_req = {}
        worker._np_sae_steering_diagnostics = diag_by_req  # type: ignore[attr-defined]
    diag_store: list[dict[str, Any]] = []
    diag_by_req[req_id] = diag_store
    for spec in specs:
        site = _write_site(worker, spec)
        mods[site] = (
            _make_sae_modifier(spec, dev=demux.dev, dt=demux.dt, prompt_len=int(prompt_len), diag_store=diag_store),
            set(),
            int(prompt_len),
        )
        _ensure_hook(worker, demux, site)


def worker_collect_sae_feature_steering_diagnostics(worker: object, req_id: str) -> list[dict[str, Any]]:
    diag_by_req = getattr(worker, "_np_sae_steering_diagnostics", {})
    return list(diag_by_req.get(req_id, []))


def worker_unregister_sae_feature_steering(worker: object, req_id: str) -> None:
    demux = _get_demux(worker)
    for site in demux.steer_mods.pop(req_id, {}):
        _release_hook(demux, site)
    getattr(worker, "_np_sae_steering_diagnostics", {}).pop(req_id, None)
    _maybe_unregister(demux, req_id)


class NeuronpediaWorkerExtension(InterpWorkerExtension):
    """vLLM worker extension with request-scoped SAE feature interventions."""

    def register_sae_feature_steering(self, req_id: str, specs: list[dict[str, Any]], prompt_len: int) -> None:
        return worker_register_sae_feature_steering(self, req_id, specs, prompt_len)

    def collect_sae_feature_steering_diagnostics(self, req_id: str) -> list[dict[str, Any]]:
        return worker_collect_sae_feature_steering_diagnostics(self, req_id)

    def unregister_sae_feature_steering(self, req_id: str) -> None:
        return worker_unregister_sae_feature_steering(self, req_id)
