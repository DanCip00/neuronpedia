"""Paired full-history window scoring with request-private downstream state."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from interp_engine import VLLMModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.source import _tokenizer_ids, _validate_ids
from neuronpedia_inference.endpoints.steer.source import _resolve_source_metadata, _uses_speculative_decode
from neuronpedia_inference.engine_adapter import BackendUnsupported, assert_steering_available, declares_static_taps
from neuronpedia_inference.memory_cost import source_window_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas.steer import SourceSteerRequest
from neuronpedia_inference.schemas.window import (
    SourceWindowScoreArm,
    SourceWindowScoreRequest,
    SourceWindowScoreResponse,
)
from neuronpedia_inference.shared import Model, RecoverableOutOfMemory, recover_from_oom, with_request_lock
from neuronpedia_inference.vllm_optional import SamplingParams

logger = logging.getLogger(__name__)
router = APIRouter()


def _one_worker(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list | tuple) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("window scoring expected one worker result")
    return payload[0]


@router.post("/steer/source/window", responses={200: {"model": SourceWindowScoreResponse}})
@with_request_lock(exclusive=False, cost=source_window_cost)
async def score_source_window(request: SourceWindowScoreRequest):
    """Compare paired causal forwards; never regenerate or alter provided tokens."""
    model = Model.get_instance()
    registered: list[str] = []
    try:
        config = Config.get_instance()
        config.check_requested_model(request.model)
        if not isinstance(model, VLLMModel) or int(getattr(model, "tensor_parallel_size", 1) or 1) != 1:
            raise ValueError("window scoring requires single-GPU vLLM")
        assert_steering_available(model, "code-window steering")
        if declares_static_taps(model) or _uses_speculative_decode(model):
            raise ValueError("window scoring requires hooked vLLM without speculative decoding")
        ids = list(request.token_ids)
        _validate_ids(ids, _tokenizer_ids(model.tokenizer), 0, "tokenIds")
        if len(ids) >= config.token_limit:
            raise ValueError("window plus one disposable output token exceeds the server token limit")
        source_request = SourceSteerRequest(
            model=request.model, prompt_token_ids=[ids], steering=request.steering, return_intervention_diagnostics=True
        )
        spec, resolved = _resolve_source_metadata(source_request)
        assert spec is not None
        d_sae = SAEManager.get_instance().get_d_sae(request.steering.source)
        if d_sae is None or any(i >= d_sae for i in request.trace_feature_indices):
            raise ValueError("trace feature index outside the source SAE")
        await model._ensure_engine()

        async def forward(arm: str) -> str:
            rid = model._new_request_id("np-code-window")
            registered.append(rid)
            task = asyncio.create_task(
                model.engine.collective_rpc("register_code_window", args=(rid, spec, len(ids), arm))
            )
            try:
                await asyncio.shield(task)
            finally:
                if not task.done():
                    with contextlib.suppress(Exception):
                        await asyncio.shield(task)
            # max_tokens=1 causes prompt-only capture. This output is discarded.
            await model._run_one(
                model._prompt(ids, private_kv_for=rid), SamplingParams(max_tokens=1, temperature=0.0), request_id=rid
            )
            return rid

        reference_id = await forward("baseline")
        reference = _one_worker(
            await model.engine.collective_rpc(
                "store_code_window_reference", args=(reference_id, spec, len(ids), request.trace_feature_indices)
            )
        )
        arms = ["feature_inactive"]
        if request.include_single_pulse:
            arms.append("single_pulse")
        if request.include_baseline_repeat:
            arms.append("baseline_repeat")
        comparisons: list[SourceWindowScoreArm] = []
        for arm in arms:
            rid = await forward(arm)
            comparison = _one_worker(
                await model.engine.collective_rpc("compare_code_window", args=(rid, reference_id, len(ids), arm))
            )
            comparisons.append(SourceWindowScoreArm.model_validate(comparison))
            await model.engine.collective_rpc("cleanup_code_window", args=(rid,))
        return SourceWindowScoreResponse(
            model_input_token_ids=ids,
            prediction_positions=list(range(len(ids) - 1)),
            resolved=resolved,
            baseline_feature_activations=reference["feature_activations"],
            baseline_residual_norms=reference["residual_norms"],
            arms=comparisons,
        ).model_dump()
    except (ValueError, BackendUnsupported) as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        if recover_from_oom(exc):
            return JSONResponse(content={"error": str(RecoverableOutOfMemory())}, status_code=503)
        logger.exception("Error scoring a code window")
        return JSONResponse(content={"error": "An error occurred while scoring the code window"}, status_code=500)
    finally:
        if isinstance(model, VLLMModel):
            for rid in registered:
                await asyncio.shield(model.engine.collective_rpc("cleanup_code_window", args=(rid,)))
