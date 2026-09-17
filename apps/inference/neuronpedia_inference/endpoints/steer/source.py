from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from typing import Any, cast

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from interp_engine import VLLMModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.endpoints.activation.source import _apply_insertion, _tokenizer_ids, _validate_ids
from neuronpedia_inference.endpoints.steer.completion import resolve_max_new_tokens
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    assert_steer_layers_declared,
    assert_steering_available,
    declares_static_taps,
    tlens_hook_to_point,
)
from neuronpedia_inference.memory_cost import steer_source_cost
from neuronpedia_inference.sae_interventions import (
    SAEFeatureIntervention,
    decoder_matrix,
    validate_feature_interventions,
)
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import (
    SourceSteerFeature,
    SourceSteerInterventionDiagnostics,
    SourceSteerRequest,
    SourceSteerResolvedMetadata,
    SourceSteerResponse,
    SourceSteerTokenLogprob,
    SourceSteerTokenTopLogprob,
)
from neuronpedia_inference.shared import Model, RecoverableOutOfMemory, recover_from_oom, with_request_lock
from neuronpedia_inference.vllm_optional import SamplingParams

logger = logging.getLogger(__name__)
router = APIRouter()

_DECODER_VECTOR_CACHE: OrderedDict[tuple[object, ...], dict[str, list[float]]] = OrderedDict()
_MAX_DECODER_VECTOR_CACHE_ENTRIES = 128


def _source_set_contains(manager: SAEManager, source_set: str, source: str) -> bool:
    return source_set in manager.get_valid_sae_sets() and source in manager.sae_set_to_saes.get(source_set, [])


def _uses_speculative_decode(model: VLLMModel) -> bool:
    kwargs = getattr(model, "_engine_kwargs", {})
    return any(kwargs.get(key) for key in ("speculative_config", "speculative_model", "num_speculative_tokens"))


def _operation_payload(request: SourceSteerRequest) -> list[SAEFeatureIntervention]:
    if request.steering is None:
        return []
    return [
        SAEFeatureIntervention(
            feature_index=int(feature.feature_index),
            operation=feature.operation.value,
            value=None if feature.value is None else float(feature.value),
        )
        for feature in request.steering.features
    ]


def _worker_feature_payload(request: SourceSteerRequest) -> list[dict[str, Any]]:
    def is_effective(feature: SourceSteerFeature) -> bool:
        if feature.operation.value == "ablate":
            return True
        assert feature.value is not None
        return not (
            (feature.operation.value == "add" and float(feature.value) == 0.0)
            or (feature.operation.value == "scale" and float(feature.value) == 1.0)
        )

    return [
        {
            "feature_index": int(feature.feature_index),
            "operation": feature.operation.value,
            "value": None if feature.value is None else float(feature.value),
        }
        for feature in (request.steering.features if request.steering else [])
        if is_effective(feature)
    ]


def _additive_decoder_vectors(
    request: SourceSteerRequest,
    manager: SAEManager,
    release: str,
    saelens_id: str,
) -> dict[str, list[float]] | None:
    steering = request.steering
    if steering is None:
        return None
    effective_ids = {int(item["feature_index"]) for item in _worker_feature_payload(request)}
    features = [feature for feature in steering.features if int(feature.feature_index) in effective_ids]
    if not features or any(feature.operation.value != "add" for feature in features):
        return None
    feature_ids = tuple(int(feature.feature_index) for feature in features)
    config = Config.get_instance()
    key = (
        request.model,
        release,
        steering.source,
        saelens_id,
        feature_ids,
        config.sae_dtype,
        str(config.device),
    )
    cached = _DECODER_VECTOR_CACHE.get(key)
    if cached is not None:
        _DECODER_VECTOR_CACHE.move_to_end(key)
        return cached
    sae = manager.get_sae(steering.source)
    matrix = decoder_matrix(sae)
    ids = torch.tensor(feature_ids, device=matrix.device, dtype=torch.long)
    selected = matrix.index_select(0, ids).detach().to(device="cpu", dtype=torch.float32)
    payload = {str(feature_id): selected[index].tolist() for index, feature_id in enumerate(feature_ids)}
    _DECODER_VECTOR_CACHE[key] = payload
    while len(_DECODER_VECTOR_CACHE) > _MAX_DECODER_VECTOR_CACHE_ENTRIES:
        _DECODER_VECTOR_CACHE.popitem(last=False)
    return payload


def _logprob_value(value: object) -> float | None:
    raw = value.get("logprob") if isinstance(value, dict) else getattr(value, "logprob", value)
    if type(raw) not in (int, float):
        return None
    return float(cast(int | float, raw))


def _emitted_logprob(entry: object, token_id: int) -> float | None:
    if not isinstance(entry, dict):
        return None
    found = entry.get(token_id)
    if found is None:
        found = entry.get(str(token_id))
    return None if found is None else _logprob_value(found)


def _top_logprobs(model: VLLMModel, per_position: object, index: int, limit: int) -> list[SourceSteerTokenTopLogprob]:
    if not limit or not per_position or index >= len(per_position):  # type: ignore[arg-type]
        return []
    entry = per_position[index]  # type: ignore[index]
    if not isinstance(entry, dict):
        return []

    def _rank(item: tuple[object, object]) -> tuple[int, float]:
        token_id, value = item
        rank = getattr(value, "rank", None)
        logprob = _logprob_value(value)
        return (10**9 if rank is None else int(rank), -(logprob if logprob is not None else float("-inf")))

    out: list[SourceSteerTokenTopLogprob] = []
    for raw_tid, lp in sorted(entry.items(), key=_rank)[:limit]:
        tid = int(raw_tid)
        logprob = _logprob_value(lp)
        if logprob is None:
            continue
        out.append(
            SourceSteerTokenTopLogprob(
                token_id=tid,
                token=model.tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                logprob=logprob,
            )
        )
    return out


def _format_logprobs(model: VLLMModel, token_ids: list[int], raw: object, limit: int) -> list[SourceSteerTokenLogprob]:
    rows: list[SourceSteerTokenLogprob] = []
    for index, token_id in enumerate(token_ids):
        rows.append(
            SourceSteerTokenLogprob(
                token_id=token_id,
                token=model.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                logprob=_emitted_logprob(raw[index] if raw and index < len(raw) else None, token_id),  # type: ignore[arg-type,index]
                top_logprobs=_top_logprobs(model, raw, index, limit),
            )
        )
    return rows


def _collapse_diagnostics(raw: list[dict[str, Any]]) -> SourceSteerInterventionDiagnostics:
    steps: list[int] = []
    norms: list[float] = []
    activations: dict[str, list[float]] = {}
    encoded = False
    active = False
    edit_count = 0
    for item in raw:
        offset = int(item.get("row_offset") or 0)
        count = int(item.get("num_rows") or 0)
        steps.extend(range(offset, offset + count))
        item_norms = [float(x) for x in item.get("perturbation_norms") or []]
        norms.extend(item_norms)
        edit_count += int(item.get("edit_count") or 0)
        encoded = encoded or bool(item.get("encoded"))
        active = active or bool(item.get("active"))
        for key, values in (item.get("feature_activations") or {}).items():
            activations.setdefault(str(key), []).extend(float(v) for v in values)
    return SourceSteerInterventionDiagnostics(
        edited_prediction_steps=steps,
        edit_count=edit_count,
        perturbation_norms=norms,
        feature_activations=activations,
        encoded=encoded,
        active=active,
    )


def _resolve_source_metadata(request: SourceSteerRequest) -> tuple[dict[str, Any] | None, SourceSteerResolvedMetadata]:
    manager = SAEManager.get_instance()
    steering = request.steering
    if steering is None or not steering.features:
        return None, SourceSteerResolvedMetadata()
    if not _source_set_contains(manager, steering.source_set, steering.source):
        raise ValueError(f"source {steering.source!r} is not in sourceSet {steering.source_set!r}")
    manager.ensure_source(steering.source)
    d_sae = manager.get_d_sae(steering.source)
    if d_sae is None:
        raise ValueError(f"source {steering.source!r} is not an SAE-backed source")
    validate_feature_interventions(_operation_payload(request), d_sae)
    worker_features = _worker_feature_payload(request)
    hook_name = manager.get_sae_hook(steering.source)
    address = tlens_hook_to_point(hook_name)
    if address.name != "resid_post" or address.layer is None:
        raise ValueError(f"/v1/steer/source currently supports resid_post SAE hooks only, got {hook_name!r}")
    release = manager.sae_id_to_release.get(steering.source) or manager.sae_data[steering.source].get("release")
    if not release:
        raise ValueError(f"source {steering.source!r} has no SAELens release recorded")
    saelens_id = str(manager.sae_data[steering.source].get("saelens_id") or steering.source)
    payload = {
        "layer": int(address.layer),
        "point": str(address.name),
        "position_policy": steering.position_policy.value,
        "features": worker_features,
        "return_diagnostics": bool(request.return_intervention_diagnostics),
        "sae": {
            "model": request.model,
            "release": str(release),
            "sae_id": saelens_id,
            "dtype": Config.get_instance().sae_dtype,
        },
    }
    decoder_vectors = _additive_decoder_vectors(request, manager, str(release), saelens_id)
    if decoder_vectors is not None:
        payload["decoder_vectors"] = decoder_vectors
    resolved = SourceSteerResolvedMetadata(
        source=steering.source,
        source_set=steering.source_set,
        saelens_release=str(release),
        saelens_id=saelens_id,
        hook_name=hook_name,
        hook_point=address.name,
        hook_layer=int(address.layer),
        position_policy=steering.position_policy,
        features=[SourceSteerFeature.model_validate(feature) for feature in worker_features],
    )
    if not worker_features:
        return None, resolved
    return payload, resolved


@router.post("/steer/source", responses={200: {"model": SourceSteerResponse}})
@with_request_lock(exclusive=False, cost=steer_source_cost)
async def steer_source(request: SourceSteerRequest):
    config = Config.get_instance()
    config.check_requested_model(request.model)
    model = Model.get_instance()
    if not isinstance(model, VLLMModel):
        return JSONResponse(content={"error": "/v1/steer/source requires the vLLM backend"}, status_code=400)
    if int(getattr(model, "tensor_parallel_size", 1) or 1) != 1:
        return JSONResponse(
            content={"error": "/v1/steer/source is currently verified only for single-GPU vLLM"}, status_code=400
        )
    try:
        tokenizer = model.tokenizer
        valid_ids = _tokenizer_ids(tokenizer)
        insertion = request.insertion
        input_ids = [int(x) for x in request.prompt_token_ids[0]]
        if insertion:
            model_ids, _alignment, _input_positions = _apply_insertion(input_ids, insertion, tokenizer, valid_ids, 0)
        else:
            _validate_ids(input_ids, valid_ids, 0, "promptTokenIds")
            model_ids, _alignment, _input_positions = input_ids, [], list(range(len(input_ids)))
        too_many = len(model_ids) > config.token_limit
        if too_many:
            return JSONResponse(
                content={"error": f"Prompt is too long: {len(model_ids)} tokens, max is {config.token_limit}"},
                status_code=400,
            )
        max_new_tokens, no_room = resolve_max_new_tokens(len(model_ids), int(request.max_new_tokens))
        if no_room is not None:
            return no_room
        spec, resolved = _resolve_source_metadata(request)
        if spec is not None:
            try:
                assert_steering_available(model, "SAE source steering")
            except BackendUnsupported as exc:
                return JSONResponse(content={"error": str(exc)}, status_code=400)
            if declares_static_taps(model):
                try:
                    assert_steer_layers_declared(model, [int(spec["layer"])], "SAE source steering")
                except BackendUnsupported as exc:
                    return JSONResponse(content={"error": str(exc)}, status_code=400)
                return JSONResponse(
                    content={"error": "/v1/steer/source currently requires hooked vLLM, not static CUDA-graph writes"},
                    status_code=400,
                )
            if _uses_speculative_decode(model):
                return JSONResponse(
                    content={"error": "/v1/steer/source does not currently support speculative decoding"},
                    status_code=400,
                )
        await model._ensure_engine()
        rid = model._new_request_id("np-sae-steer" if spec is not None else "np-sae-base")
        # The baseline takes private KV too. A steered request must (its KV is computed from an
        # edited residual), and a baseline that could hit the prefix cache would have its prompt
        # KV come from some earlier request's forward instead of its own, which is one more
        # difference between the two arms of a comparison than the intervention itself.
        prompt = model._prompt(model_ids, private_kv_for=rid)
        sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=float(request.temperature),
            logprobs=int(request.top_logprobs) or None,
        )
        registration_task: asyncio.Task[Any] | None = None
        raw_diag: list[dict[str, Any]] = []
        try:
            if spec is not None:
                registration_task = asyncio.create_task(
                    model.engine.collective_rpc("register_sae_feature_steering", args=(rid, [spec], len(model_ids)))
                )
                await asyncio.shield(registration_task)
            output = await model._run_one(prompt, sampling, request_id=rid)
            completion = output.outputs[0]
            if spec is not None and request.return_intervention_diagnostics:
                payload = await model.engine.collective_rpc("collect_sae_feature_steering_diagnostics", args=(rid,))
                raw_diag = payload[0] if isinstance(payload, list | tuple) else payload
        finally:
            if spec is not None:
                if registration_task is not None and not registration_task.done():
                    with contextlib.suppress(Exception):
                        await asyncio.shield(registration_task)
                await asyncio.shield(model.engine.collective_rpc("unregister_sae_feature_steering", args=(rid,)))
        generated_ids = [int(x) for x in completion.token_ids]
        response = SourceSteerResponse(
            model_input_token_ids=model_ids,
            generated_token_ids=generated_ids,
            generated_text=str(completion.text),
            finish_reason=None if completion.finish_reason is None else str(completion.finish_reason),
            logprobs=_format_logprobs(model, generated_ids, completion.logprobs, int(request.top_logprobs)),
            resolved=resolved,
            intervention_diagnostics=_collapse_diagnostics(raw_diag)
            if request.return_intervention_diagnostics
            else None,
        )
        return response.model_dump(exclude_none=True)
    except (BackendUnsupported, ValueError) as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        if recover_from_oom(exc):
            return JSONResponse(content={"error": str(RecoverableOutOfMemory())}, status_code=503)
        logger.exception("Error processing steer/source")
        return JSONResponse(content={"error": "An error occurred while processing the request"}, status_code=500)
