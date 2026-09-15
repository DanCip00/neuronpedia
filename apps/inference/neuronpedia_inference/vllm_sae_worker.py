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


def _load_worker_sae(spec: dict[str, Any], device: torch.device, dtype: torch.dtype) -> object:
    sae, _hook = SaeLensSAE.load(
        str(spec["release"]),
        str(spec["sae_id"]),
        str(device),
        str(spec.get("dtype") or str(dtype).removeprefix("torch.")),
    )
    return sae


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
    prefill_seen = False
    next_decode_position = prompt_len

    def _modify(full: torch.Tensor) -> torch.Tensor:
        nonlocal next_decode_position, prefill_seen
        delta = torch.zeros_like(full)
        if full.dim() != 2:
            raise ValueError(f"SAE source steering supports 2-D residual rows only, got {tuple(full.shape)}")
        if not prefill_seen:
            if full.shape[0] != prompt_len:
                raise ValueError(
                    "SAE source steering requires an unchunked initial prefill: "
                    f"expected {prompt_len} request rows, received {full.shape[0]}"
                )
            prefill_seen = True
            is_prefill = True
            rows = full[-1:]
            row_offset = prompt_len - 1
        elif policy == "each_generated_token":
            is_prefill = False
            rows = full
            row_offset = next_decode_position
            next_decode_position += int(rows.shape[0])
        else:
            return delta
        row_delta, diagnostics = sae_intervention_delta(
            rows,
            sae,
            features,
            return_diagnostics=want_diag,
            decoder_vectors_by_feature=decoder_vectors,
        )
        if is_prefill:
            delta[-1:] = row_delta
        else:
            delta[:] = row_delta
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
