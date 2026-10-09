"""Fixed-history code-window scoring inside the vLLM worker.

Capture final_norm, then use the model's actual compute_logits (no second norm or
lens approximation). References stay request-scoped on CPU; vocabulary-sized
work is limited to 16 prediction rows. The sampled extra token is never scored.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from interp_engine.vllm_capture._demux import _ensure_dev, _get_demux, _maybe_unregister, _release_hook
from interp_engine.vllm_capture._payload import hook_site
from interp_engine.vllm_capture._tree import _worker_model
from interp_engine.vllm_capture.requests import _ensure_hook, _write_site, worker_register_capture

from neuronpedia_inference.sae_interventions import PulseState, sae_intervention_delta
from neuronpedia_inference.vllm_sae_worker import _features, _load_worker_sae

LOGIT_ROWS = 16


def full_kl(baseline: torch.Tensor, steered: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary forward KL in nats, with float32 normalization.

    Shared masked-out vocabulary entries contribute zero. Non-finite KL or
    negative values beyond floating-point roundoff fail rather than disappear.
    """
    if baseline.shape != steered.shape or baseline.dim() != 2:
        raise ValueError("KL requires matching [positions, vocabulary] logits")
    p = torch.log_softmax(baseline.float(), dim=-1)
    q = torch.log_softmax(steered.float(), dim=-1)
    terms = torch.where(torch.isneginf(p), torch.zeros_like(p), p.exp() * (p - q))
    kl = terms.sum(dim=-1)
    if not bool(torch.isfinite(kl).all()) or bool((kl < -1e-5).any()):
        raise ValueError("KL is non-finite or negative beyond roundoff")
    return kl.clamp_min(0)


class WindowModifier:
    """One window's pulse state, advanced over consecutive prefill chunks."""

    def __init__(self, spec: dict[str, Any], sae: object, token_count: int, arm: str) -> None:
        self.sae = sae
        self.features = _features(spec)
        if arm == "single_pulse":
            # Within this finite window, this is exactly at most one pulse.
            self.features = [replace(f, cooldown_mode="fixed", cooldown_tokens=token_count) for f in self.features]
        self.state = PulseState()
        self.position = 0
        self.stop = token_count - 1  # Last input predicts outside the window.
        self.activations: list[float] = []
        self.gate_activations: dict[str, list[float]] = {
            str(i): [] for f in self.features for i in f.gate_feature_indices
        }
        self.residual_norms: list[float] = []
        self.perturbation_norms: list[float] = []
        self.pulses: list[int] = []
        self.invalid_layout = False

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        start = self.position
        self.position += int(rows.shape[0])
        delta = torch.zeros_like(rows)
        count = min(int(rows.shape[0]), max(0, self.stop - start))
        if self.position > self.stop + 1:
            # Never throw from a model hook: report replay/layout problems after
            # the forward, so a bad scoring request cannot crash the engine.
            self.invalid_layout = True
        if not count:
            return delta
        selected = rows[:count]
        edited, diag = sae_intervention_delta(
            selected, self.sae, self.features, return_diagnostics=True, pulse_state=self.state, row_offset=start
        )
        assert diag is not None
        index = self.features[0].feature_index
        self.activations.extend(diag.feature_activations[index])
        for helper_index, values in self.gate_activations.items():
            values.extend(diag.feature_activations[int(helper_index)])
        self.residual_norms.extend(selected.float().norm(dim=-1).detach().cpu().tolist())
        self.perturbation_norms.extend(diag.perturbation_norms)
        self.pulses.extend(diag.feature_edit_steps.get(index, []))
        delta[:count] = edited
        return delta


def _stores(worker: Any) -> tuple[dict[str, Any], dict[str, WindowModifier]]:
    if not hasattr(worker, "_np_window_references"):
        worker._np_window_references = {}
        worker._np_window_modifiers = {}
    return worker._np_window_references, worker._np_window_modifiers


def register_window(worker: Any, req_id: str, spec: dict[str, Any], token_count: int, arm: str) -> None:
    """Register final-output capture and, for steered arms, all-window edits."""
    if arm not in {"baseline", "baseline_repeat", "feature_inactive", "single_pulse"}:
        raise ValueError("unsupported code-window arm")
    points = ["final_norm"]
    if arm == "baseline":
        points.append(f"resid_post.{spec['layer']}")
    worker_register_capture(worker, req_id, points)
    if arm in {"feature_inactive", "single_pulse"}:
        demux = _get_demux(worker)
        _ensure_dev(worker, demux)
        assert demux.dev is not None and demux.dt is not None
        modifier = WindowModifier(spec, _load_worker_sae(spec["sae"], demux.dev, demux.dt), token_count, arm)
        _stores(worker)[1][req_id] = modifier
        site = _write_site(worker, spec)
        demux.steer_mods.setdefault(req_id, {})[site] = (modifier, set(), token_count)
        _ensure_hook(worker, demux, site)


def _take_rows(worker: Any, req_id: str, token_count: int) -> dict[str, torch.Tensor]:
    demux = _get_demux(worker)
    caps = demux.captures.pop(req_id, {})
    points = demux.cap_points.pop(req_id, set())
    for site in {hook_site(a) for a in points}:
        _release_hook(demux, site)
    _maybe_unregister(demux, req_id)
    rows = {key: torch.cat(tensors, dim=0) for key, tensors in caps.items() if tensors}
    if not rows or any(t.dim() != 2 or t.shape[0] != token_count for t in rows.values()):
        raise ValueError("window capture must contain every input row exactly once; missing rows or replay detected")
    return {key: tensor[:-1] for key, tensor in rows.items()}


def store_window_reference(
    worker: Any, req_id: str, spec: dict[str, Any], token_count: int, trace_indices: list[int]
) -> dict[str, Any]:
    """Store normalized baseline states and return sparse selected-feature traces."""
    rows = _take_rows(worker, req_id, token_count)
    normed = rows["final_norm"].detach().cpu()
    residual = rows[f"resid_post.{spec['layer']}"]
    sae = _load_worker_sae(spec["sae"], residual.device, residual.dtype)
    selected_feature = spec["features"][0]
    indices = sorted(
        set(trace_indices + [int(selected_feature["feature_index"])] + selected_feature.get("gate_feature_indices", []))
    )
    traces: dict[str, list[float]] = {str(i): [] for i in indices}
    with torch.no_grad():
        for part in residual.split(LOGIT_ROWS):
            encoder: Any = sae
            acts = encoder.encode(part)
            selected = acts[:, indices].float().cpu().T.tolist()
            for i, values in zip(indices, selected, strict=True):
                traces[str(i)].extend(values)
    _stores(worker)[0][req_id] = normed
    return {"feature_activations": traces, "residual_norms": residual.float().norm(dim=-1).cpu().tolist()}


def compare_window(worker: Any, req_id: str, reference_id: str, token_count: int, arm: str) -> dict[str, Any]:
    """Compare actual output-head distributions without exporting vocabularies."""
    normed = _take_rows(worker, req_id, token_count)["final_norm"]
    reference = _stores(worker)[0][reference_id]
    if normed.shape != reference.shape:
        raise ValueError("baseline and steered window shapes differ")
    model: Any = _worker_model(worker)
    kl: list[float] = []
    agree: list[bool] = []
    with torch.no_grad():
        for start in range(0, len(normed), LOGIT_ROWS):
            p = model.compute_logits(reference[start : start + LOGIT_ROWS].to(normed.device, normed.dtype))
            q = model.compute_logits(normed[start : start + LOGIT_ROWS])
            kl.extend(full_kl(p, q).cpu().tolist())
            agree.extend((p.argmax(dim=-1) == q.argmax(dim=-1)).cpu().tolist())
    modifier = _stores(worker)[1].get(req_id)
    if modifier is not None and (modifier.invalid_layout or len(modifier.activations) != token_count - 1):
        raise ValueError("window pulse trace is incomplete or was replayed")
    return {
        "arm": arm,
        "kl": kl,
        "top1_agree": agree,
        "feature_activations": [] if modifier is None else modifier.activations,
        "residual_norms": [] if modifier is None else modifier.residual_norms,
        "gate_feature_activations": {} if modifier is None else modifier.gate_activations,
        "perturbation_norms": [] if modifier is None else modifier.perturbation_norms,
        "pulse_positions": [] if modifier is None else modifier.pulses,
    }


def cleanup_window(worker: Any, req_id: str) -> None:
    """Idempotently release registrations and CPU references, including failed arms."""
    demux = _get_demux(worker)
    demux.captures.pop(req_id, None)
    for site in {hook_site(a) for a in demux.cap_points.pop(req_id, set())}:
        _release_hook(demux, site)
    for site in demux.steer_mods.pop(req_id, {}):
        _release_hook(demux, site)
    references, modifiers = _stores(worker)
    references.pop(req_id, None)
    modifiers.pop(req_id, None)
    _maybe_unregister(demux, req_id)
