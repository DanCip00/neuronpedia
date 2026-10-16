"""SAE feature-intervention math shared by endpoints and vLLM worker hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import torch

InterventionOperation = Literal["add", "scale", "ablate"]


@dataclass(frozen=True)
class SAEFeatureIntervention:
    feature_index: int
    operation: InterventionOperation
    value: float | None = None
    magnitude_mode: Literal["activation", "norm_relative"] = "activation"
    cooldown_tokens: int = 0
    cooldown_mode: Literal["fixed", "feature_inactive"] = "fixed"
    activation_threshold: float = 0.0
    threshold_mode: Literal["activation", "residual_relative"] = "activation"
    gate_feature_indices: tuple[int, ...] = ()
    gate_feature_thresholds: tuple[float, ...] = ()

    @property
    def pulsed(self) -> bool:
        return (
            self.magnitude_mode == "norm_relative"
            or self.cooldown_tokens > 0
            or self.cooldown_mode == "feature_inactive"
        )


@dataclass
class PulseState:
    """One request's per-feature refractory state, never shared with the SAE cache."""

    last_pulse: dict[int, int] = field(default_factory=dict)
    blocked: set[int] = field(default_factory=set)

    def allow(
        self, feature: SAEFeatureIntervention, activation: float, position: int, detector_accepts: bool = True
    ) -> bool:
        """Delay within the original feature episode; detector dropout never rearms it."""
        index = feature.feature_index
        if activation <= 0:
            self.blocked.discard(index)
            return False
        if not detector_accepts:
            return False
        if feature.cooldown_mode == "feature_inactive":
            if index in self.blocked:
                return False
        elif position <= self.last_pulse.get(index, -feature.cooldown_tokens - 1) + feature.cooldown_tokens:
            return False
        self.last_pulse[index] = position
        self.blocked.add(index)
        return True


@dataclass(frozen=True)
class InterventionDiagnostics:
    edited_rows: int
    edit_count: int
    perturbation_norms: list[float]
    feature_activations: dict[int, list[float]]
    encoded: bool
    active: bool
    feature_edit_steps: dict[int, list[int]] = field(default_factory=dict)
    residual_norms: list[float] = field(default_factory=list)


def validate_feature_interventions(features: list[SAEFeatureIntervention], d_sae: int | None) -> None:
    seen: set[int] = set()
    for feature in features:
        if type(feature.feature_index) is not int:
            raise ValueError("featureIndex must be an integer")
        if feature.feature_index in seen:
            raise ValueError(f"Duplicate featureIndex {feature.feature_index} in one steering source")
        seen.add(feature.feature_index)
        if feature.feature_index < 0 or (d_sae is not None and feature.feature_index >= d_sae):
            if d_sae is None:
                raise ValueError(f"featureIndex {feature.feature_index} must be nonnegative")
            raise ValueError(f"featureIndex {feature.feature_index} is outside [0, {d_sae})")
        if not isfinite(feature.activation_threshold) or feature.activation_threshold < 0:
            raise ValueError("activationThreshold must be finite and nonnegative")
        if feature.threshold_mode not in {"activation", "residual_relative"}:
            raise ValueError("Unsupported thresholdMode")
        helpers = feature.gate_feature_indices
        if len(set(helpers)) != len(helpers):
            raise ValueError("gateFeatureIndices must be unique")
        for helper in helpers:
            if type(helper) is not int or helper < 0 or (d_sae is not None and helper >= d_sae):
                raise ValueError("gateFeatureIndices must contain valid nonnegative SAE feature indices")
            if helper == feature.feature_index:
                raise ValueError("gateFeatureIndices must not include the edited feature")
        cutoffs = feature.gate_feature_thresholds
        if cutoffs and len(cutoffs) != len(helpers):
            raise ValueError("gateFeatureThresholds must align with gateFeatureIndices")
        if any(type(value) not in (int, float) or not isfinite(value) or value < 0 for value in cutoffs):
            raise ValueError("gateFeatureThresholds must be finite nonnegative numbers")
        if feature.operation == "add" and (
            feature.activation_threshold > 0 or helpers or feature.threshold_mode != "activation"
        ):
            raise ValueError("Detector gates require an activation-dependent operation")
        if feature.magnitude_mode not in {"activation", "norm_relative"}:
            raise ValueError("Unsupported magnitudeMode")
        if feature.cooldown_mode not in {"fixed", "feature_inactive"}:
            raise ValueError("Unsupported cooldownMode")
        if type(feature.cooldown_tokens) is not int or feature.cooldown_tokens < 0:
            raise ValueError("cooldownTokens must be a nonnegative integer")
        if feature.cooldown_mode == "feature_inactive" and feature.cooldown_tokens != 0:
            raise ValueError("feature_inactive cooldown must omit cooldownTokens or set it to zero")
        if feature.operation == "add" and feature.pulsed:
            raise ValueError("Pulse steering requires an activation-dependent operation")
        if feature.magnitude_mode == "norm_relative" and (
            feature.operation != "scale" or feature.value is None or not 0 < feature.value <= 1
        ):
            raise ValueError("norm_relative requires scale with rho in (0, 1]")
        if feature.operation == "add":
            if feature.value is None or not isfinite(float(feature.value)):
                raise ValueError("add requires a finite value")
        elif feature.operation == "scale":
            if feature.value is None or not isfinite(float(feature.value)) or float(feature.value) < 0:
                raise ValueError("scale requires a finite nonnegative value")
        elif feature.operation == "ablate":
            if feature.value is not None:
                raise ValueError("ablate does not take a value")
        else:
            raise ValueError(f"Unsupported intervention operation {feature.operation!r}")


def decoder_matrix(sae: object) -> torch.Tensor:
    """Return W_dec as ``[d_sae, d_in]`` in the SAE's native decoder units."""
    matrix = getattr(sae, "W_dec", None)
    if isinstance(matrix, torch.Tensor):
        return matrix
    decoder = getattr(sae, "decoder", None)
    weight = getattr(decoder, "weight", None)
    if isinstance(weight, torch.Tensor):
        d_sae = int(getattr(getattr(sae, "cfg", object()), "d_sae", weight.shape[0]))
        return weight if weight.shape[0] == d_sae else weight.T
    raise ValueError(f"{type(sae).__name__} does not expose a supported decoder matrix")


def _encode(sae: object, rows: torch.Tensor) -> torch.Tensor:
    encode = getattr(sae, "encode", None)
    if not callable(encode):
        raise ValueError(f"{type(sae).__name__} does not expose encode()")
    with torch.no_grad():
        activations = encode(rows)
    if not isinstance(activations, torch.Tensor):
        raise ValueError(f"{type(sae).__name__}.encode() did not return a tensor")
    if activations.dim() != 2 or activations.shape[0] != rows.shape[0]:
        raise ValueError(
            f"{type(sae).__name__}.encode() returned shape {tuple(activations.shape)} for rows {tuple(rows.shape)}"
        )
    return activations


def sae_intervention_delta(
    rows: torch.Tensor,
    sae: object | None,
    features: list[SAEFeatureIntervention],
    *,
    return_diagnostics: bool = False,
    decoder_vectors_by_feature: Mapping[int, torch.Tensor] | None = None,
    pulse_state: PulseState | None = None,
    row_offset: int = 0,
) -> tuple[torch.Tensor, InterventionDiagnostics | None]:
    """Compute a hidden-state delta while preserving the SAE reconstruction residual."""
    if rows.dim() != 2:
        raise ValueError(f"SAE interventions expect [rows, d_in], got {tuple(rows.shape)}")

    d_sae = None
    if sae is not None:
        d_sae = int(decoder_matrix(sae).shape[0])
    validate_feature_interventions(features, d_sae)
    effective = [
        feature
        for feature in features
        if not (
            (feature.operation == "add" and float(feature.value or 0.0) == 0.0)
            or (
                feature.operation == "scale"
                and feature.magnitude_mode == "activation"
                and float(feature.value or 0.0) == 1.0
            )
        )
    ]
    if not effective:
        diag = InterventionDiagnostics(0, 0, [], {}, encoded=False, active=False) if return_diagnostics else None
        return torch.zeros_like(rows), diag

    if any(feature.pulsed for feature in effective) and pulse_state is None:
        raise ValueError("Pulse steering requires request-local pulse_state")
    needs_encode = any(feature.operation in {"scale", "ablate"} for feature in effective)
    if needs_encode and sae is None:
        raise ValueError("scale and ablate require a loaded SAE encoder")
    acts = _encode(sae, rows) if needs_encode else None
    coeffs = torch.zeros((rows.shape[0], len(effective)), device=rows.device, dtype=torch.float32)
    activations_for_diag: dict[int, list[float]] = {}
    edit_masks = torch.ones_like(coeffs, dtype=torch.bool)
    residual_norms = rows.float().norm(dim=-1)

    for col, feature in enumerate(effective):
        if feature.operation == "add":
            if feature.value is None:
                raise ValueError("add requires a finite value")
            coeffs[:, col] = float(feature.value)
            continue
        assert acts is not None
        feature_acts = acts[:, feature.feature_index].to(rows.device, dtype=torch.float32)
        activations_for_diag[feature.feature_index] = [float(x) for x in feature_acts.detach().cpu()]
        detector_mask = feature_acts > 0
        scores = feature_acts
        if feature.threshold_mode == "residual_relative":
            detector_mask &= residual_norms > 0
            scores = torch.where(residual_norms > 0, feature_acts / residual_norms.clamp_min(1e-30), 0.0)
        detector_mask &= scores >= feature.activation_threshold
        cutoffs = feature.gate_feature_thresholds or (0.0,) * len(feature.gate_feature_indices)
        for helper_index, cutoff in zip(feature.gate_feature_indices, cutoffs, strict=True):
            helper_acts = acts[:, helper_index].to(rows.device, dtype=torch.float32)
            activations_for_diag[helper_index] = [float(x) for x in helper_acts.detach().cpu()]
            detector_mask &= (helper_acts > 0) & (helper_acts >= cutoff)
        if feature.pulsed:
            assert pulse_state is not None
            edit_masks[:, col] = torch.tensor(
                [
                    pulse_state.allow(feature, float(a), row_offset + i, bool(accepts))
                    for i, (a, accepts) in enumerate(zip(feature_acts.detach().cpu(), detector_mask.detach().cpu()))
                ],
                device=rows.device,
                dtype=torch.bool,
            )
        elif feature.activation_threshold > 0 or feature.gate_feature_indices or feature.threshold_mode != "activation":
            edit_masks[:, col] = detector_mask
        if feature.magnitude_mode == "norm_relative":
            assert feature.value is not None
            coeffs[:, col] = float(feature.value) * residual_norms * edit_masks[:, col]
            continue
        if feature.operation == "ablate":
            multiplier = 0.0
        else:
            if feature.value is None:
                raise ValueError("scale requires a finite nonnegative value")
            multiplier = float(feature.value)
        coeffs[:, col] = (multiplier - 1.0) * feature_acts * edit_masks[:, col]

    if decoder_vectors_by_feature is not None:
        try:
            vectors = torch.stack([decoder_vectors_by_feature[f.feature_index] for f in effective])
        except KeyError as exc:
            raise ValueError(f"Missing cached decoder vector for featureIndex {exc.args[0]}") from exc
    else:
        if sae is None:
            raise ValueError("add requires either a loaded SAE or cached decoder vectors")
        w_dec = decoder_matrix(sae)
        feature_ids = torch.tensor([f.feature_index for f in effective], device=w_dec.device, dtype=torch.long)
        vectors = w_dec.index_select(0, feature_ids)
    if vectors.dim() != 2 or vectors.shape[1] != rows.shape[-1]:
        raise ValueError(
            f"Decoder vectors have shape {tuple(vectors.shape)} but hidden rows have width {rows.shape[-1]}"
        )
    vectors = vectors.to(rows.device, dtype=torch.float32)
    # Normalize only residual-relative directions. Legacy edits retain native SAE units.
    for col, feature in enumerate(effective):
        if feature.magnitude_mode == "norm_relative":
            norm = vectors[col].norm()
            if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0:
                raise ValueError("norm_relative requires a finite nonzero decoder direction")
            vectors[col] = vectors[col] / norm
    delta = (coeffs @ vectors).to(dtype=rows.dtype)
    if not return_diagnostics:
        return delta, None
    norms = [float(x) for x in delta.float().norm(dim=-1).detach().cpu()]
    active = bool(torch.any(delta != 0).item())
    return delta, InterventionDiagnostics(
        edited_rows=int(rows.shape[0]),
        edit_count=int(edit_masks.sum().item()),
        perturbation_norms=norms,
        feature_activations=activations_for_diag,
        encoded=needs_encode,
        active=active,
        residual_norms=residual_norms.detach().cpu().tolist(),
        feature_edit_steps={
            f.feature_index: [
                row_offset + i for i, enabled in enumerate(edit_masks[:, col].detach().cpu()) if bool(enabled)
            ]
            for col, f in enumerate(effective)
        },
    )
