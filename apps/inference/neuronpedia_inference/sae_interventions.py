"""SAE feature-intervention math shared by endpoints and vLLM worker hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

InterventionOperation = Literal["add", "scale", "ablate"]


@dataclass(frozen=True)
class SAEFeatureIntervention:
    feature_index: int
    operation: InterventionOperation
    value: float | None = None


@dataclass(frozen=True)
class InterventionDiagnostics:
    edited_rows: int
    edit_count: int
    perturbation_norms: list[float]
    feature_activations: dict[int, list[float]]
    encoded: bool
    active: bool


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
            or (feature.operation == "scale" and float(feature.value or 0.0) == 1.0)
        )
    ]
    if not effective:
        diag = InterventionDiagnostics(0, 0, [], {}, encoded=False, active=False) if return_diagnostics else None
        return torch.zeros_like(rows), diag

    needs_encode = any(feature.operation in {"scale", "ablate"} for feature in effective)
    if needs_encode and sae is None:
        raise ValueError("scale and ablate require a loaded SAE encoder")
    acts = _encode(sae, rows) if needs_encode else None
    coeffs = torch.zeros((rows.shape[0], len(effective)), device=rows.device, dtype=torch.float32)
    activations_for_diag: dict[int, list[float]] = {}
    for col, feature in enumerate(effective):
        if feature.operation == "add":
            if feature.value is None:
                raise ValueError("add requires a finite value")
            coeffs[:, col] = float(feature.value)
            continue
        assert acts is not None
        feature_acts = acts[:, feature.feature_index].to(rows.device, dtype=torch.float32)
        activations_for_diag[feature.feature_index] = [float(x) for x in feature_acts.detach().cpu()]
        if feature.operation == "ablate":
            multiplier = 0.0
        else:
            if feature.value is None:
                raise ValueError("scale requires a finite nonnegative value")
            multiplier = float(feature.value)
        coeffs[:, col] = (multiplier - 1.0) * feature_acts

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
    delta = (coeffs @ vectors).to(dtype=rows.dtype)
    if not return_diagnostics:
        return delta, None
    norms = [float(x) for x in delta.float().norm(dim=-1).detach().cpu()]
    active = bool(torch.any(delta != 0).item())
    return delta, InterventionDiagnostics(
        edited_rows=int(rows.shape[0]),
        edit_count=len(effective) * int(rows.shape[0]),
        perturbation_norms=norms,
        feature_activations=activations_for_diag,
        encoded=needs_encode,
        active=active,
    )
