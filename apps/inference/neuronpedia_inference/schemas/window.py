"""Contracts for full-vocabulary KL on fixed-history code windows."""

from typing import Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from neuronpedia_inference.schemas.common import ExactSchema
from neuronpedia_inference.schemas.steer import SourceSteeringSpec, SourceSteerResolvedMetadata


class SourceWindowScoreRequest(ExactSchema):
    """Plain token window; steering follows actual activation throughout it."""

    model: StrictStr
    token_ids: list[StrictInt] = Field(min_length=2, max_length=512)
    steering: SourceSteeringSpec
    include_single_pulse: StrictBool = True
    include_baseline_repeat: StrictBool = True
    trace_feature_indices: list[StrictInt] = Field(default_factory=list, max_length=1024)
    fail_if_busy: StrictBool = False

    @model_validator(mode="after")
    def check_single_condition(self) -> "SourceWindowScoreRequest":
        if len(self.steering.features) != 1:
            raise ValueError("window scoring requires exactly one selected feature")
        feature = self.steering.features[0]
        if (
            feature.operation.value != "scale"
            or feature.magnitude_mode != "norm_relative"
            or feature.cooldown_mode != "feature_inactive"
            or feature.cooldown_tokens != 0
            or self.steering.position_policy.value != "each_generated_token"
        ):
            raise ValueError("window scoring requires the v5 residual-relative feature_inactive condition")
        if any(i < 0 for i in self.trace_feature_indices):
            raise ValueError("trace feature indices must be nonnegative")
        return self


class SourceWindowScoreArm(ExactSchema):
    """One distribution comparison, indexed by input prediction position."""

    arm: Literal["feature_inactive", "single_pulse", "baseline_repeat"]
    kl: list[StrictFloat]
    top1_agree: list[StrictBool]
    feature_activations: list[StrictFloat]
    residual_norms: list[StrictFloat]
    gate_feature_activations: dict[StrictStr, list[StrictFloat]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    perturbation_norms: list[StrictFloat]
    pulse_positions: list[StrictInt]


class SourceWindowScoreResponse(ExactSchema):
    """Position t predicts provided token t+1; the last input row is excluded."""

    model_input_token_ids: list[StrictInt]
    prediction_positions: list[StrictInt]
    resolved: SourceSteerResolvedMetadata
    kl_method: Literal["full_vocabulary_forward_float32"] = "full_vocabulary_forward_float32"
    baseline_feature_activations: dict[StrictStr, list[StrictFloat]]
    baseline_residual_norms: list[StrictFloat]
    arms: list[SourceWindowScoreArm]
