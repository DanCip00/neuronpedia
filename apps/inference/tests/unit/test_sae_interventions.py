from __future__ import annotations

import pytest
import torch

from neuronpedia_inference.sae_interventions import PulseState, SAEFeatureIntervention, sae_intervention_delta


class TinyTopKSAE:
    def __init__(self) -> None:
        self.W_dec = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.W_enc = torch.tensor(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        self.encode_calls = 0

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        pre = torch.relu(rows @ self.W_enc)
        values, indices = torch.topk(pre, k=2, dim=-1)
        out = torch.zeros_like(pre)
        return out.scatter(1, indices, values)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec


def test_add_uses_decoder_direction_without_encoding() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[0.0, 0.0, 0.0]])
    delta, diag = sae_intervention_delta(
        rows,
        sae,
        [SAEFeatureIntervention(2, "add", -0.5)],
        return_diagnostics=True,
    )
    assert sae.encode_calls == 0
    torch.testing.assert_close(delta, torch.tensor([[0.0, 0.0, -1.5]]))
    assert diag is not None and diag.encoded is False and diag.active is True


def test_scale_and_ablate_match_reference_residual_preserving_formulation() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 1.0, 1.0]])
    features = [
        SAEFeatureIntervention(0, "scale", 0.5),
        SAEFeatureIntervention(2, "ablate"),
        SAEFeatureIntervention(3, "add", 2.0),
    ]
    delta, _diag = sae_intervention_delta(rows, sae, features)
    acts = sae.encode(rows)
    edited = acts.clone()
    edited[:, 0] *= 0.5
    edited[:, 2] = 0.0
    reference = sae.decode(edited) - sae.decode(acts) + 2.0 * sae.W_dec[3]
    torch.testing.assert_close(delta, reference)
    torch.testing.assert_close(rows + delta, rows + sae.decode(edited) - sae.decode(acts) + 2.0 * sae.W_dec[3])


def test_scale_uses_true_topk_activation_not_positive_preactivation() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 1.0, 1.0]])
    preactivation_feature_1 = torch.relu(rows @ sae.W_enc)[0, 1]
    assert preactivation_feature_1 > 0
    assert sae.encode(rows)[0, 1] == 0
    delta, diag = sae_intervention_delta(
        rows,
        sae,
        [SAEFeatureIntervention(1, "scale", 10.0)],
        return_diagnostics=True,
    )
    assert sae.encode_calls == 2
    torch.testing.assert_close(delta, torch.zeros_like(rows))
    assert diag is not None
    assert diag.feature_activations == {1: [0.0]}
    assert diag.active is False


def test_noop_empty_features_avoids_encode() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 2.0, 3.0]])
    delta, diag = sae_intervention_delta(rows, sae, [], return_diagnostics=True)
    assert sae.encode_calls == 0
    torch.testing.assert_close(delta, torch.zeros_like(rows))
    assert diag is not None and diag.active is False


def test_explicit_noops_avoid_encode() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 2.0, 3.0]])
    features = [
        SAEFeatureIntervention(0, "add", 0.0),
        SAEFeatureIntervention(1, "scale", 1.0),
    ]

    delta, diag = sae_intervention_delta(rows, sae, features, return_diagnostics=True)

    assert sae.encode_calls == 0
    torch.testing.assert_close(delta, torch.zeros_like(rows))
    assert diag is not None
    assert diag.encoded is False
    assert diag.active is False
    assert diag.edited_rows == 0
    assert diag.edit_count == 0


def test_add_uses_cached_decoder_vectors_without_an_sae() -> None:
    rows = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    cached_vectors = {2: torch.tensor([0.0, 0.0, 3.0])}

    delta, diag = sae_intervention_delta(
        rows,
        None,
        [SAEFeatureIntervention(2, "add", -0.5)],
        return_diagnostics=True,
        decoder_vectors_by_feature=cached_vectors,
    )

    torch.testing.assert_close(delta, torch.tensor([[0.0, 0.0, -1.5], [0.0, 0.0, -1.5]]))
    assert diag is not None
    assert diag.encoded is False


def test_relative_pulse_normalizes_nonunit_decoder_and_gates_on_topk() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    feature = SAEFeatureIntervention(2, "scale", 0.25, "norm_relative", 1)
    delta, diag = sae_intervention_delta(rows, sae, [feature], pulse_state=PulseState(), return_diagnostics=True)
    torch.testing.assert_close(delta[0].norm(), rows[0].norm() * 0.25)
    torch.testing.assert_close(delta[1], torch.zeros(3))
    assert diag is not None and diag.edit_count == 1 and diag.feature_edit_steps == {2: [0]}
    silent, diag = sae_intervention_delta(
        rows,
        sae,
        [SAEFeatureIntervention(1, "scale", 0.5, "norm_relative")],
        pulse_state=PulseState(),
        return_diagnostics=True,
    )
    assert silent.count_nonzero() == 0
    assert diag is not None and diag.encoded and not diag.active and diag.edit_count == 0


def test_multi_pulses_use_the_same_unedited_residual_and_independent_cooldowns() -> None:
    sae = TinyTopKSAE()
    rows = torch.tensor([[1.0, 1.0, 1.0]] * 4)
    features = [
        SAEFeatureIntervention(0, "scale", 0.1, "norm_relative", 1),
        SAEFeatureIntervention(2, "scale", 0.5, "norm_relative", 2),
    ]
    delta, diag = sae_intervention_delta(rows, sae, features, pulse_state=PulseState(), return_diagnostics=True)
    scale = rows[0].norm()
    torch.testing.assert_close(
        delta, scale * torch.tensor([[0.1, 0.0, 0.5], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.5]])
    )
    assert diag is not None and diag.feature_edit_steps == {0: [0, 2], 2: [0, 3]}


@pytest.mark.parametrize(
    "feature",
    [
        SAEFeatureIntervention(0, "scale", 0.0, "norm_relative"),
        SAEFeatureIntervention(0, "scale", 1.1, "norm_relative"),
        SAEFeatureIntervention(0, "scale", float("nan"), "norm_relative"),
        SAEFeatureIntervention(0, "ablate", None, "norm_relative"),
        SAEFeatureIntervention(0, "add", 1.0, cooldown_tokens=2),
        SAEFeatureIntervention(0, "scale", 2.0, cooldown_tokens=-1),
        SAEFeatureIntervention(0, "scale", 2.0, cooldown_tokens=True),
        SAEFeatureIntervention(0, "scale", 2.0, cooldown_tokens=2, cooldown_mode="feature_inactive"),
    ],
)
def test_invalid_pulse_configuration_rejected(feature: SAEFeatureIntervention) -> None:
    with pytest.raises(ValueError):
        sae_intervention_delta(torch.ones(1, 3), TinyTopKSAE(), [feature], pulse_state=PulseState())


class IdentitySAE:
    W_dec = torch.eye(3)

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.relu()


def test_threshold_and_helpers_delay_without_rearming_on_detector_dropout() -> None:
    feature = SAEFeatureIntervention(
        0,
        "scale",
        0.25,
        "norm_relative",
        cooldown_mode="feature_inactive",
        activation_threshold=2.0,
        gate_feature_indices=(1,),
    )
    # Below-threshold onset, helper absent, first acceptance, both detector dropouts,
    # acceptance again within the spent episode, f disappears, and new acceptance.
    rows = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    delta, diag = sae_intervention_delta(
        rows, IdentitySAE(), [feature], pulse_state=PulseState(), return_diagnostics=True, row_offset=10
    )
    assert diag is not None and diag.feature_edit_steps == {0: [12, 17]}
    assert diag.edit_count == 2
    assert set(diag.feature_activations) == {0, 1}
    assert len(diag.residual_norms) == len(rows)
    # Helpers supply evidence only: the edit has no helper decoder component.
    assert torch.count_nonzero(delta[:, 1:]) == 0
    torch.testing.assert_close(delta[2, 0], rows[2].norm() * 0.25)


def test_residual_relative_detector_uses_preedit_norm_and_inclusive_threshold() -> None:
    feature = SAEFeatureIntervention(
        0,
        "scale",
        0.1,
        "norm_relative",
        activation_threshold=0.6,
        threshold_mode="residual_relative",
    )
    rows = torch.tensor([[3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    delta, diag = sae_intervention_delta(
        rows, IdentitySAE(), [feature], pulse_state=PulseState(), return_diagnostics=True
    )
    assert diag is not None and diag.feature_edit_steps == {0: [0]}
    torch.testing.assert_close(delta[0], torch.tensor([0.5, 0.0, 0.0]))
    assert torch.count_nonzero(delta[1:]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activation_threshold": -1.0},
        {"activation_threshold": float("nan")},
        {"gate_feature_indices": (1, 1)},
        {"gate_feature_indices": (0,)},
        {"gate_feature_indices": (3,)},
        {"gate_feature_indices": (True,)},
    ],
)
def test_invalid_detector_configuration_rejected(kwargs: dict) -> None:
    feature = SAEFeatureIntervention(0, "scale", 0.1, "norm_relative", **kwargs)
    with pytest.raises(ValueError):
        sae_intervention_delta(torch.ones(1, 3), IdentitySAE(), [feature], pulse_state=PulseState())
