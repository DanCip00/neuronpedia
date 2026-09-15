from __future__ import annotations

import torch

from neuronpedia_inference.sae_interventions import SAEFeatureIntervention, sae_intervention_delta


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
