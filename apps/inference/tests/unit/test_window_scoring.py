"""Full-vocabulary KL, causal pulse propagation, chunking, and endpoint cleanup."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

from neuronpedia_inference.endpoints.steer import window as endpoint
from neuronpedia_inference.schemas.steer import SAEInterventionPositionPolicy, SourceSteerResolvedMetadata
from neuronpedia_inference.schemas.window import SourceWindowScoreRequest
from neuronpedia_inference.vllm_window_worker import WindowModifier, compare_window, full_kl


class TinySAE:
    W_dec = torch.eye(3)

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.relu()


def spec() -> dict[str, Any]:
    return {
        "layer": 0,
        "point": "resid_post",
        "sae": {},
        "features": [
            {
                "feature_index": 0,
                "operation": "scale",
                "value": 0.25,
                "magnitude_mode": "norm_relative",
                "cooldown_mode": "feature_inactive",
            }
        ],
    }


def test_full_kl_matches_analytic_distribution_and_includes_tail() -> None:
    p = torch.tensor([[0.05] * 20 + [0.0] * 3, [0.01] * 20 + [0.2, 0.3, 0.3]], dtype=torch.float64)
    q = torch.tensor([[0.04] * 20 + [0.1, 0.05, 0.05], [0.01] * 20 + [0.4, 0.2, 0.2]], dtype=torch.float64)
    expected = torch.where(p > 0, p * (p.log() - q.log()), torch.zeros_like(p)).sum(-1)
    torch.testing.assert_close(full_kl(p.log(), q.log()).double(), expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(full_kl(q.log(), q.log()), torch.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        full_kl(q.log(), p.log())


@pytest.mark.parametrize("arm,pulses", [("feature_inactive", [0, 3]), ("single_pulse", [0])])
def test_prefill_chunks_preserve_exact_episode_schedule(arm: str, pulses: list[int]) -> None:
    rows = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    whole = WindowModifier(spec(), TinySAE(), 5, arm)
    split = WindowModifier(spec(), TinySAE(), 5, arm)
    a = whole(rows)
    b = torch.cat([split(rows[:1]), split(rows[1:3]), split(rows[3:])])
    torch.testing.assert_close(a, b)
    assert whole.pulses == split.pulses == pulses
    assert len(whole.activations) == 4  # Last input predicts outside this window.
    assert not whole.invalid_layout
    assert torch.count_nonzero(a[-1]) == 0


def test_single_pulse_changes_later_predictions_without_later_edits() -> None:
    rows = torch.tensor([[1.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 1.0]])
    modifier = WindowModifier(spec(), TinySAE(), 4, "single_pulse")
    delta = modifier(rows)
    # A simple causal downstream mixer propagates the edited value to later rows.
    divisor = torch.arange(1, 5).unsqueeze(-1)
    baseline = rows + rows.cumsum(0) / divisor
    steered = rows + delta + (rows + delta).cumsum(0) / divisor
    kl = full_kl(baseline[:-1], steered[:-1])
    assert modifier.pulses == [0]
    assert modifier.perturbation_norms[1:] == [0.0, 0.0]
    assert bool((kl > 0).all())


def test_replay_is_recorded_and_rejected_after_forward() -> None:
    modifier = WindowModifier(spec(), TinySAE(), 3, "feature_inactive")
    modifier(torch.ones(3, 3))
    modifier(torch.ones(3, 3))
    assert modifier.invalid_layout


def test_output_head_receives_already_normalized_states(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    current = baseline + 0.1
    worker = SimpleNamespace(_np_window_references={"base": baseline}, _np_window_modifiers={})
    monkeypatch.setattr("neuronpedia_inference.vllm_window_worker._take_rows", lambda *_: {"final_norm": current})
    seen = []

    def logits(rows):
        seen.append(rows.clone())
        return rows

    monkeypatch.setattr(
        "neuronpedia_inference.vllm_window_worker._worker_model", lambda *_: SimpleNamespace(compute_logits=logits)
    )
    result = compare_window(worker, "steer", "base", 3, "baseline_repeat")
    torch.testing.assert_close(seen[0], baseline)
    torch.testing.assert_close(seen[1], current)
    assert len(result["kl"]) == 2


def request(**updates) -> SourceWindowScoreRequest:
    raw = {
        "model": "toy",
        "tokenIds": [1, 2, 3],
        "steering": {
            "source": "layer0",
            "sourceSet": "toy",
            "positionPolicy": "each_generated_token",
            "features": [
                {
                    "featureIndex": 0,
                    "operation": "scale",
                    "value": 0.25,
                    "magnitudeMode": "norm_relative",
                    "cooldownMode": "feature_inactive",
                }
            ],
        },
    }
    raw.update(updates)
    return SourceWindowScoreRequest.model_validate(raw)


def test_schema_rejects_multifeature_and_fixed_cooldown() -> None:
    r = request().model_dump()
    r["steering"]["features"] *= 2
    with pytest.raises(ValidationError):
        request(steering=r["steering"])
    r = request().model_dump()
    r["steering"]["features"][0]["cooldownMode"] = "fixed"
    with pytest.raises(ValidationError):
        request(steering=r["steering"])
    with pytest.raises(ValidationError):
        request(tokenIds=[1] * 513)


@pytest.mark.parametrize("count", [1194, 4096])
def test_expanded_trace_banks_are_accepted(count: int) -> None:
    assert len(request(traceFeatureIndices=list(range(count))).trace_feature_indices) == count


def test_trace_bank_still_has_a_bounded_size() -> None:
    with pytest.raises(ValidationError, match="at most 4096"):
        request(traceFeatureIndices=list(range(4097)))


class Model:
    tensor_parallel_size = 1
    _engine_kwargs: dict[str, Any] = {}
    tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(20)})

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self.prompts = []
        self.engine = self
        self.fail = fail

    async def _ensure_engine(self):
        pass

    def _new_request_id(self, _prefix):
        return f"window-{len(self.prompts)}"

    def _prompt(self, ids, private_kv_for):
        self.prompts.append((ids, private_kv_for))
        return {"prompt_token_ids": ids}

    async def _run_one(self, *_args, **_kwargs):
        if self.fail:
            raise RuntimeError("forward failed")

    async def collective_rpc(self, method, args):
        self.calls.append((method, args))
        if method == "store_code_window_reference":
            return [{"feature_activations": {"0": [1.0, 0.0]}, "residual_norms": [4.0, 4.0]}]
        if method == "compare_code_window":
            arm = args[-1]
            return [
                {
                    "arm": arm,
                    "kl": [0.0, 0.0],
                    "top1_agree": [True, True],
                    "feature_activations": [] if arm == "baseline_repeat" else [1.0, 0.0],
                    "residual_norms": [] if arm == "baseline_repeat" else [4.0, 4.0],
                    "perturbation_norms": [] if arm == "baseline_repeat" else [1.0, 0.0],
                    "pulse_positions": [] if arm == "baseline_repeat" else [0],
                }
            ]
        return None


def patch_runtime(monkeypatch, model):
    monkeypatch.setattr(endpoint, "VLLMModel", Model)
    monkeypatch.setattr(endpoint.Model, "get_instance", lambda: model)
    monkeypatch.setattr(
        endpoint.Config, "get_instance", lambda: SimpleNamespace(token_limit=100, check_requested_model=lambda *_: None)
    )
    monkeypatch.setattr(endpoint.SAEManager, "get_instance", lambda: SimpleNamespace(get_d_sae=lambda *_: 3))
    monkeypatch.setattr(endpoint, "assert_steering_available", lambda *_: None)
    monkeypatch.setattr(endpoint, "declares_static_taps", lambda *_: False)
    monkeypatch.setattr(
        endpoint,
        "_resolve_source_metadata",
        lambda r: (
            spec(),
            SourceSteerResolvedMetadata(
                source="layer0",
                source_set="toy",
                hook_point="resid_post",
                hook_layer=0,
                position_policy=SAEInterventionPositionPolicy.EACH_GENERATED_TOKEN,
                features=r.steering.features,
            ),
        ),
    )


def test_endpoint_uses_private_identical_prompts_and_releases_references(monkeypatch) -> None:
    model = Model()
    patch_runtime(monkeypatch, model)
    result = asyncio.run(cast(Any, endpoint.score_source_window).__wrapped__(request()))
    assert result["predictionPositions"] == [0, 1]
    assert len(model.prompts) == 4
    assert all(ids == [1, 2, 3] for ids, _ in model.prompts)
    assert len({rid for _, rid in model.prompts}) == 4
    cleaned = {args[0] for method, args in model.calls if method == "cleanup_code_window"}
    assert cleaned == {rid for _, rid in model.prompts}


def test_endpoint_cleans_partial_forward_failure(monkeypatch) -> None:
    model = Model(fail=True)
    patch_runtime(monkeypatch, model)
    result = asyncio.run(cast(Any, endpoint.score_source_window).__wrapped__(request()))
    assert result.status_code == 500
    assert ("cleanup_code_window", ("window-0",)) in model.calls


def test_registration_initializes_device_before_loading_sae(monkeypatch) -> None:
    from neuronpedia_inference import vllm_window_worker as worker_module

    demux = SimpleNamespace(dev=None, dt=None, steer_mods={})
    worker = SimpleNamespace()
    monkeypatch.setattr(worker_module, "worker_register_capture", lambda *_: None)
    monkeypatch.setattr(worker_module, "_get_demux", lambda *_: demux)

    def initialize(*_):
        demux.dev, demux.dt = torch.device("cpu"), torch.float32

    monkeypatch.setattr(worker_module, "_ensure_dev", initialize)
    monkeypatch.setattr(worker_module, "_write_site", lambda *_: "site")
    monkeypatch.setattr(worker_module, "_ensure_hook", lambda *_: None)

    def load(_spec, device, dtype):
        assert device == torch.device("cpu") and dtype == torch.float32
        return TinySAE()

    monkeypatch.setattr(worker_module, "_load_worker_sae", load)
    worker_module.register_window(worker, "window", spec(), 3, "single_pulse")
    assert "site" in demux.steer_mods["window"]


def test_window_gate_delays_within_episode_across_chunks_and_single_pulse() -> None:
    gated = spec()
    gated["features"][0].update(activation_threshold=2.0, gate_feature_indices=[1])
    rows = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    inactive = WindowModifier(gated, TinySAE(), len(rows), "feature_inactive")
    one = WindowModifier(gated, TinySAE(), len(rows), "single_pulse")
    whole = inactive(rows)
    split = torch.cat([one(rows[:2]), one(rows[2:5]), one(rows[5:])])
    assert inactive.pulses == [2, 6]
    assert one.pulses == [2]
    assert inactive.gate_activations == {"1": rows[:-1, 1].tolist()}
    assert torch.count_nonzero(whole[:, 1:]) == torch.count_nonzero(split[:, 1:]) == 0


@pytest.mark.parametrize("cutoff,pulses", [(0.0, [0, 4]), (2.0, [1, 4])])
def test_helper_cutoff_matches_across_window_prefill_chunks(cutoff, pulses) -> None:
    configured = spec()
    configured["features"][0].update(gate_feature_indices=[1], gate_feature_thresholds=[cutoff])
    rows = torch.tensor(
        [[1.0, 0.1, 0.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
    )
    whole = WindowModifier(configured, TinySAE(), len(rows), "feature_inactive")
    split = WindowModifier(configured, TinySAE(), len(rows), "feature_inactive")
    expected = whole(rows)
    actual = torch.cat([split(rows[:1]), split(rows[1:3]), split(rows[3:])])
    torch.testing.assert_close(actual, expected)
    assert whole.pulses == split.pulses == pulses
    assert torch.count_nonzero(actual[:, 1:]) == 0
