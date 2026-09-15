from __future__ import annotations

import gc
import weakref

import pytest
import torch

from neuronpedia_inference.vllm_sae_worker import _make_sae_modifier, worker_unregister_sae_feature_steering


class WorkerTinySAE:
    W_dec = torch.eye(3)

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return torch.relu(rows)


def _spec(policy: str = "next_token", operation: str = "add", value: float | None = None) -> dict:
    if value is None and operation != "ablate":
        value = 2.0
    spec = {
        "sae": {"model": "toy", "release": "toy-release", "sae_id": "layer0"},
        "position_policy": policy,
        "return_diagnostics": True,
        "features": [{"feature_index": 1, "operation": operation, "value": value}],
    }
    if operation == "add":
        spec["decoder_vectors"] = {"1": [0.0, 1.0, 0.0]}
    return spec


def test_next_token_edits_only_final_prefill_row(monkeypatch) -> None:
    sae = WorkerTinySAE()
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._load_worker_sae", lambda *_args, **_kwargs: sae)
    diagnostics: list[dict] = []
    modifier = _make_sae_modifier(
        _spec("next_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=3,
        diag_store=diagnostics,
    )
    prefill = torch.zeros(3, 3)
    torch.testing.assert_close(modifier(prefill), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.zeros(1, 3))
    assert diagnostics[0]["row_offset"] == 2


def test_each_generated_token_edits_prefill_final_row_and_decode_rows(monkeypatch) -> None:
    sae = WorkerTinySAE()
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._load_worker_sae", lambda *_args, **_kwargs: sae)
    diagnostics: list[dict] = []
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=3,
        diag_store=diagnostics,
    )
    torch.testing.assert_close(
        modifier(torch.zeros(3, 3)), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    )
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.tensor([[0.0, 2.0, 0.0]]))
    assert [entry["row_offset"] for entry in diagnostics] == [2, 3]


def test_additive_worker_modifier_does_not_encode(monkeypatch) -> None:
    sae = WorkerTinySAE()
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._load_worker_sae", lambda *_args, **_kwargs: sae)
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=[],
    )
    modifier(torch.zeros(1, 3))
    assert sae.encode_calls == 0


def test_scale_worker_modifier_encodes_only_edited_rows(monkeypatch) -> None:
    sae = WorkerTinySAE()
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._load_worker_sae", lambda *_args, **_kwargs: sae)
    modifier = _make_sae_modifier(
        _spec("next_token", "scale"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=3,
        diag_store=[],
    )
    delta = modifier(torch.tensor([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [1.0, 3.0, 1.0]]))
    assert sae.encode_calls == 1
    torch.testing.assert_close(delta, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))


def test_one_token_next_token_does_not_mistake_decode_for_prefill(monkeypatch) -> None:
    monkeypatch.setattr(
        "neuronpedia_inference.vllm_sae_worker._load_worker_sae",
        lambda *_args, **_kwargs: pytest.fail("additive-only steering loaded an SAE"),
    )
    diagnostics: list[dict] = []
    modifier = _make_sae_modifier(
        _spec("next_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=diagnostics,
    )
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.tensor([[0.0, 2.0, 0.0]]))
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.zeros(1, 3))
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.zeros(1, 3))
    assert [(entry["row_offset"], entry["num_rows"]) for entry in diagnostics] == [(0, 1)]


def test_each_generated_token_positions_advance_across_decode_calls() -> None:
    diagnostics: list[dict] = []
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=diagnostics,
    )
    modifier(torch.zeros(1, 3))
    modifier(torch.zeros(1, 3))
    modifier(torch.zeros(2, 3))
    assert [(entry["row_offset"], entry["num_rows"]) for entry in diagnostics] == [(0, 1), (1, 1), (2, 2)]
    assert [entry["edit_count"] for entry in diagnostics] == [1, 1, 2]


def test_additive_modifier_uses_supplied_vectors_without_loading_sae(monkeypatch) -> None:
    monkeypatch.setattr(
        "neuronpedia_inference.vllm_sae_worker._load_worker_sae",
        lambda *_args, **_kwargs: pytest.fail("additive-only steering loaded an SAE"),
    )
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=[],
    )
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.tensor([[0.0, 2.0, 0.0]]))


def test_additive_modifier_rejects_missing_supplied_vectors() -> None:
    spec = _spec("next_token", "add")
    del spec["decoder_vectors"]
    with pytest.raises(ValueError, match="requires supplied decoder_vectors"):
        _make_sae_modifier(spec, dev=torch.device("cpu"), dt=torch.float32, prompt_len=1, diag_store=[])


def test_noop_add_reports_zero_edits_and_advances_position(monkeypatch) -> None:
    monkeypatch.setattr(
        "neuronpedia_inference.vllm_sae_worker._load_worker_sae",
        lambda *_args, **_kwargs: pytest.fail("no-op additive steering loaded an SAE"),
    )
    diagnostics: list[dict] = []
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add", value=0.0),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=diagnostics,
    )
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.zeros(1, 3))
    torch.testing.assert_close(modifier(torch.zeros(1, 3)), torch.zeros(1, 3))
    assert [(entry["row_offset"], entry["num_rows"], entry["edit_count"]) for entry in diagnostics] == [
        (0, 0, 0),
        (1, 0, 0),
    ]


def test_rejects_chunked_or_ambiguous_initial_prefill() -> None:
    modifier = _make_sae_modifier(
        _spec("each_generated_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=3,
        diag_store=[],
    )
    with pytest.raises(ValueError, match="requires an unchunked initial prefill"):
        modifier(torch.zeros(1, 3))


def test_modifier_delta_changes_downstream_logits_but_capture_clone_does_not() -> None:
    hidden = torch.zeros(1, 3)
    unembedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    baseline_logits = hidden @ unembedding

    capture_modifier = _make_sae_modifier(
        _spec("next_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=[],
    )
    captured = hidden.clone()
    captured += capture_modifier(captured)
    capture_only_logits = hidden @ unembedding
    torch.testing.assert_close(capture_only_logits, baseline_logits)

    causal_modifier = _make_sae_modifier(
        _spec("next_token", "add"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=[],
    )
    steered_logits = (hidden + causal_modifier(hidden)) @ unembedding
    assert int(baseline_logits.argmax(dim=-1).item()) == 0
    assert int(steered_logits.argmax(dim=-1).item()) == 1


def test_unregister_releases_request_scoped_sae(monkeypatch) -> None:
    sae = WorkerTinySAE()
    sae_ref = weakref.ref(sae)
    sae_holder = [sae]
    monkeypatch.setattr(
        "neuronpedia_inference.vllm_sae_worker._load_worker_sae",
        lambda *_args, **_kwargs: sae_holder.pop(),
    )
    modifier = _make_sae_modifier(
        _spec("next_token", "scale"),
        dev=torch.device("cpu"),
        dt=torch.float32,
        prompt_len=1,
        diag_store=[],
    )

    class Demux:
        steer_mods = {"request": {"site": (modifier, set(), 1)}}

    demux = Demux()
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._get_demux", lambda _worker: demux)
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._release_hook", lambda *_args: None)
    monkeypatch.setattr("neuronpedia_inference.vllm_sae_worker._maybe_unregister", lambda *_args: None)
    del modifier
    del sae
    worker_unregister_sae_feature_steering(object(), "request")
    gc.collect()
    assert sae_ref() is None
