"""vLLM batched capture: every row is submitted to the engine at once and scattered back in order."""

import asyncio

import pytest
import torch

import neuronpedia_inference.engine_adapter as adapter


class _ParallelModel:
    """Each capture blocks until every row has been submitted, so a sequential loop would hang."""

    def __init__(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows
        self.started: list[int] = []
        self.all_started = asyncio.Event()

    async def capture(self, token_ids, points):  # type: ignore[no-untyped-def]
        self.started.append(int(token_ids[-1]))
        if len(self.started) == self.expected_rows:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=0.5)
        return {points[0]: torch.full((len(token_ids), 3), float(token_ids[-1]))}


class _FailingModel:
    async def capture(self, token_ids, points):  # type: ignore[no-untyped-def]
        if int(token_ids[-1]) == 10:
            raise RuntimeError("capture failed")
        return {points[0]: torch.zeros((len(token_ids), 3))}


def _patch_vllm_capture(monkeypatch, model_type) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter, "VLLMModel", model_type)
    monkeypatch.setattr(adapter, "_vllm_points_use_native_resid", lambda *_args: False)
    monkeypatch.setattr(adapter, "_assert_vllm_points_supported", lambda *_args: None)


def test_vllm_capture_submits_every_row_at_once_and_preserves_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    model = _ParallelModel(expected_rows=3)
    _patch_vllm_capture(monkeypatch, _ParallelModel)
    padded = torch.tensor([[1, 10, 0], [2, 20, 30], [3, 40, 0]])
    hook = "blocks.0.hook_resid_post"

    result = asyncio.run(adapter.capture_padded_cache_async(model, padded, [2, 3, 2], [hook]))

    assert model.started == [10, 30, 40]
    assert result[hook].shape == (3, 3, 3)
    assert torch.equal(result[hook][0, :2], torch.full((2, 3), 10.0))
    assert torch.equal(result[hook][0, 2], torch.zeros(3))
    assert torch.equal(result[hook][1], torch.full((3, 3), 30.0))
    assert torch.equal(result[hook][2, :2], torch.full((2, 3), 40.0))


def test_vllm_capture_failure_surfaces_the_original_exception(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Endpoints inspect the raised exception for OOM recovery, so it must not be wrapped."""
    _patch_vllm_capture(monkeypatch, _FailingModel)
    padded = torch.tensor([[1, 10], [2, 20]])

    with pytest.raises(RuntimeError, match="capture failed"):
        asyncio.run(adapter.capture_padded_cache_async(_FailingModel(), padded, [2, 2], ["blocks.0.hook_resid_post"]))
