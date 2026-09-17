from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from neuronpedia_inference.endpoints.steer import source as source_endpoint
from neuronpedia_inference.schemas import (
    SAEInterventionPositionPolicy,
    SourceSteerFeature,
    SourceSteeringSpec,
    SourceSteerRequest,
)


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(1000)}

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:  # noqa: ARG002
        return "".join(f"<{token_id}>" for token_id in token_ids)


class _Engine:
    def __init__(self, fail_register: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_register = fail_register

    async def collective_rpc(self, method: str, args: tuple[Any, ...]) -> Any:
        self.calls.append((method, args))
        if method == "register_sae_feature_steering" and self.fail_register:
            raise RuntimeError("register failed after worker dispatch")
        if method == "collect_sae_feature_steering_diagnostics":
            return [[{"row_offset": 2, "num_rows": 1, "edit_count": 1, "perturbation_norms": [2.0], "active": True}]]
        return None


class _VLLMModel:
    tensor_parallel_size = 1
    tokenizer = _Tokenizer()

    def __init__(self, *, cancel: bool = False, fail_register: bool = False) -> None:
        self.engine = _Engine(fail_register=fail_register)
        self._engine_kwargs = {"max_num_batched_tokens": 100}
        self.prompt_ids: list[list[int]] = []
        self.cancel = cancel
        self.request_number = 0

    async def _ensure_engine(self) -> None:
        return None

    def _new_request_id(self, prefix: str) -> str:
        self.request_number += 1
        return f"{prefix}-{self.request_number}"

    def _prompt(self, token_ids: list[int], private_kv_for: str | None = None) -> dict[str, Any]:
        self.prompt_ids.append(list(token_ids))
        return {"prompt_token_ids": list(token_ids), "private_kv_for": private_kv_for}

    async def _run_one(self, prompt: object, sampling: object, request_id: str) -> Any:  # noqa: ARG002
        if self.cancel:
            raise asyncio.CancelledError
        first = SimpleNamespace(logprob=-0.2, rank=1)
        alternative = SimpleNamespace(logprob=-1.2, rank=2)
        completion = SimpleNamespace(
            token_ids=[7],
            text="<7>",
            finish_reason="length",
            logprobs=[{7: first, 8: alternative}],
        )
        return SimpleNamespace(outputs=[completion])


class _Manager:
    sae_set_to_saes = {"toy-release": ["layer0"]}
    sae_id_to_release = {"layer0": "toy-release"}
    sae_data = {"layer0": {"release": "toy-release", "saelens_id": "checkpoint0"}}

    def __init__(self) -> None:
        self.get_sae_calls = 0

    def get_valid_sae_sets(self) -> list[str]:
        return ["toy-release"]

    def ensure_source(self, source: str) -> None:  # noqa: ARG002
        return None

    def get_d_sae(self, source: str) -> int:  # noqa: ARG002
        return 3

    def get_sae_hook(self, source: str) -> str:  # noqa: ARG002
        return "blocks.0.hook_resid_post"

    def get_sae(self, source: str) -> Any:  # noqa: ARG002
        self.get_sae_calls += 1
        return SimpleNamespace(W_dec=torch.eye(3))


def _request(features: list[dict[str, Any]] | None = None, diagnostics: bool = False) -> SourceSteerRequest:
    steering: SourceSteeringSpec | None = None
    if features is not None:
        steering = SourceSteeringSpec(
            source="layer0",
            source_set="toy-release",
            position_policy=SAEInterventionPositionPolicy.NEXT_TOKEN,
            features=[SourceSteerFeature.model_validate(feature) for feature in features],
        )
    return SourceSteerRequest(
        model="toy",
        prompt_token_ids=[[11, 12, 13]],
        steering=steering,
        max_new_tokens=1,
        temperature=0.0,
        top_logprobs=2,
        return_intervention_diagnostics=diagnostics,
    )


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, model: _VLLMModel, manager: _Manager) -> None:
    config = SimpleNamespace(
        token_limit=100,
        max_tokens=100,
        sae_dtype="float32",
        device="cpu",
        check_requested_model=lambda requested: None,
        clamp_completion_tokens=lambda prompt_len, requested: min(requested, 100 - prompt_len),
    )
    monkeypatch.setattr(source_endpoint, "VLLMModel", _VLLMModel)
    monkeypatch.setattr(source_endpoint.Config, "get_instance", lambda: config)
    monkeypatch.setattr(source_endpoint.Model, "get_instance", lambda: model)
    monkeypatch.setattr(source_endpoint.SAEManager, "get_instance", lambda: manager)
    monkeypatch.setattr(source_endpoint, "assert_steering_available", lambda *_args: None)
    monkeypatch.setattr(source_endpoint, "declares_static_taps", lambda _model: False)
    source_endpoint._DECODER_VECTOR_CACHE.clear()


async def _call(request: SourceSteerRequest) -> Any:
    handler = cast(
        Callable[[SourceSteerRequest], Awaitable[Any]],
        cast(Any, source_endpoint.steer_source).__wrapped__,
    )
    return await handler(request)


def test_baseline_preserves_exact_ids_and_generation_logprob_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    _patch_runtime(monkeypatch, model, manager)

    response = asyncio.run(_call(_request()))

    assert response["modelInputTokenIds"] == [11, 12, 13]
    assert response["generatedTokenIds"] == [7]
    assert response["logprobs"][0]["tokenId"] == 7
    assert response["logprobs"][0]["logprob"] == -0.2
    assert [item["tokenId"] for item in response["logprobs"][0]["topLogprobs"]] == [7, 8]
    assert model.prompt_ids == [[11, 12, 13]]
    assert model.engine.calls == []
    assert manager.get_sae_calls == 0


def test_noop_uses_baseline_path_without_loading_or_registering(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    _patch_runtime(monkeypatch, model, manager)

    response = asyncio.run(_call(_request([{"featureIndex": 1, "operation": "scale", "value": 1.0}], True)))

    assert model.engine.calls == []
    assert manager.get_sae_calls == 0
    assert response["resolved"]["source"] == "layer0"
    assert response["resolved"]["features"] == []
    assert response["interventionDiagnostics"]["active"] is False


def test_baseline_steered_baseline_has_no_cross_request_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    _patch_runtime(monkeypatch, model, manager)

    baseline_one = asyncio.run(_call(_request()))
    steered = asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}], True)))
    baseline_two = asyncio.run(_call(_request()))

    methods = [method for method, _args in model.engine.calls]
    assert methods == [
        "register_sae_feature_steering",
        "collect_sae_feature_steering_diagnostics",
        "unregister_sae_feature_steering",
    ]
    assert baseline_one["generatedTokenIds"] == baseline_two["generatedTokenIds"] == [7]
    assert steered["interventionDiagnostics"]["editedPredictionSteps"] == [2]
    assert steered["interventionDiagnostics"]["editCount"] == 1
    assert steered["resolved"]["features"] == [{"featureIndex": 1, "operation": "add", "value": 2.0}]
    register_spec = model.engine.calls[0][1][1][0]
    assert register_spec["sae"]["sae_id"] == "checkpoint0"
    assert register_spec["sae"]["dtype"] == "float32"
    assert register_spec["decoder_vectors"]["1"] == [0.0, 1.0, 0.0]


def test_both_arms_take_private_kv_so_the_baseline_prefill_is_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    _patch_runtime(monkeypatch, model, manager)
    prompts: list[dict[str, Any]] = []
    original = model._prompt

    def recording_prompt(token_ids: list[int], private_kv_for: str | None = None) -> dict[str, Any]:
        built = original(token_ids, private_kv_for=private_kv_for)
        prompts.append(built)
        return built

    monkeypatch.setattr(model, "_prompt", recording_prompt)

    asyncio.run(_call(_request()))
    asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}])))

    assert [prompt["private_kv_for"] for prompt in prompts] == ["np-sae-base-1", "np-sae-steer-2"]


def test_prompt_longer_than_one_prefill_step_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker finds the edited row in whichever prefill chunk holds it, so nothing to refuse."""
    model, manager = _VLLMModel(), _Manager()
    model._engine_kwargs = {"max_num_batched_tokens": 2}
    _patch_runtime(monkeypatch, model, manager)

    response = asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}])))

    assert response["generatedTokenIds"] == [7]
    assert response["resolved"]["prefillChunking"] == "supported"
    assert [method for method, _args in model.engine.calls] == [
        "register_sae_feature_steering",
        "unregister_sae_feature_steering",
    ]


def test_resolved_features_echo_every_effective_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    _patch_runtime(monkeypatch, model, manager)
    features = [
        {"featureIndex": 0, "operation": "add", "value": 4.0},
        {"featureIndex": 1, "operation": "scale", "value": 0.5},
        {"featureIndex": 2, "operation": "ablate"},
    ]

    response = asyncio.run(_call(_request(features)))

    assert response["resolved"]["features"] == features
    register_spec = model.engine.calls[0][1][1][0]
    assert register_spec["features"] == [
        {"feature_index": 0, "operation": "add", "value": 4.0},
        {"feature_index": 1, "operation": "scale", "value": 0.5},
        {"feature_index": 2, "operation": "ablate", "value": None},
    ]


def test_cancelled_generation_unregisters_worker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(cancel=True), _Manager()
    _patch_runtime(monkeypatch, model, manager)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}])))

    assert [method for method, _args in model.engine.calls] == [
        "register_sae_feature_steering",
        "unregister_sae_feature_steering",
    ]


def test_failed_registration_still_attempts_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(fail_register=True), _Manager()
    _patch_runtime(monkeypatch, model, manager)

    response = asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}])))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert [method for method, _args in model.engine.calls] == [
        "register_sae_feature_steering",
        "unregister_sae_feature_steering",
    ]


def test_speculative_decode_is_rejected_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    model, manager = _VLLMModel(), _Manager()
    model._engine_kwargs["num_speculative_tokens"] = 4
    _patch_runtime(monkeypatch, model, manager)

    response = asyncio.run(_call(_request([{"featureIndex": 1, "operation": "add", "value": 2.0}])))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert b"speculative decoding" in response.body
    assert model.engine.calls == []


@pytest.mark.parametrize(
    "feature",
    [
        {"featureIndex": True, "operation": "add", "value": 1.0},
        {"featureIndex": 1, "operation": "add"},
        {"featureIndex": 1, "operation": "add", "value": float("nan")},
        {"featureIndex": 1, "operation": "scale", "value": -0.1},
        {"featureIndex": 1, "operation": "ablate", "value": 0.0},
    ],
    ids=["bool-index", "missing-add-value", "nan", "negative-scale", "ablate-value"],
)
def test_invalid_feature_contract_is_rejected(feature: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _request([feature])


def test_duplicate_features_and_batches_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate featureIndex"):
        _request(
            [
                {"featureIndex": 1, "operation": "add", "value": 1.0},
                {"featureIndex": 1, "operation": "ablate"},
            ]
        )
    with pytest.raises(ValidationError, match="at most 1 item"):
        SourceSteerRequest.model_validate({"model": "toy", "promptTokenIds": [[1], [2]]})
