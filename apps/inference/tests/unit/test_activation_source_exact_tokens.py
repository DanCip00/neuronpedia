import asyncio
import base64
from types import SimpleNamespace
from typing import Any, Literal, cast, overload

import pytest
import torch
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from neuronpedia_inference.endpoints.activation import source as source_endpoint
from neuronpedia_inference.schemas import ActivationSourceRequest, ChatMessage


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def get_vocab(self):
        return {str(index): index for index in range(10_000)}

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [10, 20] if text else []

    def batch_decode(self, rows, clean_up_tokenization_spaces=False):  # noqa: ARG002
        return [f"token-{row[0]}" for row in rows]


class _Model:
    tokenizer = _Tokenizer()

    def to_str_tokens(self, ids, prepend_bos=False):  # noqa: ARG002
        return [f"token-{token_id}" for token_id in ids]


class _Sae:
    def __init__(self):
        self.encode_shapes: list[tuple[int, ...]] = []

    def encode(self, activation_data):
        self.encode_shapes.append(tuple(activation_data.shape))
        if activation_data.ndim == 3:
            result = torch.zeros((*activation_data.shape[:2], 2), dtype=torch.float32)
            result[..., 0] = activation_data[..., 0]
            result[..., 1] = activation_data[..., 1]
            return result
        if activation_data.ndim == 2:
            result = torch.zeros((activation_data.shape[0], 2), dtype=torch.float32)
            result[:, 0] = activation_data[:, 0]
            result[:, 1] = activation_data[:, 1]
            return result
        raise AssertionError(f"unexpected activation_data shape {tuple(activation_data.shape)}")


@overload
def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, activation_batch_size: int = 4, return_sae: Literal[False] = False
) -> dict[str, Any]: ...


@overload
def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, activation_batch_size: int = 4, *, return_sae: Literal[True]
) -> tuple[dict[str, Any], _Sae]: ...


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, activation_batch_size: int = 4, return_sae: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], _Sae]:
    model = _Model()
    config = SimpleNamespace(
        device="cpu",
        activation_token_limit=100,
        activation_batch_size=activation_batch_size,
        check_requested_model=lambda _model: None,
    )
    sae = _Sae()
    manager = SimpleNamespace(
        get_sae_hook=lambda _source: "hook",
        get_sae=lambda _source: sae,
        get_d_sae=lambda _source: 2,
        get_d_in=lambda _source: 3,
    )
    monkeypatch.setattr(source_endpoint.Model, "get_instance", lambda: model)
    monkeypatch.setattr(source_endpoint.Config, "get_instance", lambda: config)
    monkeypatch.setattr(source_endpoint.SAEManager, "get_instance", lambda: manager)
    captured = {}

    async def capture(_model, tokens, lengths, hooks):
        captured["tokens"] = tokens.detach().cpu().tolist()
        captured["lengths"] = lengths
        hidden = torch.zeros((*tokens.shape, 3), dtype=torch.float32)
        for batch_index, seq_len in enumerate(lengths):
            for position in range(seq_len):
                hidden[batch_index, position, 0] = position + 1
                hidden[batch_index, position, 1] = batch_index + 1
                hidden[batch_index, position, 2] = float(tokens[batch_index, position].item())
        return {hooks[0]: hidden}

    monkeypatch.setattr(source_endpoint, "capture_padded_cache_async", capture)
    return (captured, sae) if return_sae else captured


def test_request_requires_exactly_one_input_mode():
    with pytest.raises(ValidationError, match="exactly one"):
        ActivationSourceRequest(model="gemma-3-4b-it", source="22-gemmascope-2-res-16k")
    with pytest.raises(ValidationError, match="exactly one"):
        ActivationSourceRequest(
            model="gemma-3-4b-it",
            source="22-gemmascope-2-res-16k",
            prompts=["x"],
            prompt_token_ids=[[1]],
        )


def test_invalid_exact_token_id_names_input_and_position(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 50_000]],
    )
    with pytest.raises(ValueError, match=r"Invalid token ID 50000 at input 0, tokenIds\[1\]"):
        source_endpoint._prepare_inputs(request)


def test_exact_candidates_remain_one_token_at_len_x_in_order(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_runtime(monkeypatch)
    x = [818, 5279, 529, 7001, 563]
    rows = [x + [9079], x + [5860]]
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=rows,
    )

    prepared = source_endpoint._prepare_inputs(request)
    results = asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))

    assert captured == {"tokens": rows, "lengths": [len(x) + 1, len(x) + 1]}
    assert [row.model_input_token_ids for row in results] == rows
    assert [row.input_to_model_positions for row in results] == [list(range(6)), list(range(6))]
    assert [row.model_input_token_ids[len(x)] for row in results] == [9079, 5860]
    assert all(row.active_features is not None and row.active_features["0"][-1][0] == len(x) for row in results)


def test_source_request_allows_more_than_four_rows_before_runtime_limit():
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10] for _ in range(5)],
    )
    assert len(request.prompt_token_ids or []) == 5


def test_configured_source_batch_size_rejects_oversized_batch(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch, activation_batch_size=4)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10] for _ in range(5)],
    )
    prepared = source_endpoint._prepare_inputs(request)

    with pytest.raises(ValueError, match="Batch size 5 exceeds maximum of 4"):
        asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))


def test_configured_source_batch_size_accepts_larger_batch(monkeypatch: pytest.MonkeyPatch):
    captured = _patch_runtime(monkeypatch, activation_batch_size=8)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10] for _ in range(5)],
    )
    prepared = source_endpoint._prepare_inputs(request)

    results = asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))

    assert len(results) == 5
    assert captured["lengths"] == [1, 1, 1, 1, 1]


def test_insertion_alignment_is_explicit(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20]],
        insertion=source_endpoint.ActivationSourceInsertion(
            bos=source_endpoint.TokenInsertionMode.ALWAYS,
            eos=source_endpoint.TokenInsertionMode.ALWAYS,
            prefix_token_ids=[30],
            suffix_token_ids=[40],
        ),
    )
    row = source_endpoint._prepare_inputs(request)[0]
    assert row.model_token_ids == [1, 30, 10, 20, 40, 2]
    assert row.input_to_model_positions == [2, 3]
    assert [entry.source for entry in row.alignment] == ["bos", "prefix", "provided", "provided", "suffix", "eos"]


def test_legacy_text_keeps_implicit_bos(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it", source="22-gemmascope-2-res-16k", prompts=["literal text"]
    )
    row = source_endpoint._prepare_inputs(request)[0]
    assert row.input_token_ids == [10, 20]
    assert row.model_token_ids == [1, 10, 20]
    assert row.input_to_model_positions == [1, 2]
    assert row.legacy_compat is True


def test_chat_alignment_uses_utf8_byte_spans(monkeypatch: pytest.MonkeyPatch):
    rendered = "<u>éx</u>"
    model_ids = [1, 3, 4]

    class ChatTokenizer(_Tokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):  # noqa: ARG002
            assert text == rendered
            return {"input_ids": model_ids, "offset_mapping": [(0, 3), (3, 5), (5, 9)]}

    class ChatModel(_Model):
        tokenizer = ChatTokenizer()

    spans = [
        SimpleNamespace(token_id=1, message_index=0, section="header"),
        SimpleNamespace(token_id=3, message_index=0, section="content"),
        SimpleNamespace(token_id=4, message_index=0, section="footer"),
    ]
    tok = SimpleNamespace(
        has_chat_template=lambda: True,
        apply_chat_template=lambda _messages, tokenize, **_kwargs: model_ids if tokenize else rendered,
        message_spans=lambda _messages, **_kwargs: spans,
    )
    monkeypatch.setattr(source_endpoint, "get_tokenize", lambda _model: tok)
    row = source_endpoint.ActivationSourceChatInput(
        type="chat", messages=[ChatMessage(role="user", content="éx")], add_generation_prompt=False
    )

    prepared = source_endpoint._prepare_chat(ChatModel(), row, 0, set(range(10_000)))

    content = prepared.alignment[1]
    assert (content.rendered_byte_start, content.rendered_byte_end) == (3, 6)
    assert content.token_bytes == base64.b64encode("éx".encode()).decode()
    assert content.origins is not None
    assert content.origins[0].content_byte_start == 0
    assert content.origins[0].content_byte_end == 3


def test_chat_generation_modes_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="cannot both be true"):
        source_endpoint.ActivationSourceChatInput(
            type="chat",
            messages=[ChatMessage(role="user", content="hello")],
            add_generation_prompt=True,
            continue_final_message=True,
        )


def _active_pairs(result, feature: str = "0"):
    assert result.active_features is not None
    return result.active_features.get(feature, [])


def test_selective_exact_final_positions_match_legacy_values(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    x = [818, 5279, 529, 7001, 563]
    rows = [x + [9079], x + [5860]]
    legacy_request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=rows,
    )
    legacy_prepared = source_endpoint._prepare_inputs(legacy_request)
    legacy_results = asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(legacy_request, legacy_prepared)
    )

    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=rows,
        activation_positions=[[-1], [-1]],
    )
    prepared = source_endpoint._prepare_inputs(request)
    results = asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))

    assert [row.model_input_token_ids for row in results] == rows
    assert [row.activation_positions for row in results] == [[len(x)], [len(x)]]
    assert [_active_pairs(row) for row in results] == [[[len(x), 6.0]], [[len(x), 6.0]]]
    assert [_active_pairs(row) for row in results] == [
        [pair for pair in _active_pairs(legacy_results[0]) if pair[0] == len(x)],
        [pair for pair in _active_pairs(legacy_results[1]) if pair[0] == len(x)],
    ]


def test_selective_multiple_positions_keep_model_positions(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20, 30, 40]],
        activation_positions=[[0, 2, -1]],
    )
    prepared = source_endpoint._prepare_inputs(request)

    result = asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))[0]

    assert result.activation_positions == [0, 2, 3]
    assert _active_pairs(result) == [[0, 1.0], [2, 3.0], [3, 4.0]]


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        ([[0], [0]], "activationPositions has 2 entries but request has 1 input"),
        ([[]], "activationPositions[0] must contain at least one position"),
        ([[1, -1]], r"duplicate position 1 after normalization"),
        ([[-2]], r"invalid position -2"),
        ([[2]], r"position 2 resolved to 2, outside model input length 2"),
        ([[10]], r"position 10 resolved to 10, outside model input length 2"),
        ([["1"]], r"malformed position '1'; positions must be integers"),
        ([[1.0]], r"malformed position 1.0; positions must be integers"),
        ([[True]], r"malformed position True; positions must be integers"),
    ],
)
def test_selective_position_validation_returns_400(monkeypatch: pytest.MonkeyPatch, positions, match: str):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20]],
        activation_positions=positions,
    )

    response = cast(JSONResponse, asyncio.run(source_endpoint.activation_source(request)))

    assert response.status_code == 400
    assert match in bytes(response.body).decode()


def test_selective_positions_are_resolved_after_insertions(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20]],
        insertion=source_endpoint.ActivationSourceInsertion(
            bos=source_endpoint.TokenInsertionMode.ALWAYS,
            eos=source_endpoint.TokenInsertionMode.ALWAYS,
            prefix_token_ids=[30],
            suffix_token_ids=[40],
        ),
        activation_positions=[[2, -1]],
    )
    prepared = source_endpoint._prepare_inputs(request)
    result = asyncio.run(source_endpoint.ActivationProcessor().process_activations_batch(request, prepared))[0]

    assert result.model_input_token_ids == [1, 30, 10, 20, 40, 2]
    assert result.activation_positions == [2, 5]
    assert _active_pairs(result) == [[2, 3.0], [5, 6.0]]


def test_selective_text_final_position_with_and_without_implicit_bos(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    implicit_bos = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompts=["literal text"],
        activation_positions=[[-1]],
    )
    explicit_no_bos = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompts=["literal text"],
        insertion=source_endpoint.ActivationSourceInsertion(bos=source_endpoint.TokenInsertionMode.NEVER),
        activation_positions=[[-1]],
    )

    implicit_result = asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(
            implicit_bos, source_endpoint._prepare_inputs(implicit_bos)
        )
    )[0]
    explicit_result = asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(
            explicit_no_bos, source_endpoint._prepare_inputs(explicit_no_bos)
        )
    )[0]

    assert implicit_result.model_input_token_ids == [1, 10, 20]
    assert implicit_result.activation_positions == [2]
    assert _active_pairs(implicit_result) == [[2, 3.0]]
    assert explicit_result.model_input_token_ids == [10, 20]
    assert explicit_result.activation_positions == [1]
    assert _active_pairs(explicit_result) == [[1, 2.0]]


def test_selective_chat_final_position_preserves_alignment(monkeypatch: pytest.MonkeyPatch):
    rendered = "<u>éx</u>"
    model_ids = [1, 3, 4]

    class ChatTokenizer(_Tokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):  # noqa: ARG002
            assert text == rendered
            return {"input_ids": model_ids, "offset_mapping": [(0, 3), (3, 5), (5, 9)]}

    class ChatModel(_Model):
        tokenizer = ChatTokenizer()

    captured, _sae = _patch_runtime(monkeypatch, return_sae=True)
    monkeypatch.setattr(source_endpoint.Model, "get_instance", lambda: ChatModel())
    spans = [
        SimpleNamespace(token_id=1, message_index=0, section="header"),
        SimpleNamespace(token_id=3, message_index=0, section="content"),
        SimpleNamespace(token_id=4, message_index=0, section="footer"),
    ]
    tok = SimpleNamespace(
        has_chat_template=lambda: True,
        apply_chat_template=lambda _messages, tokenize, **_kwargs: model_ids if tokenize else rendered,
        message_spans=lambda _messages, **_kwargs: spans,
    )
    monkeypatch.setattr(source_endpoint, "get_tokenize", lambda _model: tok)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        inputs=[
            source_endpoint.ActivationSourceChatInput(
                type="chat", messages=[ChatMessage(role="user", content="éx")], add_generation_prompt=False
            )
        ],
        activation_positions=[[-1]],
    )

    result = asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(request, source_endpoint._prepare_inputs(request))
    )[0]

    assert captured["tokens"] == [model_ids]
    assert result.activation_positions == [2]
    assert _active_pairs(result) == [[2, 3.0]]
    assert result.rendered_text == rendered
    assert result.token_alignment[1].token_bytes == base64.b64encode("éx".encode()).decode()


def test_legacy_all_position_response_omits_activation_positions(monkeypatch: pytest.MonkeyPatch):
    _patch_runtime(monkeypatch)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20]],
    )
    result = asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(request, source_endpoint._prepare_inputs(request))
    )[0]

    assert result.activation_positions is None
    assert "activationPositions" not in result.model_dump(exclude_none=True)


def test_selective_positions_are_encoded_together_without_full_sequence(monkeypatch: pytest.MonkeyPatch):
    _captured, sae = _patch_runtime(monkeypatch, return_sae=True)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20, 30, 40], [10, 20, 30, 50]],
        activation_positions=[[-1], [-1]],
    )
    asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(request, source_endpoint._prepare_inputs(request))
    )

    assert sae.encode_shapes == [(2, 3)]


def test_legacy_path_keeps_per_sequence_encode_shape(monkeypatch: pytest.MonkeyPatch):
    _captured, sae = _patch_runtime(monkeypatch, return_sae=True)
    request = ActivationSourceRequest(
        model="gemma-3-4b-it",
        source="22-gemmascope-2-res-16k",
        prompt_token_ids=[[10, 20, 30], [10, 20]],
    )
    asyncio.run(
        source_endpoint.ActivationProcessor().process_activations_batch(request, source_endpoint._prepare_inputs(request))
    )

    assert sae.encode_shapes == [(1, 3, 3), (1, 2, 3)]
