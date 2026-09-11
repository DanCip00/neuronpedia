import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch
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
    def encode(self, activation_data):
        seq_len = activation_data.shape[1]
        result = torch.zeros((1, seq_len, 2), dtype=torch.float32)
        result[0, :, 0] = torch.arange(1, seq_len + 1)
        return result


def _patch_runtime(monkeypatch: pytest.MonkeyPatch):
    model = _Model()
    config = SimpleNamespace(device="cpu", activation_token_limit=100)
    manager = SimpleNamespace(get_sae_hook=lambda _source: "hook", get_sae=lambda _source: _Sae())
    monkeypatch.setattr(source_endpoint.Model, "get_instance", lambda: model)
    monkeypatch.setattr(source_endpoint.Config, "get_instance", lambda: config)
    monkeypatch.setattr(source_endpoint.SAEManager, "get_instance", lambda: manager)
    captured = {}

    async def capture(_model, tokens, lengths, hooks):
        captured["tokens"] = tokens.detach().cpu().tolist()
        captured["lengths"] = lengths
        return {hooks[0]: torch.zeros((*tokens.shape, 3), dtype=torch.float32)}

    monkeypatch.setattr(source_endpoint, "capture_padded_cache_async", capture)
    return captured


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
