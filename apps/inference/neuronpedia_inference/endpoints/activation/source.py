import base64
import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import BackendUnsupported, capture_padded_cache_async, get_tokenize
from neuronpedia_inference.memory_cost import activation_source_cost
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.schemas import ActivationSourceRequest, ActivationSourceResponse, ActivationSourceResult
from neuronpedia_inference.schemas.activation import (
    ActivationSourceChatInput,
    ActivationSourceInsertion,
    ActivationSourceOrigin,
    ActivationSourceTextInput,
    ActivationSourceTokenAlignment,
    ActivationSourceTokensInput,
    TokenInsertionMode,
)
from neuronpedia_inference.shared import Model, RecoverableOutOfMemory, recover_from_oom, with_request_lock

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_BATCH_SIZE = 4
ROUND_DECIMALS = 3


@dataclass
class _PreparedInput:
    input_type: Literal["text", "tokens", "chat"]
    input_token_ids: list[int] | None
    model_token_ids: list[int]
    tokens: list[str]
    alignment: list[ActivationSourceTokenAlignment]
    input_to_model_positions: list[int] | None
    rendered_text: str | None = None
    legacy_compat: bool = False


def _tokenizer_ids(tokenizer: Any) -> set[int]:
    """Return every ID the selected tokenizer can name, including added tokens."""
    return {int(token_id) for token_id in tokenizer.get_vocab().values()}


def _validate_ids(ids: list[int], valid_ids: set[int], input_index: int, field: str) -> None:
    """Reject an ID before it can become an embedding lookup/device-side assert."""
    for position, token_id in enumerate(ids):
        if token_id not in valid_ids:
            raise ValueError(
                f"Invalid token ID {token_id} at input {input_index}, {field}[{position}]; "
                "the selected model tokenizer does not contain this ID"
            )


def _decode_tokens(model: Any, ids: list[int]) -> list[str]:
    """Decode each position for display without using the result as model input."""
    return [str(token) for token in model.to_str_tokens(ids, prepend_bos=False)]


def _boundary_token(tokenizer: Any, name: str, input_index: int) -> int:
    token_id = getattr(tokenizer, f"{name}_token_id", None)
    if token_id is None:
        raise ValueError(
            f"Input {input_index} requested {name.upper()} insertion, but the tokenizer has no {name.upper()} token"
        )
    return int(token_id)


def _apply_insertion(
    input_ids: list[int],
    insertion: ActivationSourceInsertion,
    tokenizer: Any,
    valid_ids: set[int],
    input_index: int,
) -> tuple[list[int], list[ActivationSourceTokenAlignment], list[int]]:
    """Apply the documented boundary order and construct an exact position map."""
    _validate_ids(input_ids, valid_ids, input_index, "tokenIds")
    prefix = [int(token_id) for token_id in insertion.prefix_token_ids]
    suffix = [int(token_id) for token_id in insertion.suffix_token_ids]
    _validate_ids(prefix, valid_ids, input_index, "insertion.prefixTokenIds")
    _validate_ids(suffix, valid_ids, input_index, "insertion.suffixTokenIds")

    bos: list[int] = []
    if insertion.bos != TokenInsertionMode.NEVER:
        bos_id = _boundary_token(tokenizer, "bos", input_index)
        if insertion.bos == TokenInsertionMode.ALWAYS or not input_ids or input_ids[0] != bos_id:
            bos = [bos_id]
    eos: list[int] = []
    if insertion.eos != TokenInsertionMode.NEVER:
        eos_id = _boundary_token(tokenizer, "eos", input_index)
        if insertion.eos == TokenInsertionMode.ALWAYS or not input_ids or input_ids[-1] != eos_id:
            eos = [eos_id]

    model_ids = bos + prefix + input_ids + suffix + eos
    if not model_ids:
        raise ValueError(f"Input {input_index} produces an empty token sequence")
    _validate_ids(model_ids, valid_ids, input_index, "modelInputTokenIds")
    input_start = len(bos) + len(prefix)
    input_positions = list(range(input_start, input_start + len(input_ids)))
    sources: list[Literal["provided", "bos", "eos", "prefix", "suffix"]] = []
    sources.extend("bos" for _ in bos)
    sources.extend("prefix" for _ in prefix)
    sources.extend("provided" for _ in input_ids)
    sources.extend("suffix" for _ in suffix)
    sources.extend("eos" for _ in eos)
    input_by_model = {model_position: input_position for input_position, model_position in enumerate(input_positions)}
    texts = tokenizer.batch_decode([[token_id] for token_id in model_ids], clean_up_tokenization_spaces=False)
    alignment = [
        ActivationSourceTokenAlignment(
            model_position=model_position,
            token_id=token_id,
            token_text=str(texts[model_position]),
            input_position=input_by_model.get(model_position),
            source=source,
        )
        for model_position, (token_id, source) in enumerate(zip(model_ids, sources, strict=True))
    ]
    return model_ids, alignment, input_positions


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    return offsets


def _message_byte_ranges(
    rendered: bytes,
    messages: list[dict[str, str]],
    offsets: list[tuple[int, int] | None],
    spans: list[Any],
) -> list[tuple[int, int]]:
    """Locate verbatim content inside the engine-reported content-token region."""
    ranges: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        content = message["content"].encode("utf-8")
        content_offsets = [
            offset
            for offset, span in zip(offsets, spans, strict=True)
            if offset is not None and span.message_index == index and span.section == "content"
        ]
        if content:
            if not content_offsets:
                raise ValueError(f"Could not map chat message {index} content exactly into the rendered template")
            search_start = min(offset[0] for offset in content_offsets)
            search_end = max(offset[1] for offset in content_offsets)
        else:
            search_start = ranges[-1][1] if ranges else 0
            search_end = search_start
        start = rendered.find(content, search_start, search_end + 1)
        if start < 0:
            raise ValueError(f"Could not map chat message {index} content exactly into the rendered template")
        end = start + len(content)
        ranges.append((start, end))
    return ranges


def _chat_byte_offsets(tokenizer: Any, rendered: str, model_ids: list[int]) -> list[tuple[int, int] | None]:
    """Resolve token spans in rendered UTF-8, using a verified decode fallback for slow tokenizers."""
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offset_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = encoded.get("offset_mapping")
    except (NotImplementedError, TypeError, ValueError):
        offset_ids, offsets = [], None
    if offset_ids == model_ids and offsets is not None and len(offsets) == len(model_ids):
        char_bytes = _char_to_byte_offsets(rendered)
        return [
            (char_bytes[int(start)], char_bytes[int(end)]) if int(end) > int(start) else None
            for start, end in offsets
        ]

    # SentencePiece-backed tokenizers are often intentionally loaded as slow tokenizers and
    # cannot emit offsets. Prefix decoding is still exact when every decoded prefix is a byte
    # prefix of the final rendered text; verify that invariant rather than assuming it.
    rendered_bytes = rendered.encode("utf-8")
    full = tokenizer.decode(model_ids, clean_up_tokenization_spaces=False).encode("utf-8")
    if full != rendered_bytes:
        raise ValueError("Tokenizer cannot provide trustworthy offsets for this chat-template rendering")
    boundaries = [0]
    for end in range(1, len(model_ids) + 1):
        prefix = tokenizer.decode(model_ids[:end], clean_up_tokenization_spaces=False).encode("utf-8")
        if not rendered_bytes.startswith(prefix) or len(prefix) < boundaries[-1]:
            raise ValueError("Tokenizer cannot provide trustworthy offsets for this chat-template rendering")
        boundaries.append(len(prefix))
    return [
        (boundaries[index], boundaries[index + 1]) if boundaries[index + 1] > boundaries[index] else None
        for index in range(len(model_ids))
    ]


def _prepare_chat(model: Any, row: ActivationSourceChatInput, input_index: int, valid_ids: set[int]) -> _PreparedInput:
    tok = get_tokenize(model)
    if not tok.has_chat_template():
        raise ValueError("The selected model has no configured chat template; use text or tokens input")
    messages = [message.model_dump(exclude_none=True) for message in row.messages]
    model_ids = [
        int(token_id)
        for token_id in tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=row.add_generation_prompt,
            continue_final_message=row.continue_final_message,
        )
    ]
    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=row.add_generation_prompt,
        continue_final_message=row.continue_final_message,
    )
    if not isinstance(rendered, str):
        raise ValueError("The selected model chat template did not produce rendered text")
    _validate_ids(model_ids, valid_ids, input_index, "chatTemplateTokenIds")
    spans = tok.message_spans(
        messages,
        add_generation_prompt=row.add_generation_prompt,
        continue_final_message=row.continue_final_message,
    )
    if [int(span.token_id) for span in spans] != model_ids:
        raise ValueError("Chat-template token spans do not match the exact model input IDs")

    offsets = _chat_byte_offsets(model.tokenizer, rendered, model_ids)
    rendered_bytes = rendered.encode("utf-8")
    message_ranges = _message_byte_ranges(rendered_bytes, messages, offsets, spans)
    token_texts = _decode_tokens(model, model_ids)
    alignment: list[ActivationSourceTokenAlignment] = []
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    for position, (token_id, token_text, offset) in enumerate(zip(model_ids, token_texts, offsets, strict=True)):
        byte_start, byte_end = offset if offset is not None else (None, None)
        origins: list[ActivationSourceOrigin] = []
        if byte_start is None or byte_end is None:
            origin_type = "bos" if bos_id is not None and token_id == int(bos_id) else "chat_template"
            origins.append(ActivationSourceOrigin(type=origin_type))
        else:
            for message_index, (message_start, message_end) in enumerate(message_ranges):
                overlap_start = max(byte_start, message_start)
                overlap_end = min(byte_end, message_end)
                if overlap_start < overlap_end:
                    origins.append(
                        ActivationSourceOrigin(
                            type="message_content",
                            message_index=message_index,
                            message_role=messages[message_index]["role"],
                            content_byte_start=overlap_start - message_start,
                            content_byte_end=overlap_end - message_start,
                        )
                    )
            covered = sum(
                int(origin.content_byte_end or 0) - int(origin.content_byte_start or 0)
                for origin in origins
                if origin.type == "message_content"
            )
            if covered < byte_end - byte_start:
                origins.insert(0, ActivationSourceOrigin(type="chat_template"))
        source: Literal["message", "bos", "chat_template"]
        if any(origin.type == "message_content" for origin in origins):
            source = "message"
        elif any(origin.type == "bos" for origin in origins):
            source = "bos"
        else:
            source = "chat_template"
        alignment.append(
            ActivationSourceTokenAlignment(
                model_position=position,
                token_id=token_id,
                token_text=token_text,
                source=source,
                token_bytes=(
                    base64.b64encode(rendered_bytes[byte_start:byte_end]).decode("ascii")
                    if byte_start is not None and byte_end is not None
                    else None
                ),
                rendered_byte_start=byte_start,
                rendered_byte_end=byte_end,
                origins=origins,
            )
        )
    return _PreparedInput("chat", None, model_ids, token_texts, alignment, None, rendered_text=rendered)


def _prepare_inputs(request: ActivationSourceRequest) -> list[_PreparedInput]:
    model = Model.get_instance()
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")
    valid_ids = _tokenizer_ids(tokenizer)
    prepared: list[_PreparedInput] = []

    if request.prompts is not None:
        insertion = request.insertion or ActivationSourceInsertion(bos=TokenInsertionMode.IF_MISSING)
        for index, prompt in enumerate(request.prompts):
            input_ids = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
            model_ids, alignment, input_positions = _apply_insertion(input_ids, insertion, tokenizer, valid_ids, index)
            prepared.append(
                _PreparedInput(
                    "text",
                    input_ids,
                    model_ids,
                    _decode_tokens(model, model_ids),
                    alignment,
                    input_positions,
                    legacy_compat=request.insertion is None,
                )
            )
        return prepared

    if request.prompt_token_ids is not None:
        insertion = request.insertion or ActivationSourceInsertion()
        for index, ids in enumerate(request.prompt_token_ids):
            input_ids = [int(token_id) for token_id in ids]
            model_ids, alignment, input_positions = _apply_insertion(input_ids, insertion, tokenizer, valid_ids, index)
            prepared.append(
                _PreparedInput(
                    "tokens", input_ids, model_ids, _decode_tokens(model, model_ids), alignment, input_positions
                )
            )
        return prepared

    assert request.inputs is not None
    for index, row in enumerate(request.inputs):
        if isinstance(row, ActivationSourceChatInput):
            prepared.append(_prepare_chat(model, row, index, valid_ids))
            continue
        if isinstance(row, ActivationSourceTextInput):
            input_ids = [int(token_id) for token_id in tokenizer.encode(row.text, add_special_tokens=False)]
            input_type: Literal["text", "tokens"] = "text"
        elif isinstance(row, ActivationSourceTokensInput):
            input_ids = [int(token_id) for token_id in row.token_ids]
            input_type = "tokens"
        else:  # pragma: no cover
            raise ValueError(f"Unsupported activation input at position {index}")
        model_ids, alignment, input_positions = _apply_insertion(input_ids, row.insertion, tokenizer, valid_ids, index)
        prepared.append(
            _PreparedInput(input_type, input_ids, model_ids, _decode_tokens(model, model_ids), alignment, input_positions)
        )
    return prepared


@router.post("/activation/source", responses={200: {"model": ActivationSourceResponse}})
@with_request_lock(exclusive=False, cost=activation_source_cost)
async def activation_source(request: ActivationSourceRequest):
    Config.get_instance().check_requested_model(request.model)
    try:
        prepared = _prepare_inputs(request)
        result = await ActivationProcessor().process_activations_batch(request, prepared)
        return ActivationSourceResponse(results=result).model_dump(exclude_none=True)
    except (BackendUnsupported, ValueError) as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        if recover_from_oom(exc):
            return JSONResponse(content={"error": str(RecoverableOutOfMemory())}, status_code=503)
        logger.exception("Error processing activation/source")
        return JSONResponse(content={"error": "An error occurred while processing the request"}, status_code=500)


class ActivationProcessor:
    async def process_activations_batch(
        self, request: ActivationSourceRequest, prepared: list[_PreparedInput]
    ) -> list[ActivationSourceResult]:
        """Capture one ordered padded batch and encode each unpadded row."""
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()
        config = Config.get_instance()
        batch_size = len(prepared)
        batch_token_limit = (
            config.activation_token_limit if batch_size == 1 else config.activation_token_limit / MAX_BATCH_SIZE
        )
        for index, row in enumerate(prepared):
            if len(row.model_token_ids) > batch_token_limit:
                raise ValueError(
                    f"Input {index} is too long: {len(row.model_token_ids)} tokens, max is {int(batch_token_limit)}"
                )

        max_len = max(len(row.model_token_ids) for row in prepared)
        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token")
        padded_tokens = torch.full(
            (batch_size, max_len), int(pad_token_id), dtype=torch.long, device=config.device
        )
        original_lengths = [len(row.model_token_ids) for row in prepared]
        for index, row in enumerate(prepared):
            padded_tokens[index, : len(row.model_token_ids)] = torch.tensor(
                row.model_token_ids, dtype=torch.long, device=config.device
            )

        hook_name = sae_manager.get_sae_hook(request.source)
        cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])
        sae = sae_manager.get_sae(request.source)
        results: list[ActivationSourceResult] = []
        for index, row in enumerate(prepared):
            seq_len = original_lengths[index]
            with torch.no_grad():
                activation_data = cache[hook_name][index : index + 1, :seq_len].to(config.device)
                prompt_activations = sae.encode(activation_data)[0].float().cpu().numpy()
            token_indices, feature_indices = np.nonzero(prompt_activations)
            activation_values = prompt_activations[token_indices, feature_indices]
            active_features: dict[str, list[list[float]]] = {}
            for token_idx, feature_idx, activation_value in zip(
                token_indices, feature_indices, activation_values, strict=True
            ):
                if row.legacy_compat and token_idx == 0:
                    continue
                active_features.setdefault(str(int(feature_idx)), []).append(
                    [int(token_idx), round(float(activation_value), ROUND_DECIMALS)]
                )
            results.append(
                ActivationSourceResult(
                    tokens=row.tokens,
                    input_type=row.input_type,
                    input_token_ids=row.input_token_ids,
                    model_input_token_ids=row.model_token_ids,
                    token_alignment=row.alignment,
                    input_to_model_positions=row.input_to_model_positions,
                    rendered_text=row.rendered_text,
                    activeFeatures=active_features,
                )
            )
        return results
