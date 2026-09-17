import base64
import logging
from collections.abc import Set as AbstractSet
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


# Keyed by object identity, holding the tokenizer so the key cannot be recycled by a new object.
_VALID_IDS_BY_TOKENIZER: dict[int, tuple[Any, frozenset[int]]] = {}


def _tokenizer_ids(tokenizer: Any) -> frozenset[int]:
    """Return every ID the selected tokenizer can name, including added tokens.

    Computed once per tokenizer object. ``get_vocab()`` materializes the whole vocabulary --
    250k entries for Qwen -- and doing that per request cost ~150 ms, more than the model
    forward for a 400-token row. The vocabulary of a loaded tokenizer does not change.
    """
    cached = _VALID_IDS_BY_TOKENIZER.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    valid_ids = frozenset(int(token_id) for token_id in tokenizer.get_vocab().values())
    _VALID_IDS_BY_TOKENIZER[id(tokenizer)] = (tokenizer, valid_ids)
    return valid_ids


def _validate_ids(ids: list[int], valid_ids: AbstractSet[int], input_index: int, field: str) -> None:
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
    valid_ids: AbstractSet[int],
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
            (char_bytes[int(start)], char_bytes[int(end)]) if int(end) > int(start) else None for start, end in offsets
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


def _prepare_chat(
    model: Any, row: ActivationSourceChatInput, input_index: int, valid_ids: AbstractSet[int]
) -> _PreparedInput:
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


def _resolve_activation_positions(
    requested: list[list[int]] | None, prepared: list[_PreparedInput]
) -> list[list[int]] | None:
    """Normalize requested model-input positions after all input transformations."""
    if requested is None:
        return None
    if len(requested) != len(prepared):
        raise ValueError(f"activationPositions has {len(requested)} entries but request has {len(prepared)} input(s)")

    resolved_by_input: list[list[int]] = []
    for input_index, (positions, row) in enumerate(zip(requested, prepared, strict=True)):
        if not positions:
            raise ValueError(f"activationPositions[{input_index}] must contain at least one position")
        seq_len = len(row.model_token_ids)
        seen: set[int] = set()
        resolved_positions: list[int] = []
        for raw_position in positions:
            if type(raw_position) is not int:
                raise ValueError(
                    f"activationPositions[{input_index}] contains malformed position {raw_position!r}; "
                    "positions must be integers"
                )
            position = raw_position
            if position < -1:
                raise ValueError(
                    f"activationPositions[{input_index}] contains invalid position {position}; "
                    "only -1 or nonnegative positions are supported"
                )
            resolved = seq_len - 1 if position == -1 else position
            if resolved < 0 or resolved >= seq_len:
                raise ValueError(
                    f"activationPositions[{input_index}] contains position {position} resolved to {resolved}, "
                    f"outside model input length {seq_len}"
                )
            if resolved in seen:
                raise ValueError(
                    f"activationPositions[{input_index}] contains duplicate position {resolved} after normalization"
                )
            seen.add(resolved)
            resolved_positions.append(resolved)
        resolved_by_input.append(resolved_positions)
    return resolved_by_input


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Number of leading positions on which two token sequences agree."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _causal_owner_by_position(capture_tokens: list[tuple[int, ...]]) -> list[list[int]]:
    """For each capture row and position, the earliest capture row sharing that causal prefix.

    A hidden state at position ``p`` depends only on ``tokens[:p + 1]``, so every capture row that
    agrees through ``p`` holds the same value there in exact arithmetic. Separate vLLM forwards do
    not honour that identity in reduced precision: the two rows of a shared-prefix pair land in
    different batch shapes or prefill chunks and disagree at ``p`` by a few bf16 ulps, which a TopK
    SAE turns into different feature values or support. Reading every causal prefix from one
    canonical row -- the first in request order that contains it -- makes the shared positions
    bitwise identical regardless of how the engine scheduled the rows.

    Consistency follows from the choice of "earliest": two rows that agree through ``p`` see the
    same set of earlier rows agreeing through ``p``, so they pick the same owner.
    """
    owners: list[list[int]] = []
    for index, tokens in enumerate(capture_tokens):
        row_owners = [index] * len(tokens)
        # Walk earlier rows from nearest to first so the earliest match is the one that sticks.
        for earlier in reversed(range(index)):
            shared = _common_prefix_length(capture_tokens[earlier], tokens)
            row_owners[:shared] = [earlier] * shared
        owners.append(row_owners)
    return owners


def _sparse_all_positions(
    encoded_by_capture: list[np.ndarray], capture_index: int, owners: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonzero ``(position, feature, value)`` triples for one row, honouring position owners.

    Positions owned by another capture row (typically just a shared BOS or chat header) are
    swapped in individually rather than by rebuilding the whole ``[seq, d_sae]`` array, which
    for an 80k-feature SAE is a ~130 MB copy per 400-token row. Output is sorted by
    ``(position, feature)``, the same order ``np.nonzero`` yields on an unshared row.
    """
    own = encoded_by_capture[capture_index]
    token_indices, feature_indices = np.nonzero(own)
    borrowed = [position for position, owner in enumerate(owners) if owner != capture_index]
    if not borrowed:
        return token_indices, feature_indices, own[token_indices, feature_indices]

    keep = ~np.isin(token_indices, borrowed)
    tokens = [token_indices[keep]]
    features = [feature_indices[keep]]
    values = [own[tokens[0], features[0]]]
    for position in borrowed:
        owner_row = encoded_by_capture[owners[position]][position]
        active = np.nonzero(owner_row)[0]
        tokens.append(np.full(active.shape, position, dtype=token_indices.dtype))
        features.append(active)
        values.append(owner_row[active])
    token_indices = np.concatenate(tokens)
    feature_indices = np.concatenate(features)
    activation_values = np.concatenate(values)
    order = np.lexsort((feature_indices, token_indices))
    return token_indices[order], feature_indices[order], activation_values[order]


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
            _PreparedInput(
                input_type, input_ids, model_ids, _decode_tokens(model, model_ids), alignment, input_positions
            )
        )
    return prepared


@router.post("/activation/source", responses={200: {"model": ActivationSourceResponse}})
# Shared admission is fine here: rows that must agree read one captured tensor (see
# _causal_owner_by_position), so nothing depends on how vLLM batches concurrent requests.
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
        """Capture each distinct forwarded row once and read every causal prefix from one row.

        Two levels of reuse, both request-scoped. Rows whose forwarded tokens (truncated after the
        latest requested position) are identical share one capture. Beyond that, any position whose
        causal prefix also appears in an earlier capture row is read from that earlier row, so a
        shared-prefix pair agrees bitwise on the prefix even when each row also asks for a position
        past it and therefore needs its own forward.
        """
        model = Model.get_instance()
        sae_manager = SAEManager.get_instance()
        config = Config.get_instance()
        batch_size = len(prepared)
        if batch_size > config.activation_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum of {config.activation_batch_size}")
        batch_token_limit = (
            config.activation_token_limit
            if batch_size == 1
            else config.activation_token_limit / config.activation_batch_size
        )
        for index, row in enumerate(prepared):
            if len(row.model_token_ids) > batch_token_limit:
                raise ValueError(
                    f"Input {index} is too long: {len(row.model_token_ids)} tokens, max is {int(batch_token_limit)}"
                )

        selected_positions_by_input = _resolve_activation_positions(request.activation_positions, prepared)
        unique_capture_tokens: list[tuple[int, ...]] = []
        capture_index_by_key: dict[tuple[int, ...], int] = {}
        capture_index_by_input: list[int] = []
        for input_index, row in enumerate(prepared):
            capture_end = (
                max(selected_positions_by_input[input_index]) + 1
                if selected_positions_by_input is not None
                else len(row.model_token_ids)
            )
            capture_key = tuple(row.model_token_ids[:capture_end])
            capture_index = capture_index_by_key.get(capture_key)
            if capture_index is None:
                capture_index = len(unique_capture_tokens)
                capture_index_by_key[capture_key] = capture_index
                unique_capture_tokens.append(capture_key)
            capture_index_by_input.append(capture_index)

        max_len = max(len(tokens) for tokens in unique_capture_tokens)
        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token")
        padded_tokens = torch.full(
            (len(unique_capture_tokens), max_len), int(pad_token_id), dtype=torch.long, device=config.device
        )
        original_lengths = [len(tokens) for tokens in unique_capture_tokens]
        for index, tokens in enumerate(unique_capture_tokens):
            padded_tokens[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=config.device)

        hook_name = sae_manager.get_sae_hook(request.source)
        cache = await capture_padded_cache_async(model, padded_tokens, original_lengths, [hook_name])
        sae = sae_manager.get_sae(request.source)
        owners_by_capture = _causal_owner_by_position(unique_capture_tokens)
        results: list[ActivationSourceResult] = []
        if selected_positions_by_input is None:
            encoded_by_capture: list[np.ndarray] = []
            for capture_index, seq_len in enumerate(original_lengths):
                with torch.no_grad():
                    activation_data = cache[hook_name][capture_index : capture_index + 1, :seq_len].to(config.device)
                    encoded_by_capture.append(sae.encode(activation_data)[0].float().cpu().numpy())
            for input_index, row in enumerate(prepared):
                capture_index = capture_index_by_input[input_index]
                token_indices, feature_indices, activation_values = _sparse_all_positions(
                    encoded_by_capture, capture_index, owners_by_capture[capture_index]
                )
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

        unique_sae_keys: list[tuple[int, int]] = []
        sae_index_by_key: dict[tuple[int, int], int] = {}
        sae_indices_by_input: list[list[int]] = []
        for input_index, positions in enumerate(selected_positions_by_input):
            owners = owners_by_capture[capture_index_by_input[input_index]]
            input_sae_indices: list[int] = []
            for position in positions:
                key = (owners[position], position)
                sae_index = sae_index_by_key.get(key)
                if sae_index is None:
                    sae_index = len(unique_sae_keys)
                    sae_index_by_key[key] = sae_index
                    unique_sae_keys.append(key)
                input_sae_indices.append(sae_index)
            sae_indices_by_input.append(input_sae_indices)

        selected_activations: np.ndarray | None = None
        if unique_sae_keys:
            selected_hidden_states = [
                cache[hook_name][capture_index, model_position] for capture_index, model_position in unique_sae_keys
            ]
            with torch.no_grad():
                selected_hidden = torch.stack(selected_hidden_states).to(config.device)
                selected_activations = sae.encode(selected_hidden).float().cpu().numpy()

        for row, positions, sae_indices in zip(
            prepared, selected_positions_by_input, sae_indices_by_input, strict=True
        ):
            active_features: dict[str, list[list[float]]] = {}
            assert selected_activations is not None
            for model_position, sae_index in zip(positions, sae_indices, strict=True):
                row_activations = selected_activations[sae_index]
                feature_indices = np.nonzero(row_activations)[0]
                activation_values = row_activations[feature_indices]
                for feature_idx, activation_value in zip(feature_indices, activation_values, strict=True):
                    active_features.setdefault(str(int(feature_idx)), []).append(
                        [int(model_position), round(float(activation_value), ROUND_DECIMALS)]
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
                    activation_positions=positions,
                    activeFeatures=active_features,
                )
            )
        return results
