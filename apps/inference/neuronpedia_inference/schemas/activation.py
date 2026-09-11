"""Wire models for ``/v1/activation/*``.

The endpoints split three ways: ``single``/``all``/``topk-by-token`` ask a different
question of the same SAE machinery, ``source`` returns the sparse form, and ``raw`` skips
SAEs entirely. Each has a batch variant whose response is just the singular one wrapped in
``results``.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from neuronpedia_inference.schemas.chat_template import ChatMessage
from neuronpedia_inference.schemas.common import BaseSchema

# The three request fields every SAE-backed activation endpoint shares are spelled out per
# class rather than inherited: they carry per-endpoint descriptions that end up in the
# published spec, and a shared base would flatten them into one wording.


class ActivationValues(BaseSchema):
    """One feature's activation over a prompt, plus its DFA attribution if computed."""

    values: list[StrictFloat]
    max_value: StrictFloat
    max_value_index: StrictInt
    dfa_values: list[StrictFloat] | None = None
    dfa_max_value: StrictFloat | None = None
    dfa_target_index: StrictInt | None = None


class ActivationSingleRequest(BaseSchema):
    """
    Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")
    """

    prompt: StrictStr = Field(description="Input text prompt to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source: StrictStr | None = Field(
        default=None,
        description='Source identifier - could be an SAE ID (eg 5-gemmascope-res-16k). Must be specified with "index", or not at all.',
    )
    index: StrictStr | None = Field(
        default=None, description='Index of the SAE. Must be specified with "source", or not at all.'
    )
    vector: list[StrictFloat] | None = Field(
        default=None, description='Custom vector to test activations. Must be specified with "hook".'
    )
    hook: StrictStr | None = Field(
        default=None, description='Hook that the custom vector applies to. Must be specified with "vector".'
    )


class ActivationSingleResponse(BaseSchema):
    """
    Response for NPActivationSingleRequest. Contains the activation values and tokenized prompt.
    """

    activation: ActivationValues
    tokens: list[StrictStr]
    tokens_is_special: list[StrictBool] = Field(
        description="Index-aligned with \"tokens\". True where the token is one of the tokenizer's special tokens (BOS, EOS, padding, chat turn markers, and so on) rather than content. Lets a client filter scaffolding without knowing any model family's token literals."
    )


class ActivationSingleBatchRequest(BaseSchema):
    """
    Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")
    """

    prompts: list[StrictStr] = Field(description="Input text prompts to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source: StrictStr | None = Field(
        default=None,
        description='Source identifier - could be an SAE ID (eg 5-gemmascope-res-16k). Must be specified with "index", or not at all.',
    )
    index: StrictStr | None = Field(
        default=None, description='Index of the SAE. Must be specified with "source", or not at all.'
    )
    vector: list[StrictFloat] | None = Field(
        default=None, description='Custom vector to test activations. Must be specified with "hook".'
    )
    hook: StrictStr | None = Field(
        default=None, description='Hook that the custom vector applies to. Must be specified with "vector".'
    )


class ActivationSingleBatchResult(BaseSchema):
    """One prompt's result within a batched single-feature request."""

    activation: ActivationValues
    tokens: list[StrictStr]
    tokens_is_special: list[StrictBool] = Field(
        description="Index-aligned with \"tokens\". True where the token is one of the tokenizer's special tokens (BOS, EOS, padding, chat turn markers, and so on) rather than content. Lets a client filter scaffolding without knowing any model family's token literals."
    )


class ActivationSingleBatchResponse(BaseSchema):
    """
    Response for NPActivationBatchRequest. Contains the batch results of activation values and tokenized prompt.
    """

    results: list[ActivationSingleBatchResult]


class ActivationAllFeature(BaseSchema):
    """
    One feature and its activation in an NPActivationAllResponse
    """

    source: StrictStr
    index: StrictInt
    values: list[StrictFloat]
    sum_values: StrictFloat | None = None
    max_value: StrictFloat
    max_value_index: StrictInt
    dfa_values: list[StrictFloat] | None = None
    dfa_target_index: StrictInt | None = None
    dfa_max_value: StrictFloat | None = None


class ActivationAllRequest(BaseSchema):
    """
    For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
    """

    prompt: StrictStr = Field(description="Input text prompt to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source_set: StrictStr = Field(description="The source set name of the SAEs (eg gemmascope-res-16k)")
    selected_sources: list[StrictStr] = Field(
        description='List of specific SAEs to get activations for (eg ["0-gemmascope-res-65k", "5-gemmascope-res-65k"]). If not specified, will get activations for all SAEs in the source set.'
    )
    sort_by_token_indexes: list[StrictInt] = Field(
        description="Sort the results by the sum of the activations at the specified token indexes."
    )
    ignore_bos: StrictBool = Field(
        description="Whether or not to include features whose highest activation value is the BOS token."
    )
    num_results: StrictInt | None = Field(default=25, description="Optional. The number of top features to return.")


class ActivationAllResponse(BaseSchema):
    """
    Response for NPActivationAllRequest. Contains activations for each top feature and the tokenized prompt.
    """

    activations: list[ActivationAllFeature]
    tokens: list[StrictStr]
    counts: list[list[StrictFloat]] | None = Field(
        default=None,
        description="Not currently supported and may be incorrect. This is the number of features that activated by layer, starting from layer 0 of this SAE. Need to be redesigned.",
    )


class ActivationAllBatchRequest(BaseSchema):
    """
    For a given batch of prompts, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
    """

    prompts: list[StrictStr] = Field(description="Input text prompts to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source_set: StrictStr = Field(description="The source set name of the SAEs (eg gemmascope-res-16k)")
    selected_sources: list[StrictStr] = Field(
        description='List of specific SAEs to get activations for (eg ["0-gemmascope-res-65k", "5-gemmascope-res-65k"]). If not specified, will get activations for all SAEs in the source set.'
    )
    sort_by_token_indexes: list[StrictInt] = Field(
        description="Sort the results by the sum of the activations at the specified token indexes."
    )
    ignore_bos: StrictBool = Field(
        description="Whether or not to include features whose highest activation value is the BOS token."
    )
    num_results: StrictInt | None = Field(default=25, description="Optional. The number of top features to return.")


class ActivationAllBatchResult(BaseSchema):
    """One prompt's result within a batched top-features request."""

    activations: list[ActivationAllFeature]
    tokens: list[StrictStr]
    counts: list[list[StrictFloat]] | None = Field(
        default=None,
        description="Not currently supported and may be incorrect. This is the number of features that activated by layer, starting from layer 0 of this SAE. Need to be redesigned.",
    )


class ActivationAllBatchResponse(BaseSchema):
    """
    Response for NPActivationAllBatchRequest. Contains the batch results of activations for each top feature and the tokenized prompts.
    """

    results: list[ActivationAllBatchResult]


class ActivationTopkByTokenFeature(BaseSchema):
    """One feature active at a single token position."""

    feature_index: StrictInt = Field(description="The index of the feature in the SAE.")
    activation_value: StrictFloat = Field(description="The activation value of this feature at this token position.")


class ActivationTopkByTokenResult(BaseSchema):
    """
    One token's TopK result, including its top features.
    """

    token_position: StrictInt = Field(description="The index of the token in the prompt.")
    token: StrictStr = Field(description="The token string")
    is_special: StrictBool = Field(
        description="Whether this token is one of the tokenizer's special tokens (BOS, EOS, padding, chat turn markers, and so on) rather than content. Lets a client filter scaffolding without knowing any model family's token literals."
    )
    top_features: list[ActivationTopkByTokenFeature]


class ActivationTopkByTokenRequest(BaseSchema):
    """
    Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")
    """

    prompt: StrictStr = Field(description="Input text prompt to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source: StrictStr = Field(
        description='Source identifier - could be an SAE ID (eg 5-gemmascope-res-16k). Must be specified with "index", or not at NPActivationAllRequest.'
    )
    top_k: StrictInt | None = Field(
        default=None, description="The number of features to include for each token position."
    )
    ignore_bos: StrictBool = Field(
        description="Whether or not to include features whose highest activation value is the BOS token."
    )


class ActivationTopkByTokenResponse(BaseSchema):
    """
    Response for NPActivationTopkByTokenRequest.
    """

    results: list[ActivationTopkByTokenResult]
    tokens: list[StrictStr]


class ActivationTopkByTokenBatchRequest(BaseSchema):
    """
    Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")
    """

    prompts: list[StrictStr] = Field(description="Input text prompts to get activations for")
    model: StrictStr = Field(description="Name of the model to test activations on")
    source: StrictStr = Field(
        description='Source identifier - could be an SAE ID (eg 5-gemmascope-res-16k). Must be specified with "index", or not at NPActivationAllRequest.'
    )
    top_k: StrictInt | None = Field(
        default=None, description="The number of features to include for each token position."
    )
    ignore_bos: StrictBool = Field(
        description="Whether or not to include features whose highest activation value is the BOS token."
    )


class ActivationTopkByTokenBatchResult(BaseSchema):
    """One prompt's result within a batched topk-by-token request."""

    results: list[ActivationTopkByTokenResult]
    tokens: list[StrictStr]


class ActivationTopkByTokenBatchResponse(BaseSchema):
    """
    Response for NPActivationTopkByTokenBatchRequest. Contains the batch results of top features at each token position and the tokenized prompts.
    """

    results: list[ActivationTopkByTokenBatchResult]


class TokenInsertionMode(StrEnum):
    """Whether a tokenizer-owned boundary token should be inserted."""

    NEVER = "never"
    IF_MISSING = "if_missing"
    ALWAYS = "always"


class ActivationSourceInsertion(BaseSchema):
    """Explicit additions around a text or exact-token activation input."""

    bos: TokenInsertionMode = TokenInsertionMode.NEVER
    eos: TokenInsertionMode = TokenInsertionMode.NEVER
    prefix_token_ids: list[StrictInt] = Field(default_factory=list)
    suffix_token_ids: list[StrictInt] = Field(default_factory=list)


class ActivationSourceTextInput(BaseSchema):
    """One raw-text row in a discriminated activation batch."""

    type: Literal["text"]
    text: StrictStr
    insertion: ActivationSourceInsertion = Field(default_factory=ActivationSourceInsertion)


class ActivationSourceTokensInput(BaseSchema):
    """One exact-token row in a discriminated activation batch."""

    type: Literal["tokens"]
    token_ids: list[StrictInt] = Field(min_length=1)
    insertion: ActivationSourceInsertion = Field(default_factory=ActivationSourceInsertion)


class ActivationSourceChatInput(BaseSchema):
    """One conversation rendered by the selected model's configured chat template."""

    type: Literal["chat"]
    messages: list[ChatMessage] = Field(min_length=1)
    apply_chat_template: Literal[True] = True
    add_generation_prompt: StrictBool = True
    continue_final_message: StrictBool = False

    @model_validator(mode="after")
    def check_generation_mode(self) -> "ActivationSourceChatInput":
        """The tokenizer APIs define these as mutually exclusive render modes."""
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError("addGenerationPrompt and continueFinalMessage cannot both be true")
        return self


ActivationSourceInput = Annotated[
    ActivationSourceTextInput | ActivationSourceTokensInput | ActivationSourceChatInput,
    Field(discriminator="type"),
]


class ActivationSourceRequest(BaseSchema):
    """
    For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index.
    """

    prompts: list[StrictStr] | None = Field(default=None, min_length=1, max_length=4)
    prompt_token_ids: list[list[StrictInt]] | None = Field(default=None, min_length=1, max_length=4)
    inputs: list[ActivationSourceInput] | None = Field(default=None, min_length=1, max_length=4)
    insertion: ActivationSourceInsertion | None = Field(
        default=None,
        description="Uniform insertion policy for prompts or promptTokenIds. Use per-input policies with inputs.",
    )
    model: StrictStr = Field(description="Name of the model to test activations on")
    source: StrictStr = Field(description="The source (eg 5-gemmascope-res-16k)")

    @model_validator(mode="after")
    def check_input_mode(self) -> "ActivationSourceRequest":
        """Require one unambiguous batch representation."""
        supplied = [self.prompts is not None, self.prompt_token_ids is not None, self.inputs is not None]
        if sum(supplied) != 1:
            raise ValueError("exactly one of prompts, promptTokenIds, or inputs must be provided")
        if self.inputs is not None and self.insertion is not None:
            raise ValueError("top-level insertion cannot be used with inputs; set insertion on each text/token input")
        if self.prompt_token_ids is not None:
            for index, row in enumerate(self.prompt_token_ids):
                if not row:
                    raise ValueError(f"promptTokenIds[{index}] must contain at least one token ID")
        return self


class ActivationSourceOrigin(BaseSchema):
    """Where bytes represented by one chat-template token originated."""

    type: Literal["chat_template", "message_content", "bos"]
    message_index: StrictInt | None = None
    message_role: StrictStr | None = None
    content_byte_start: StrictInt | None = None
    content_byte_end: StrictInt | None = None


class ActivationSourceTokenAlignment(BaseSchema):
    """One model token's exact position and provenance."""

    model_position: StrictInt
    token_id: StrictInt
    token_text: StrictStr
    input_position: StrictInt | None = None
    source: Literal["provided", "bos", "eos", "prefix", "suffix", "chat_template", "message"]
    token_bytes: StrictStr | None = None
    rendered_byte_start: StrictInt | None = None
    rendered_byte_end: StrictInt | None = None
    origins: list[ActivationSourceOrigin] | None = None


class ActivationSourceResult(BaseSchema):
    """
    One prompt's results, only including non-zero values and non-zero activations
    """

    tokens: list[StrictStr] = Field(description="The prompt, tokenized.")
    input_type: Literal["text", "tokens", "chat"]
    input_token_ids: list[StrictInt] | None = None
    model_input_token_ids: list[StrictInt]
    token_alignment: list[ActivationSourceTokenAlignment]
    input_to_model_positions: list[StrictInt] | None = None
    rendered_text: StrictStr | None = None
    active_features: dict[str, list[Annotated[list[StrictFloat], Field(min_length=2, max_length=2)]]] | None = Field(
        default=None,
        description="Dictionary mapping feature indices to arrays of [token_index, activation_value]",
        alias="activeFeatures",
    )


class ActivationSourceResponse(BaseSchema):
    """
    All prompts results, only including non-zero features and non-zero activations
    """

    results: list[ActivationSourceResult]


# The models below were hand-written in their endpoint files rather than generated, and keep
# their loose `str`/`int`/`float` annotations. The JSON Schema is identical either way, but
# `float` coerces an integer input where `StrictFloat` rejects it, so tightening them here
# would be a silent validation change rather than a move.


class ActivationAttentionRequest(BaseSchema):
    """Attention pattern for one (layer, head) of the model."""

    prompt: str
    model: str
    # Integer layer + head. Attention heads are not SAE/source-based, so we index
    # the model's attention layers/query-heads directly.
    layer: int
    head: int


class ActivationAttentionResponse(BaseSchema):
    """A sparse COO attention matrix for one head.

    Only the top keys per query row survive, each encoded as a flat ``q * seq_len + k``
    index, so the two arrays are index-aligned with each other rather than with the tokens.
    """

    seq_len: int
    attention_indices: list[int]
    attention_values: list[float]
    # Largest weight outside row 0 / column 0, which is the position-0 attention sink and
    # would otherwise dominate every head.
    max_activation: float
    tokens: list[str]


class ActivationRawRequest(BaseSchema):
    """Residual-stream vectors at each prompt's final token, with no SAE involved."""

    prompts: list[str]
    # Absent or empty means every layer. Out-of-range entries are rejected rather than
    # clamped: silently returning a different set of layers than was asked for would be
    # indistinguishable from the model having fewer layers than the caller thinks.
    layers: list[int] | None = None
    model: str | None = None
    hook_point: str = Field(default="residual_stream")
    type: str = Field(default="final_output_token")


class ActivationRawLayer(BaseSchema):
    """The captured vectors for one layer of one prompt."""

    layer: int
    token_indices: list[int]
    values: list[list[float]]


class ActivationRawPromptResult(BaseSchema):
    """One prompt's tokens and its per-layer captured vectors."""

    token_strings: list[str]
    token_ids: list[int]
    activations: list[ActivationRawLayer]


class ActivationRawResponse(BaseSchema):
    """Raw capture results, echoing the capture settings they were produced under."""

    hook_point: str
    type: str
    dtype: str
    device: str
    results: list[ActivationRawPromptResult]
