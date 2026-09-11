import type { ActivationSourceInput, ActivationSourceInsertion } from '@/lib/api/inference-types';
import { assertUserCanAccessModelAndSourceSet } from '@/lib/db/userCanAccess';
import { InferenceServerError, runInferenceActivationSource } from '@/lib/utils/inference';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';
import * as yup from 'yup';

const insertionSchema = yup.object({
  bos: yup.string().oneOf(['never', 'if_missing', 'always']).default('never'),
  eos: yup.string().oneOf(['never', 'if_missing', 'always']).default('never'),
  prefix_token_ids: yup.array().of(yup.number().integer().required()).default([]),
  suffix_token_ids: yup.array().of(yup.number().integer().required()).default([]),
});

const messageSchema = yup.object({ role: yup.string().required(), content: yup.string().required() });
const inputSchema = yup.lazy((value: unknown) => {
  const type = value && typeof value === 'object' ? (value as { type?: unknown }).type : undefined;
  if (type === 'text') {
    return yup.object({
      type: yup.string().oneOf(['text']).required(),
      text: yup.string().required(),
      insertion: insertionSchema.default(undefined),
    });
  }
  if (type === 'tokens') {
    return yup.object({
      type: yup.string().oneOf(['tokens']).required(),
      token_ids: yup.array().of(yup.number().integer().required()).min(1).required(),
      insertion: insertionSchema.default(undefined),
    });
  }
  if (type === 'chat') {
    return yup.object({
      type: yup.string().oneOf(['chat']).required(),
      messages: yup.array().of(messageSchema).min(1).required(),
      apply_chat_template: yup.boolean().oneOf([true]).default(true),
      add_generation_prompt: yup.boolean().default(true),
      continue_final_message: yup.boolean().default(false),
    });
  }
  return yup.mixed().test(() => false);
});

const activationSourceSchema = yup
  .object({
    modelId: yup.string().required('modelId is required'),
    source: yup.string().required('source is required'),
    customText: yup.lazy((value) =>
      value === undefined
        ? yup.mixed().notRequired()
        : Array.isArray(value)
          ? yup.array().of(yup.string().required()).min(1).max(4).required()
          : yup.string().required(),
    ),
    prompts: yup.array().of(yup.string().required()).min(1).max(4),
    prompt_token_ids: yup.array().of(yup.array().of(yup.number().integer().required()).min(1).required()).min(1).max(4),
    inputs: yup.array().of(inputSchema).min(1).max(4),
    insertion: insertionSchema.default(undefined),
  })
  .test(
    'one-input-mode',
    'exactly one of customText, prompts, prompt_token_ids, or inputs must be provided',
    (body) => {
      if (!body) return false;
      return (
        [
          body.customText !== undefined,
          body.prompts !== undefined,
          body.prompt_token_ids !== undefined,
          body.inputs !== undefined,
        ].filter(Boolean).length === 1
      );
    },
  );

type PublicInsertion = yup.InferType<typeof insertionSchema>;
type PublicInput =
  | { type: 'text'; text: string; insertion?: PublicInsertion }
  | { type: 'tokens'; token_ids: number[]; insertion?: PublicInsertion }
  | {
      type: 'chat';
      messages: Array<{ role: string; content: string; channel?: string | null }>;
      apply_chat_template?: true;
      add_generation_prompt?: boolean;
      continue_final_message?: boolean;
    };

const toInferenceInsertion = (insertion: PublicInsertion | undefined): ActivationSourceInsertion | undefined =>
  insertion
    ? {
        bos: insertion.bos,
        eos: insertion.eos,
        prefixTokenIds: insertion.prefix_token_ids,
        suffixTokenIds: insertion.suffix_token_ids,
      }
    : undefined;

const toInferenceInput = (input: PublicInput): ActivationSourceInput => {
  if (input.type === 'text') {
    return { type: 'text', text: input.text, insertion: toInferenceInsertion(input.insertion as PublicInsertion) };
  }
  if (input.type === 'tokens') {
    return {
      type: 'tokens',
      tokenIds: input.token_ids,
      insertion: toInferenceInsertion(input.insertion as PublicInsertion),
    };
  }
  return {
    type: 'chat',
    messages: input.messages,
    applyChatTemplate: input.apply_chat_template ?? true,
    addGenerationPrompt: input.add_generation_prompt ?? true,
    continueFinalMessage: input.continue_final_message ?? false,
  };
};

const toPublicResponse = (response: Record<string, unknown>) => ({
  results: ((response.results as Array<Record<string, unknown>>) || []).map((result) => ({
    tokens: result.tokens,
    activeFeatures: result.activeFeatures,
    input_type: result.inputType,
    ...(result.inputTokenIds === undefined ? {} : { input_token_ids: result.inputTokenIds }),
    model_input_token_ids: result.modelInputTokenIds,
    ...(result.inputToModelPositions === undefined ? {} : { input_to_model_positions: result.inputToModelPositions }),
    ...(result.renderedText === undefined ? {} : { rendered_text: result.renderedText }),
    token_alignment: ((result.tokenAlignment as Array<Record<string, unknown>>) || []).map((entry) => ({
      model_position: entry.modelPosition,
      token_id: entry.tokenId,
      token_text: entry.tokenText,
      ...(entry.inputPosition === undefined ? {} : { input_position: entry.inputPosition }),
      source: entry.source,
      ...(entry.tokenBytes === undefined ? {} : { token_bytes: entry.tokenBytes }),
      ...(entry.renderedByteStart === undefined ? {} : { rendered_byte_start: entry.renderedByteStart }),
      ...(entry.renderedByteEnd === undefined ? {} : { rendered_byte_end: entry.renderedByteEnd }),
      ...(entry.origins === undefined
        ? {}
        : {
            origins: (entry.origins as Array<Record<string, unknown>>).map((origin) => ({
              type: origin.type,
              ...(origin.messageIndex === undefined ? {} : { message_index: origin.messageIndex }),
              ...(origin.messageRole === undefined ? {} : { message_role: origin.messageRole }),
              ...(origin.contentByteStart === undefined ? {} : { content_byte_start: origin.contentByteStart }),
              ...(origin.contentByteEnd === undefined ? {} : { content_byte_end: origin.contentByteEnd }),
            })),
          }),
    })),
  })),
});

/**
 * @swagger
 * /api/activation/source:
 *   post:
 *     summary: Extract sparse SAE activations from text, exact token IDs, or templated chats
 *     description: Existing customText requests remain supported. Exact token IDs are forwarded without decoding or retokenization.
 *     tags: [Activations]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [modelId, source]
 *             properties:
 *               modelId: { type: string, example: gemma-3-4b-it }
 *               source: { type: string, example: 22-gemmascope-2-res-16k }
 *               customText: { oneOf: [{type: string}, {type: array, items: {type: string}}] }
 *               prompts: { type: array, items: {type: string}, maxItems: 4 }
 *               prompt_token_ids:
 *                 type: array
 *                 maxItems: 4
 *                 items: { type: array, minItems: 1, items: {type: integer} }
 *               inputs: { type: array, maxItems: 4, description: Discriminated text, tokens, or chat inputs. }
 *               insertion: { type: object, description: Explicit BOS/EOS and prefix/suffix token insertion. }
 *     responses:
 *       200: { description: Ordered sparse activations with exact token alignment }
 *       400: { description: Invalid request or model-specific token ID }
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  let body: yup.InferType<typeof activationSourceSchema>;
  try {
    body = await activationSourceSchema.validate(await request.json(), { abortEarly: false });
  } catch (error) {
    return NextResponse.json(
      { message: error instanceof Error ? error.message : 'Invalid request body' },
      { status: 400 },
    );
  }

  try {
    await assertUserCanAccessModelAndSourceSet(body.modelId, getSourceSetNameFromSource(body.source), request.user);
    const customText = body.customText as string | string[] | undefined;
    const input =
      customText !== undefined
        ? typeof customText === 'string'
          ? [customText]
          : customText
        : {
            ...(body.prompts === undefined ? {} : { prompts: body.prompts }),
            ...(body.prompt_token_ids === undefined ? {} : { promptTokenIds: body.prompt_token_ids }),
            ...(body.inputs === undefined
              ? {}
              : { inputs: body.inputs.map((item) => toInferenceInput(item as PublicInput)) }),
            ...(body.insertion === undefined ? {} : { insertion: toInferenceInsertion(body.insertion) }),
          };
    const activation = await runInferenceActivationSource(body.modelId, body.source, input, request.user);
    return NextResponse.json(toPublicResponse(activation as unknown as Record<string, unknown>));
  } catch (error) {
    const status = error instanceof InferenceServerError ? error.status : 500;
    return NextResponse.json({ message: error instanceof Error ? error.message : 'Unknown Error' }, { status });
  }
});
