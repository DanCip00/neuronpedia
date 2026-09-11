import { beforeEach, describe, expect, it, vi } from 'vitest';

const { runInferenceActivationSource } = vi.hoisted(() => ({ runInferenceActivationSource: vi.fn() }));

vi.mock('@/lib/utils/inference', () => ({
  InferenceServerError: class InferenceServerError extends Error {
    status = 400;
  },
  runInferenceActivationSource,
}));
vi.mock('@/lib/db/userCanAccess', () => ({ assertUserCanAccessModelAndSourceSet: vi.fn() }));
vi.mock('@/lib/utils/source', () => ({ getSourceSetNameFromSource: () => 'gemmascope-2-res-16k' }));
vi.mock('@/lib/with-user', () => ({
  withOptionalUser: (handler: (request: Request & { user: null }) => Promise<Response>) => (request: Request) =>
    handler(Object.assign(request, { user: null })),
}));

import { POST } from './route';

describe('/api/activation/source', () => {
  beforeEach(() => runInferenceActivationSource.mockReset());

  it('forwards exact IDs unchanged and returns alignment in the public casing', async () => {
    const rows = [
      [818, 5279, 529, 7001, 563, 9079],
      [818, 5279, 529, 7001, 563, 5860],
    ];
    runInferenceActivationSource.mockResolvedValue({
      results: rows.map((ids) => ({
        inputType: 'tokens',
        inputTokenIds: ids,
        modelInputTokenIds: ids,
        inputToModelPositions: ids.map((_, index) => index),
        tokens: ids.map(String),
        activeFeatures: { '7': [[5, 1.25]] },
        tokenAlignment: ids.map((tokenId, modelPosition) => ({
          modelPosition,
          tokenId,
          tokenText: String(tokenId),
          inputPosition: modelPosition,
          source: 'provided',
        })),
      })),
    });

    const response = await POST(
      new Request('http://localhost/api/activation/source', {
        method: 'POST',
        body: JSON.stringify({
          modelId: 'gemma-3-4b-it',
          source: '22-gemmascope-2-res-16k',
          prompt_token_ids: rows,
          insertion: { bos: 'never', eos: 'never', prefix_token_ids: [], suffix_token_ids: [] },
        }),
      }) as never,
    );

    expect(response.status).toBe(200);
    expect(runInferenceActivationSource).toHaveBeenCalledWith(
      'gemma-3-4b-it',
      '22-gemmascope-2-res-16k',
      {
        promptTokenIds: rows,
        insertion: { bos: 'never', eos: 'never', prefixTokenIds: [], suffixTokenIds: [] },
      },
      null,
    );
    const payload = await response.json();
    expect(payload.results.map((result: { model_input_token_ids: number[] }) => result.model_input_token_ids)).toEqual(
      rows,
    );
    expect(payload.results[0].token_alignment[5]).toMatchObject({
      model_position: 5,
      token_id: 9079,
      input_position: 5,
    });
  });

  it('keeps the legacy customText call shape and implicit engine behavior', async () => {
    runInferenceActivationSource.mockResolvedValue({ results: [] });
    const response = await POST(
      new Request('http://localhost/api/activation/source', {
        method: 'POST',
        body: JSON.stringify({ modelId: 'gpt2-small', source: '7-res-jb', customText: 'Hello' }),
      }) as never,
    );

    expect(response.status).toBe(200);
    expect(runInferenceActivationSource).toHaveBeenCalledWith('gpt2-small', '7-res-jb', ['Hello'], null);
  });
});
