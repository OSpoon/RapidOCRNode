import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getModelDirectory } from '../../src/models/download.js'
import { OCRTextStructurer } from '../../src/pipeline/structured-output.js'

const { disposeMock, generatorMock, pipelineMock } = vi.hoisted(() => ({
  disposeMock: vi.fn(),
  generatorMock: vi.fn(),
  pipelineMock: vi.fn(),
}))

vi.mock('@huggingface/transformers', () => ({
  pipeline: pipelineMock,
}))

beforeEach(() => {
  disposeMock.mockReset()
  generatorMock.mockReset()
  pipelineMock.mockReset()
  generatorMock.mockResolvedValue(makeResponse({ invoiceNumber: 'INV-001' }))
  pipelineMock.mockResolvedValue(Object.assign(generatorMock, { dispose: disposeMock }))
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('ocr text structurer', () => {
  it('defaults to CPU, loads the optional Q4F16 text model lazily, and grounds values in source lines', async () => {
    generatorMock.mockResolvedValueOnce(makeResponse({
      invoiceNumber: 'INV-001',
      totalAmount: '¥999.00',
    }))
    const structurer = new OCRTextStructurer()
    const lines = [
      { text: '发票号码 INV-001' },
      { text: '价税合计（小写）¥100.00' },
    ]
    const fields = { invoiceNumber: '发票号码', totalAmount: '价税合计' }

    expect(pipelineMock).not.toHaveBeenCalled()
    const result = await structurer.extract(lines, fields)

    expect(result).toMatchObject({
      evidence: { invoiceNumber: [0], totalAmount: [] },
      values: { invoiceNumber: 'INV-001', totalAmount: null },
      timings: {
        queue: expect.any(Number),
        modelLoad: expect.any(Number),
        generation: expect.any(Number),
        total: expect.any(Number),
      },
    })
    expect(pipelineMock).toHaveBeenCalledWith(
      'text-generation',
      'onnx-community/Qwen3-0.6B-ONNX',
      expect.objectContaining({
        device: 'cpu',
        dtype: 'q4f16',
        cache_dir: getModelDirectory(),
      }),
    )
    expect(generatorMock).toHaveBeenCalledOnce()
    expect(generatorMock).toHaveBeenCalledWith(expect.any(Array), {
      do_sample: false,
      max_length: null,
      max_new_tokens: null,
    })
    await structurer.close()
    expect(disposeMock).toHaveBeenCalledOnce()
  })

  it('uses a caller-selected model and token limit', async () => {
    const structurer = new OCRTextStructurer({
      modelId: 'local/receipt-model',
      device: 'webgpu',
      maxNewTokens: 1024,
      modelDirectory: './test-models',
    })
    await structurer.extract([{ text: '单据号 INV-001' }], { invoiceNumber: '单据号' })

    expect(pipelineMock).toHaveBeenCalledWith(
      'text-generation',
      'local/receipt-model',
      expect.objectContaining({
        device: 'webgpu',
        dtype: 'q4f16',
        cache_dir: getModelDirectory('./test-models'),
      }),
    )
    expect(generatorMock).toHaveBeenCalledWith(expect.any(Array), {
      do_sample: false,
      max_new_tokens: 1024,
    })
    await structurer.close()
  })

  it('rejects automatic device selection', () => {
    expect(() => new OCRTextStructurer({ device: 'auto' as never }))
      .toThrow('device must be selected explicitly; auto is not supported for structured extraction.')
  })

  it('skips model loading when OCR contains no non-empty lines', async () => {
    const structurer = new OCRTextStructurer()
    const result = await structurer.extract([{ text: ' ' }, { text: '' }], {
      invoiceNumber: '发票号码',
      totalAmount: '价税合计',
    })

    expect(result).toMatchObject({
      evidence: { invoiceNumber: [], totalAmount: [] },
      values: { invoiceNumber: null, totalAmount: null },
      timings: {
        queue: expect.any(Number),
        modelLoad: 0,
        generation: 0,
        total: expect.any(Number),
      },
    })
    expect(pipelineMock).not.toHaveBeenCalled()
    await structurer.close()
  })

  it('validates requested fields and rejects malformed model JSON', async () => {
    const structurer = new OCRTextStructurer()
    await expect(structurer.extract([], {})).rejects.toThrow('fields must contain at least one field.')
    await expect(structurer.extract([], { invoiceNumber: ' ' })).rejects.toThrow('Field description for "invoiceNumber" must be a non-empty string.')

    generatorMock.mockResolvedValueOnce(makeResponseText('not JSON'))
    await expect(structurer.extract([{ text: '发票号码 INV-001' }], { invoiceNumber: '发票号码' }))
      .rejects
      .toThrow('The text generation model did not return a JSON object.')
    await structurer.close()
  })

  it('waits for active extraction before disposing and rejects work after closing starts', async () => {
    const structurer = new OCRTextStructurer()
    let finishGeneration: (() => void) | undefined
    let announceGeneration: (() => void) | undefined
    const generationStarted = new Promise<void>((resolve) => {
      announceGeneration = resolve
    })
    generatorMock.mockImplementationOnce(async () => {
      announceGeneration?.()
      await new Promise<void>((resolve) => {
        finishGeneration = resolve
      })
      return makeResponse({
        invoiceNumber: 'INV-001',
      })
    })

    const extraction = structurer.extract([{ text: '发票号码 INV-001' }], { invoiceNumber: '发票号码' })
    await generationStarted
    const closing = structurer.close()

    expect(disposeMock).not.toHaveBeenCalled()
    await expect(structurer.extract([{ text: '发票号码 INV-002' }], { invoiceNumber: '发票号码' }))
      .rejects
      .toThrow('OCRTextStructurer is closing.')

    finishGeneration?.()
    await expect(extraction).resolves.toMatchObject({ values: { invoiceNumber: 'INV-001' } })
    await closing
    expect(disposeMock).toHaveBeenCalledOnce()
  })
})

function makeResponse(payload: Record<string, string | null>): unknown[] {
  return makeResponseText(JSON.stringify(payload))
}

function makeResponseText(content: string): unknown[] {
  return [{
    generated_text: [
      { role: 'assistant', content },
    ],
  }]
}
