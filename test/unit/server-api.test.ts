import type { RapidOCRResult } from '../../src/shared/types.js'
import { Buffer } from 'node:buffer'
import sharp from 'sharp'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { RapidOCR } from '../../src/pipeline/rapidocr.js'
import { createRapidOCRServer } from '../../src/server/api.js'

const openServers: Array<Awaited<ReturnType<typeof createRapidOCRServer>>> = []
let sampleImage: Buffer

beforeAll(async () => {
  sampleImage = await sharp({ create: { width: 2, height: 2, channels: 3, background: '#ffffff' } }).png().toBuffer()
})

afterEach(async () => {
  await Promise.all(openServers.splice(0).map(server => server.close()))
  vi.restoreAllMocks()
})

describe('rapidOCR HTTP API', () => {
  it('reports liveness/readiness and serves the OpenAPI document', async () => {
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize: vi.fn() } as never, ready: false })
    openServers.push(app)

    expect((await app.inject('/health/live')).json()).toEqual({ status: 'live' })
    expect((await app.inject('/health/ready')).statusCode).toBe(503)
    const document = (await app.inject('/openapi.json')).json()
    expect(document.openapi).toBe('3.1.0')
    expect(document.paths['/v1/ocr'].post.operationId).toBe('recognizeImage')
    expect(document.components.schemas.OcrOptions.properties.structuredFields).toMatchObject({ type: 'object', minProperties: 1 })
    expect(document.components.schemas.OcrResult.properties.structured.$ref).toBe('#/components/schemas/StructuredResult')
    app.setRapidOcrReady(true)
    expect((await app.inject('/health/ready')).statusCode).toBe(200)
  })

  it('accepts the same flat OCR options as RapidOCR.create', async () => {
    const fakeOcr = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => makeResult()),
      warmup: vi.fn(async () => {}),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(fakeOcr as never)
    const app = await createRapidOCRServer({
      logger: false,
      executionProviders: ['webgpu', 'cpu'],
      detectionExecutionProviders: ['cpu'],
      recognitionExecutionProviders: ['webgpu', 'cpu'],
    })
    openServers.push(app)

    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      executionProviders: ['webgpu', 'cpu'],
      detectionExecutionProviders: ['cpu'],
      recognitionExecutionProviders: ['webgpu', 'cpu'],
    }))
  })

  it('runs one full OCR call and serializes the upstream fields for JSON transport', async () => {
    const recognize = vi.fn(async () => makeResult())
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize } as never })
    openServers.push(app)
    const { headers, payload } = multipartImage({
      file: sampleImage,
      options: JSON.stringify({ useDetection: false, boxThreshold: 0.4, returnWordBoxes: true, includeImage: true }),
    })

    const response = await app.inject({ method: 'POST', url: '/v1/ocr', headers, payload })
    expect(response.statusCode).toBe(200)
    expect(response.headers['content-type']).toContain('application/json')
    expect(response.json()).toMatchObject({
      type: 'ocr',
      img: { data: 'AQID', encoding: 'base64', format: 'BGR', height: 1, width: 1 },
      boxes: [],
      txts: [],
      scores: [],
      word_results: [],
      elapse_list: [0.1, 0.2, 0.3],
      elapse: 0.6,
    })
    expect(response.json().request_id).toBeTruthy()
    expect(recognize).toHaveBeenCalledWith(expect.any(Buffer), expect.objectContaining({
      useDetection: false,
      boxThreshold: 0.4,
      returnWordBoxes: true,
    }))
  })

  it('preserves empty-result fields and stage timings', async () => {
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize: vi.fn(async () => makeEmptyResult()) } as never })
    openServers.push(app)
    const upload = multipartImage({ file: sampleImage })

    const response = await app.inject({ method: 'POST', url: '/v1/ocr', headers: upload.headers, payload: upload.payload })
    expect(response.statusCode).toBe(200)
    expect(response.json()).toMatchObject({
      type: 'empty',
      img: null,
      boxes: null,
      txts: null,
      scores: null,
      word_results: [],
      elapse_list: [0.25, 0, 0],
      elapse: 0.25,
    })
  })

  it('extracts request-defined structured fields and includes the result in the OCR response', async () => {
    const recognize = vi.fn(async () => ({
      ...makeResult(),
      lines: [{ text: '发票号码：12345678' }],
    }))
    const extract = vi.fn(async () => ({
      values: { invoiceNumber: '12345678' },
      evidence: { invoiceNumber: [0] },
    }))
    const close = vi.fn(async () => {})
    const app = await createRapidOCRServer({
      logger: false,
      ocr: { recognize } as never,
      structurer: { extract, close } as never,
    })
    openServers.push(app)
    const upload = multipartImage({
      file: sampleImage,
      options: JSON.stringify({
        includeImage: false,
        structuredFields: { invoiceNumber: '发票号码' },
      }),
    })

    const response = await app.inject({ method: 'POST', url: '/v1/ocr', headers: upload.headers, payload: upload.payload })
    expect(response.statusCode).toBe(200)
    expect(response.json()).toMatchObject({
      type: 'ocr',
      img: null,
      structured: {
        values: { invoiceNumber: '12345678' },
        evidence: { invoiceNumber: [0] },
      },
    })
    expect(extract).toHaveBeenCalledWith([{ text: '发票号码：12345678' }], { invoiceNumber: '发票号码' })
    await app.close()
    openServers.splice(openServers.indexOf(app), 1)
    expect(close).not.toHaveBeenCalled()
  })

  it('rejects invalid structured field definitions without calling the structurer', async () => {
    const recognize = vi.fn(async () => makeResult())
    const extract = vi.fn()
    const app = await createRapidOCRServer({
      logger: false,
      ocr: { recognize } as never,
      structurer: { extract, close: vi.fn() } as never,
    })
    openServers.push(app)
    const upload = multipartImage({ file: sampleImage, options: JSON.stringify({ structuredFields: {} }) })

    const response = await app.inject({ method: 'POST', url: '/v1/ocr', headers: upload.headers, payload: upload.payload })
    expect(response.statusCode).toBe(400)
    expect(response.json().code).toBe('invalid_option')
    expect(recognize).not.toHaveBeenCalled()
    expect(extract).not.toHaveBeenCalled()
  })

  it('returns problem details for malformed options, missing images, and non-multipart requests', async () => {
    const recognize = vi.fn(async () => makeResult())
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize } as never })
    openServers.push(app)

    const malformed = multipartImage({ file: sampleImage, options: '{bad', optionsContentType: 'text/plain' })
    const invalidOptions = await app.inject({ method: 'POST', url: '/v1/ocr', headers: malformed.headers, payload: malformed.payload })
    expect(invalidOptions.statusCode).toBe(400)
    expect(invalidOptions.headers['content-type']).toContain('application/problem+json')
    expect(invalidOptions.json()).toMatchObject({ type: 'about:blank', code: 'invalid_options' })

    const empty = multipartImage({})
    const missingFile = await app.inject({ method: 'POST', url: '/v1/ocr', headers: empty.headers, payload: empty.payload })
    expect(missingFile.statusCode).toBe(400)
    expect(missingFile.json().code).toBe('missing_file')

    const unsupported = await app.inject({ method: 'POST', url: '/v1/ocr', payload: '{}' })
    expect(unsupported.statusCode).toBe(415)
    expect(recognize).not.toHaveBeenCalled()

    const notFound = await app.inject('/unknown')
    expect(notFound.statusCode).toBe(404)
    expect(notFound.json().code).toBe('not_found')
  })

  it('rejects oversized uploads and maps image decoding failures to 422', async () => {
    const recognize = vi.fn(async () => makeResult())
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize } as never, maxFileSize: 32 })
    openServers.push(app)

    const oversized = multipartImage({ file: Buffer.alloc(64) })
    const tooLarge = await app.inject({ method: 'POST', url: '/v1/ocr', headers: oversized.headers, payload: oversized.payload })
    expect(tooLarge.statusCode).toBe(413)
    expect(tooLarge.json().code).toBe('file_too_large')

    const invalidImage = multipartImage({ file: Buffer.from([1, 2, 3]) })
    const decodeError = await app.inject({ method: 'POST', url: '/v1/ocr', headers: invalidImage.headers, payload: invalidImage.payload })
    expect(decodeError.statusCode).toBe(422)
    expect(decodeError.json().code).toBe('invalid_image')
  })

  it('rejects unknown options and omits image bytes when requested', async () => {
    const recognize = vi.fn(async () => makeResult())
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize } as never })
    openServers.push(app)

    const invalid = multipartImage({ file: sampleImage, options: JSON.stringify({ image_url: 'https://example.invalid' }) })
    const invalidResponse = await app.inject({ method: 'POST', url: '/v1/ocr', headers: invalid.headers, payload: invalid.payload })
    expect(invalidResponse.statusCode).toBe(400)
    expect(invalidResponse.json().code).toBe('unknown_option')

    const oldSnakeCase = multipartImage({ file: sampleImage, options: JSON.stringify({ include_image: false }) })
    const oldNameResponse = await app.inject({ method: 'POST', url: '/v1/ocr', headers: oldSnakeCase.headers, payload: oldSnakeCase.payload })
    expect(oldNameResponse.statusCode).toBe(400)
    expect(oldNameResponse.json().code).toBe('unknown_option')

    const withoutImage = multipartImage({ file: sampleImage, options: JSON.stringify({ includeImage: false }) })
    const response = await app.inject({ method: 'POST', url: '/v1/ocr', headers: withoutImage.headers, payload: withoutImage.payload })
    expect(response.statusCode).toBe(200)
    expect(response.json().img).toBeNull()
  })

  it('bounds concurrent recognition and gives callers a retry hint', async () => {
    let allowRecognition: ((result: RapidOCRResult) => void) | undefined
    let announceRecognition: (() => void) | undefined
    const recognitionStarted = new Promise<void>((resolve) => {
      announceRecognition = resolve
    })
    const recognize = vi.fn(() => new Promise<RapidOCRResult>((resolve) => {
      allowRecognition = resolve
      announceRecognition?.()
    }))
    const app = await createRapidOCRServer({ logger: false, ocr: { recognize } as never, maxConcurrency: 1 })
    openServers.push(app)
    const firstPayload = multipartImage({ file: sampleImage })
    const firstRequest = app.inject({ method: 'POST', url: '/v1/ocr', headers: firstPayload.headers, payload: firstPayload.payload })
    await recognitionStarted

    const secondPayload = multipartImage({ file: sampleImage })
    const secondResponse = await app.inject({ method: 'POST', url: '/v1/ocr', headers: secondPayload.headers, payload: secondPayload.payload })
    expect(secondResponse.statusCode).toBe(429)
    expect(secondResponse.headers['retry-after']).toBe('1')

    allowRecognition?.(makeResult())
    expect((await firstRequest).statusCode).toBe(200)
  })
})

function makeResult(): RapidOCRResult {
  return {
    type: 'ocr',
    img: { data: new Uint8Array([1, 2, 3]), format: 'BGR', height: 1, width: 1 },
    boxes: [],
    boxPoints: [],
    lines: [],
    txts: [],
    scores: [],
    wordResults: [],
    word_results: [],
    lang_rec: 'ch',
    elapse: 0.6,
    elapseList: [0.1, 0.2, 0.3],
    elapse_list: [0.1, 0.2, 0.3],
    timings: { classification: 0.2, detection: 0.1, recognition: 0.3 },
    toJSON: () => [],
    to_json: () => [],
    toMarkdown: () => '',
    to_markdown: () => '',
    vis: async () => undefined,
    visBuffer: async () => undefined,
  }
}

function makeEmptyResult(): RapidOCRResult {
  return {
    type: 'empty',
    img: null,
    boxes: null,
    boxPoints: null,
    lines: [],
    txts: null,
    scores: null,
    wordResults: [],
    word_results: [],
    elapse: 0.25,
    elapseList: [0.25, 0, 0],
    elapse_list: [0.25, 0, 0],
    timings: { classification: 0, detection: 0.25, recognition: 0 },
    toJSON: () => null,
    to_json: () => null,
    toMarkdown: () => '',
    to_markdown: () => '',
    vis: async () => undefined,
    visBuffer: async () => undefined,
  }
}

function multipartImage(fields: { file?: Buffer, options?: string, optionsContentType?: string }): { headers: Record<string, string>, payload: Buffer } {
  const boundary = 'rapidocr-test-boundary'
  const parts: Buffer[] = []
  if (fields.options !== undefined) {
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="options"\r\nContent-Type: ${fields.optionsContentType ?? 'application/json'}\r\n\r\n${fields.options}\r\n`))
  }
  if (fields.file !== undefined) {
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n`))
    parts.push(fields.file)
    parts.push(Buffer.from('\r\n'))
  }
  parts.push(Buffer.from(`--${boundary}--\r\n`))
  return {
    headers: { 'content-type': `multipart/form-data; boundary=${boundary}` },
    payload: Buffer.concat(parts),
  }
}
