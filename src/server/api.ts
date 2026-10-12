import type { FastifyInstance } from 'fastify'
import type { RapidOCRCallOptions, RapidOCROptions, RapidOCRResult } from '../shared/types.js'
import { Buffer } from 'node:buffer'
import multipart from '@fastify/multipart'
import Fastify from 'fastify'
import { LoadImageError } from '../image/image.js'
import { RapidOCR } from '../pipeline/rapidocr.js'

export interface RapidOCRServerOptions {
  maxConcurrency?: number
  maxFileSize?: number
  logger?: boolean
  ocr?: Pick<RapidOCR, 'recognize'>
  rapidOptions?: RapidOCROptions
  ready?: boolean
}

export interface RapidOCRServer extends FastifyInstance {
  readonly rapidOcrReady: boolean
  setRapidOcrReady: (ready: boolean) => void
}

interface RequestOptions extends RapidOCRCallOptions {
  includeImage: boolean
}

const DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024
const DEFAULT_MAX_CONCURRENCY = 1
const MAX_OPTIONS_SIZE = 16 * 1024

/** Creates the HTTP API. An injected OCR instance remains owned by its caller. */
export async function createRapidOCRServer(options: RapidOCRServerOptions = {}): Promise<RapidOCRServer> {
  const maxFileSize = options.maxFileSize ?? DEFAULT_MAX_FILE_SIZE
  const maxConcurrency = options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY
  if (!Number.isSafeInteger(maxFileSize) || maxFileSize < 1)
    throw new RangeError('maxFileSize must be a positive integer.')
  if (!Number.isSafeInteger(maxConcurrency) || maxConcurrency < 1)
    throw new RangeError('maxConcurrency must be a positive integer.')

  const ownsOcr = options.ocr === undefined
  const ocr = options.ocr ?? await RapidOCR.create({
    ...options.rapidOptions,
    useDetection: true,
    useClassifier: true,
    useRecognition: true,
  })
  if (ownsOcr) {
    try {
      await (ocr as RapidOCR).warmup()
    }
    catch (error) {
      await (ocr as RapidOCR).close().catch(() => {})
      throw error
    }
  }
  const app = Fastify({
    bodyLimit: maxFileSize + MAX_OPTIONS_SIZE + 64 * 1024,
    logger: options.logger ?? true,
    maxRequestsPerSocket: 100,
  })
  let activeRequests = 0
  let ready = options.ready ?? true
  Object.defineProperties(app, {
    rapidOcrReady: { enumerable: true, get: () => ready },
    setRapidOcrReady: { enumerable: false, value: (value: boolean) => { ready = value } },
  })
  const rapidOcrApp = app as unknown as RapidOCRServer
  if (ownsOcr)
    app.addHook('onClose', async () => (ocr as RapidOCR).close())

  await app.register(multipart, {
    limits: {
      fileSize: maxFileSize,
      files: 1,
      fields: 1,
      parts: 2,
      fieldSize: MAX_OPTIONS_SIZE,
      headerPairs: 20,
    },
  })

  app.get('/health/live', async () => ({ status: 'live' }))
  app.get('/health/ready', async (_request, reply) => {
    if (!rapidOcrApp.rapidOcrReady)
      return reply.code(503).send({ status: 'not_ready' })
    return { status: 'ready' }
  })
  app.get('/openapi.json', async () => openApiDocument(maxFileSize))
  app.setNotFoundHandler((request, reply) => {
    reply
      .code(404)
      .type('application/problem+json')
      .send({
        type: 'about:blank',
        title: 'Not Found',
        status: 404,
        detail: 'The requested API route does not exist.',
        code: 'not_found',
        request_id: request.id,
      })
  })

  app.post('/v1/ocr', async (request) => {
    if (!rapidOcrApp.rapidOcrReady)
      throw new ApiError(503, 'service_not_ready', 'OCR service is not ready.')
    if (activeRequests >= maxConcurrency)
      throw new ApiError(429, 'too_many_requests', 'OCR capacity is currently full.', { 'retry-after': '1' })
    activeRequests++
    try {
      if (!request.isMultipart())
        throw new ApiError(415, 'unsupported_media_type', 'Send a multipart/form-data request with an image file in the "file" field.')

      let image: Buffer | undefined
      let requestOptions: RequestOptions = { includeImage: true }
      for await (const part of request.parts()) {
        if (part.type === 'file') {
          if (part.fieldname !== 'file') {
            part.file.resume()
            throw new ApiError(400, 'invalid_file_field', 'The image must use the multipart field name "file".')
          }
          if (!part.mimetype.toLowerCase().startsWith('image/')) {
            part.file.resume()
            throw new ApiError(415, 'unsupported_media_type', 'The uploaded file must have an image/* content type.')
          }
          if (image !== undefined) {
            part.file.resume()
            throw new ApiError(400, 'too_many_files', 'Only one image file is accepted.')
          }
          image = await part.toBuffer()
          if (part.file.truncated)
            throw new ApiError(413, 'file_too_large', `Image exceeds the ${maxFileSize} byte upload limit.`)
        }
        else {
          if (part.fieldname !== 'options')
            throw new ApiError(400, 'invalid_form_field', 'Only the optional "options" form field is accepted.')
          if (typeof part.value !== 'string')
            throw new ApiError(400, 'invalid_options', 'The "options" field must contain a JSON object.')
          requestOptions = parseRequestOptions(part.value)
        }
      }

      if (image === undefined)
        throw new ApiError(400, 'missing_file', 'A multipart image file is required in the "file" field.')

      let result: RapidOCRResult
      try {
        result = await ocr.recognize(image, {
          useDetection: true,
          useClassifier: true,
          useRecognition: true,
          ...(requestOptions.boxThreshold === undefined ? {} : { boxThreshold: requestOptions.boxThreshold }),
          ...(requestOptions.unclipRatio === undefined ? {} : { unclipRatio: requestOptions.unclipRatio }),
          ...(requestOptions.textScoreThreshold === undefined ? {} : { textScoreThreshold: requestOptions.textScoreThreshold }),
          ...(requestOptions.returnWordBoxes === undefined ? {} : { returnWordBoxes: requestOptions.returnWordBoxes }),
          ...(requestOptions.returnSingleCharBoxes === undefined ? {} : { returnSingleCharBoxes: requestOptions.returnSingleCharBoxes }),
        })
      }
      catch (error) {
        if (error instanceof LoadImageError)
          throw new ApiError(422, 'invalid_image', 'The uploaded data could not be decoded as an image.')
        throw error
      }
      return serializeResult(result, requestOptions.includeImage, request.id)
    }
    finally {
      activeRequests--
    }
  })

  app.setErrorHandler((error, request, reply) => {
    const apiError = error instanceof ApiError ? error : multipartError(error instanceof Error ? error : new Error(String(error)))
    const status = apiError?.status ?? 500
    if (status >= 500)
      request.log.error({ err: error }, 'OCR request failed')
    reply
      .code(status)
      .type('application/problem+json')
      .headers(apiError?.headers ?? {})
      .send({
        type: 'about:blank',
        title: apiError?.title ?? 'Internal Server Error',
        status,
        detail: apiError?.message ?? 'The OCR request failed.',
        code: apiError?.code ?? 'internal_error',
        instance: request.url,
        request_id: request.id,
      })
  })

  return rapidOcrApp
}

class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly headers?: Record<string, string>,
  ) {
    super(message)
    this.name = 'ApiError'
  }

  get title(): string {
    return ({
      400: 'Bad Request',
      404: 'Not Found',
      413: 'Payload Too Large',
      415: 'Unsupported Media Type',
      422: 'Unprocessable Content',
      429: 'Too Many Requests',
      503: 'Service Unavailable',
    } as Record<number, string>)[this.status] ?? 'Internal Server Error'
  }
}

function multipartError(error: Error): ApiError | undefined {
  const candidate = error as Error & { code?: string, statusCode?: number }
  if (candidate.code === 'FST_REQ_FILE_TOO_LARGE')
    return new ApiError(413, 'file_too_large', 'The uploaded file exceeds the configured size limit.')
  if (candidate.code?.startsWith('FST_PARTS_LIMIT') || candidate.code?.startsWith('FST_FIELDS_LIMIT') || candidate.code?.startsWith('FST_FILES_LIMIT'))
    return new ApiError(413, 'multipart_limit_exceeded', 'The multipart request contains too many fields, files, or parts.')
  if (candidate.statusCode === 413)
    return new ApiError(413, 'request_too_large', 'The request exceeds the configured body size limit.')
  if (candidate.statusCode === 415)
    return new ApiError(415, 'unsupported_media_type', 'Send a multipart/form-data request with an image file in the "file" field.')
  if (candidate.statusCode === 400)
    return new ApiError(400, 'malformed_request', 'The request could not be parsed as valid multipart data.')
  if (candidate.statusCode === 404)
    return new ApiError(404, 'not_found', 'The requested API route does not exist.')
  return undefined
}

function parseRequestOptions(value: string): RequestOptions {
  if (Buffer.byteLength(value, 'utf8') > MAX_OPTIONS_SIZE)
    throw new ApiError(413, 'options_too_large', 'The options field exceeds the configured size limit.')
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  }
  catch {
    throw new ApiError(400, 'invalid_options', 'The "options" field must contain valid JSON.')
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed))
    throw new ApiError(400, 'invalid_options', 'The "options" field must contain a JSON object.')
  const values = parsed as Record<string, unknown>
  const allowed = new Set(['box_threshold', 'unclip_ratio', 'text_score_threshold', 'return_word_boxes', 'return_single_char_boxes', 'include_image'])
  for (const key of Object.keys(values)) {
    if (!allowed.has(key))
      throw new ApiError(400, 'unknown_option', `Unsupported OCR option: ${key}.`)
  }
  const numberOption = (key: string, minimum: number, maximum?: number): number | undefined => {
    const value = values[key]
    if (value === undefined)
      return undefined
    if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || (maximum !== undefined && value > maximum))
      throw new ApiError(400, 'invalid_option', `${key} must be a finite number${maximum === undefined ? ` greater than or equal to ${minimum}` : ` between ${minimum} and ${maximum}`}.`)
    return value
  }
  const booleanOption = (key: string, fallback?: boolean): boolean | undefined => {
    const value = values[key]
    if (value === undefined)
      return fallback
    if (typeof value !== 'boolean')
      throw new ApiError(400, 'invalid_option', `${key} must be a boolean.`)
    return value
  }
  const boxThreshold = numberOption('box_threshold', 0, 1)
  const unclipRatio = numberOption('unclip_ratio', 0)
  const textScoreThreshold = numberOption('text_score_threshold', 0, 1)
  const returnWordBoxes = booleanOption('return_word_boxes')
  const returnSingleCharBoxes = booleanOption('return_single_char_boxes')
  return {
    includeImage: booleanOption('include_image', true)!,
    ...(boxThreshold === undefined ? {} : { boxThreshold }),
    ...(unclipRatio === undefined ? {} : { unclipRatio }),
    ...(textScoreThreshold === undefined ? {} : { textScoreThreshold }),
    ...(returnWordBoxes === undefined ? {} : { returnWordBoxes }),
    ...(returnSingleCharBoxes === undefined ? {} : { returnSingleCharBoxes }),
  }
}

function serializeResult(result: RapidOCRResult, includeImage: boolean, requestId: string): Record<string, unknown> {
  if (result.type === 'empty') {
    return {
      request_id: requestId,
      type: 'empty',
      img: null,
      boxes: null,
      txts: null,
      scores: null,
      word_results: [],
      elapse_list: result.elapse_list,
      elapse: result.elapse,
    }
  }
  if (result.type !== 'ocr')
    throw new Error(`The HTTP OCR pipeline returned an unexpected result type: ${result.type}.`)
  const { img } = result
  return {
    request_id: requestId,
    type: 'ocr',
    img: includeImage
      ? {
          data: Buffer.from(img.data).toString('base64'),
          encoding: 'base64',
          format: img.format,
          height: img.height,
          width: img.width,
        }
      : null,
    boxes: result.boxes,
    txts: result.txts,
    scores: result.scores,
    word_results: result.word_results,
    lang_rec: result.lang_rec,
    elapse_list: result.elapse_list,
    elapse: result.elapse,
  }
}

function openApiDocument(maxFileSize: number): Record<string, unknown> {
  return {
    openapi: '3.1.0',
    info: { title: 'RapidOCR Node API', version: '1.0.0' },
    paths: {
      '/health/live': { get: { operationId: 'live', responses: { 200: { description: 'Process is alive.' } } } },
      '/health/ready': { get: { operationId: 'ready', responses: { 200: { description: 'Models are ready.' }, 503: { description: 'Models are not ready.' } } } },
      '/v1/ocr': {
        post: {
          operationId: 'recognizeImage',
          summary: 'Run full text detection, classification, and recognition on one image.',
          requestBody: {
            required: true,
            content: {
              'multipart/form-data': {
                schema: {
                  type: 'object',
                  required: ['file'],
                  properties: {
                    file: { type: 'string', format: 'binary', description: `Image file, up to ${maxFileSize} bytes.` },
                    options: {
                      type: 'string',
                      description: 'Optional JSON object: box_threshold, unclip_ratio, text_score_threshold, return_word_boxes, return_single_char_boxes, include_image.',
                    },
                  },
                },
              },
            },
          },
          responses: {
            200: { description: 'OCR result in RapidOCR field naming.' },
            400: { description: 'Invalid multipart data or options.' },
            404: { description: 'The API route was not found.' },
            413: { description: 'Image or options exceed their size limit.' },
            415: { description: 'Unsupported request or image media type.' },
            422: { description: 'Image data could not be decoded.' },
            429: { description: 'Configured OCR concurrency is full.' },
            503: { description: 'OCR models are not ready.' },
          },
        },
      },
    },
  }
}
