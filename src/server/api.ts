import type { FastifyInstance } from 'fastify'
import type { OCRStructuredFields, OCRStructuredResult, OCRTextStructurerOptions } from '../pipeline/structured-output.js'
import type { RapidOCRCallOptions, RapidOCROptions, RapidOCRResult } from '../shared/types.js'
import { Buffer } from 'node:buffer'
import multipart from '@fastify/multipart'
import Fastify from 'fastify'
import sharp from 'sharp'
import packageJson from '../../package.json' with { type: 'json' }
import { LoadImageError } from '../image/image.js'
import { RapidOCR } from '../pipeline/rapidocr.js'
import { OCRTextStructurer } from '../pipeline/structured-output.js'

export interface RapidOCRServerOptions extends RapidOCROptions {
  maxConcurrency?: number
  maxFileSize?: number
  maxImagePixels?: number
  logger?: boolean
  ocr?: Pick<RapidOCR, 'recognize'>
  structurer?: Pick<OCRTextStructurer, 'extract' | 'close'>
  structuredOutput?: OCRTextStructurerOptions
  ready?: boolean
}

export interface RapidOCRServer extends FastifyInstance {
  readonly rapidOcrReady: boolean
  setRapidOcrReady: (ready: boolean) => void
}

interface RequestOptions extends RapidOCRCallOptions {
  includeImage: boolean
  structuredFields?: OCRStructuredFields
}

const DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024
const DEFAULT_MAX_CONCURRENCY = 1
const DEFAULT_MAX_IMAGE_PIXELS = 20_000_000
const MAX_OPTIONS_SIZE = 16 * 1024

/** Creates the HTTP API. An injected OCR instance remains owned by its caller. */
export async function createRapidOCRServer(options: RapidOCRServerOptions = {}): Promise<RapidOCRServer> {
  const maxFileSize = options.maxFileSize ?? DEFAULT_MAX_FILE_SIZE
  const maxConcurrency = options.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY
  const maxImagePixels = options.maxImagePixels ?? DEFAULT_MAX_IMAGE_PIXELS
  if (!Number.isSafeInteger(maxFileSize) || maxFileSize < 1)
    throw new RangeError('maxFileSize must be a positive integer.')
  if (!Number.isSafeInteger(maxConcurrency) || maxConcurrency < 1)
    throw new RangeError('maxConcurrency must be a positive integer.')
  if (!Number.isSafeInteger(maxImagePixels) || maxImagePixels < 1)
    throw new RangeError('maxImagePixels must be a positive integer.')

  const rapidOptions: RapidOCROptions = options
  const ownsOcr = options.ocr === undefined
  const ocr = options.ocr ?? await RapidOCR.create(rapidOptions)
  if (ownsOcr) {
    try {
      await (ocr as RapidOCR).warmup()
    }
    catch (error) {
      await (ocr as RapidOCR).close().catch(() => {})
      throw error
    }
  }
  const ownsStructurer = options.structurer === undefined
  const structurer = options.structurer ?? new OCRTextStructurer({
    ...(options.structuredOutput ?? {}),
    ...(options.structuredOutput?.modelDirectory === undefined
      && options.structuredOutput?.cacheDirectory === undefined
      && rapidOptions.modelDirectory !== undefined
      ? { modelDirectory: rapidOptions.modelDirectory }
      : {}),
  })
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
  if (ownsStructurer)
    app.addHook('onClose', async () => structurer.close())

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
  app.get('/openapi.json', async () => openApiDocument(maxFileSize, maxImagePixels))
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
          requestOptions = parseRequestOptions(part.value)
        }
      }

      if (image === undefined)
        throw new ApiError(400, 'missing_file', 'A multipart image file is required in the "file" field.')
      await validateImagePixelLimit(image, maxImagePixels)

      let result: RapidOCRResult
      try {
        result = await ocr.recognize(image, {
          ...(requestOptions.useDetection === undefined ? {} : { useDetection: requestOptions.useDetection }),
          ...(requestOptions.useClassifier === undefined ? {} : { useClassifier: requestOptions.useClassifier }),
          ...(requestOptions.useRecognition === undefined ? {} : { useRecognition: requestOptions.useRecognition }),
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
      if (requestOptions.structuredFields !== undefined && result.type !== 'ocr' && result.type !== 'empty')
        throw new Error(`Structured extraction requires a full OCR result; received ${result.type}.`)
      const structured = requestOptions.structuredFields === undefined
        ? undefined
        : await structurer.extract(result.lines, requestOptions.structuredFields)
      return serializeResult(result, requestOptions.includeImage, request.id, structured)
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

async function validateImagePixelLimit(image: Buffer, maxImagePixels: number): Promise<void> {
  let width: number | undefined
  let height: number | undefined
  try {
    const metadata = await sharp(image, { failOn: 'error' }).metadata()
    width = metadata.width
    height = metadata.height
  }
  catch {
    throw new ApiError(422, 'invalid_image', 'The uploaded data could not be decoded as an image.')
  }
  if (!width || !height)
    throw new ApiError(422, 'invalid_image', 'The uploaded data could not be decoded as an image.')
  if (width > maxImagePixels / height)
    throw new ApiError(413, 'image_too_large', `Image exceeds the ${maxImagePixels} pixel processing limit.`)
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
  if (candidate.code === 'FST_INVALID_JSON_FIELD_ERROR')
    return new ApiError(400, 'invalid_options', 'The "options" field must contain valid JSON.')
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

function parseRequestOptions(value: unknown): RequestOptions {
  let serializedValue: string
  try {
    serializedValue = typeof value === 'string' ? value : JSON.stringify(value) ?? ''
  }
  catch {
    throw new ApiError(400, 'invalid_options', 'The "options" field must contain a JSON object.')
  }
  if (Buffer.byteLength(serializedValue, 'utf8') > MAX_OPTIONS_SIZE)
    throw new ApiError(413, 'options_too_large', 'The options field exceeds the configured size limit.')
  let parsed: unknown = value
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value)
    }
    catch {
      throw new ApiError(400, 'invalid_options', 'The "options" field must contain valid JSON.')
    }
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed))
    throw new ApiError(400, 'invalid_options', 'The "options" field must contain a JSON object.')
  const values = parsed as Record<string, unknown>
  const allowed = new Set([
    'useDetection',
    'useClassifier',
    'useRecognition',
    'boxThreshold',
    'unclipRatio',
    'textScoreThreshold',
    'returnWordBoxes',
    'returnSingleCharBoxes',
    'includeImage',
    'structuredFields',
  ])
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
  const useDetection = booleanOption('useDetection')
  const useClassifier = booleanOption('useClassifier')
  const useRecognition = booleanOption('useRecognition')
  const boxThreshold = numberOption('boxThreshold', 0, 1)
  const unclipRatio = numberOption('unclipRatio', 0)
  const textScoreThreshold = numberOption('textScoreThreshold', 0, 1)
  const returnWordBoxes = booleanOption('returnWordBoxes')
  const returnSingleCharBoxes = booleanOption('returnSingleCharBoxes')
  const structuredFields = parseStructuredFields(values.structuredFields)
  return {
    includeImage: booleanOption('includeImage', true)!,
    ...(useDetection === undefined ? {} : { useDetection }),
    ...(useClassifier === undefined ? {} : { useClassifier }),
    ...(useRecognition === undefined ? {} : { useRecognition }),
    ...(boxThreshold === undefined ? {} : { boxThreshold }),
    ...(unclipRatio === undefined ? {} : { unclipRatio }),
    ...(textScoreThreshold === undefined ? {} : { textScoreThreshold }),
    ...(returnWordBoxes === undefined ? {} : { returnWordBoxes }),
    ...(returnSingleCharBoxes === undefined ? {} : { returnSingleCharBoxes }),
    ...(structuredFields === undefined ? {} : { structuredFields }),
  }
}

function parseStructuredFields(value: unknown): OCRStructuredFields | undefined {
  if (value === undefined)
    return undefined
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw new ApiError(400, 'invalid_option', 'structuredFields must be an object mapping field keys to descriptions.')
  const fields = value as Record<string, unknown>
  const keys = Object.keys(fields)
  if (keys.length === 0)
    throw new ApiError(400, 'invalid_option', 'structuredFields must contain at least one field.')
  for (const key of keys) {
    if (key.trim().length === 0 || typeof fields[key] !== 'string' || (fields[key] as string).trim().length === 0)
      throw new ApiError(400, 'invalid_option', 'structuredFields must map non-empty field keys to non-empty descriptions.')
  }
  return fields as OCRStructuredFields
}

function serializeResult(
  result: RapidOCRResult,
  includeImage: boolean,
  requestId: string,
  structured?: OCRStructuredResult<OCRStructuredFields>,
): Record<string, unknown> {
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
      ...(structured === undefined ? {} : { structured }),
    }
  }
  const img = serializeImage(result.img, includeImage)
  const timings = { elapse_list: result.elapse_list, elapse: result.elapse }
  switch (result.type) {
    case 'ocr':
      return {
        request_id: requestId,
        type: 'ocr',
        img,
        boxes: result.boxes,
        txts: result.txts,
        scores: result.scores,
        word_results: result.word_results,
        lang_rec: result.lang_rec,
        ...timings,
        ...(structured === undefined ? {} : { structured }),
      }
    case 'detection':
      return {
        request_id: requestId,
        type: 'detection',
        img,
        boxes: result.boxes,
        scores: result.scores,
        ...timings,
      }
    case 'classification':
      return {
        request_id: requestId,
        type: 'classification',
        img,
        cls_res: result.cls_res,
        results: result.results,
        img_list: includeImage ? result.img_list.map(image => serializeImage(image, true)) : [],
        ...timings,
      }
    case 'recognition':
      return {
        request_id: requestId,
        type: 'recognition',
        img,
        results: result.results,
        imgs: includeImage ? result.imgs.map(image => serializeImage(image, true)) : [],
        txts: result.txts,
        scores: result.scores,
        word_results: result.word_results,
        lang_rec: result.lang_rec,
        ...timings,
      }
  }
}

function serializeImage(
  image: { data: Uint8Array, format: 'BGR', height: number, width: number },
  includeImage: boolean,
): Record<string, unknown> | null {
  if (!includeImage)
    return null
  return {
    data: Buffer.from(image.data).toString('base64'),
    encoding: 'base64',
    format: image.format,
    height: image.height,
    width: image.width,
  }
}

function openApiDocument(maxFileSize: number, maxImagePixels: number): Record<string, unknown> {
  return {
    openapi: '3.1.0',
    info: { title: 'RapidOCR Node API', version: packageJson.version },
    components: {
      schemas: {
        HealthStatus: {
          type: 'object',
          required: ['status'],
          properties: { status: { type: 'string', enum: ['live', 'ready', 'not_ready'] } },
        },
        Point: {
          type: 'array',
          items: { type: 'number' },
          minItems: 2,
          maxItems: 2,
        },
        Box: {
          type: 'array',
          items: { $ref: '#/components/schemas/Point' },
          minItems: 4,
          maxItems: 4,
        },
        EncodedImage: {
          type: 'object',
          required: ['data', 'encoding', 'format', 'height', 'width'],
          properties: {
            data: { type: 'string', contentEncoding: 'base64', description: 'Base64-encoded decoded BGR pixel bytes.' },
            encoding: { type: 'string', const: 'base64' },
            format: { type: 'string', const: 'BGR' },
            height: { type: 'integer', minimum: 1 },
            width: { type: 'integer', minimum: 1 },
          },
        },
        WordResult: {
          type: 'array',
          prefixItems: [
            { type: 'string' },
            { type: 'number' },
            { oneOf: [{ $ref: '#/components/schemas/Box' }, { type: 'null' }] },
          ],
          items: false,
          minItems: 3,
          maxItems: 3,
        },
        OcrOptions: {
          type: 'object',
          additionalProperties: false,
          properties: {
            useDetection: { type: 'boolean', default: true },
            useClassifier: { type: 'boolean', default: true },
            useRecognition: { type: 'boolean', default: true },
            boxThreshold: { type: 'number', minimum: 0, maximum: 1 },
            unclipRatio: { type: 'number', minimum: 0 },
            textScoreThreshold: { type: 'number', minimum: 0, maximum: 1 },
            returnWordBoxes: { type: 'boolean', default: false },
            returnSingleCharBoxes: { type: 'boolean', default: false },
            includeImage: { type: 'boolean', default: true },
            structuredFields: {
              type: 'object',
              minProperties: 1,
              additionalProperties: { type: 'string', minLength: 1 },
              description: 'Optional field names mapped to descriptions, for example {"invoiceNumber":"发票号码"}.',
            },
          },
        },
        OcrResponse: {
          oneOf: [
            { $ref: '#/components/schemas/OcrResult' },
            { $ref: '#/components/schemas/EmptyResult' },
            { $ref: '#/components/schemas/StageResult' },
          ],
        },
        StageResult: {
          type: 'object',
          required: ['request_id', 'type', 'img', 'elapse_list', 'elapse'],
          properties: {
            request_id: { type: 'string' },
            type: { type: 'string', enum: ['detection', 'classification', 'recognition'] },
            img: { oneOf: [{ $ref: '#/components/schemas/EncodedImage' }, { type: 'null' }] },
            boxes: { type: 'array', items: { $ref: '#/components/schemas/Box' } },
            scores: { type: 'array', items: { type: 'number' } },
            cls_res: {
              type: 'array',
              items: {
                type: 'array',
                prefixItems: [{ type: 'string' }, { type: 'number' }],
                items: false,
                minItems: 2,
                maxItems: 2,
              },
            },
            results: { type: 'array', items: { type: 'object', additionalProperties: true } },
            img_list: { type: 'array', items: { $ref: '#/components/schemas/EncodedImage' } },
            imgs: { type: 'array', items: { $ref: '#/components/schemas/EncodedImage' } },
            txts: { type: 'array', items: { type: 'string' } },
            word_results: { type: 'array', items: {} },
            lang_rec: { type: 'string' },
            elapse_list: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
            elapse: { type: 'number' },
          },
          additionalProperties: true,
        },
        OcrResult: {
          type: 'object',
          required: ['request_id', 'type', 'img', 'boxes', 'txts', 'scores', 'word_results', 'lang_rec', 'elapse_list', 'elapse'],
          properties: {
            request_id: { type: 'string' },
            type: { type: 'string', const: 'ocr' },
            img: { oneOf: [{ $ref: '#/components/schemas/EncodedImage' }, { type: 'null' }] },
            boxes: { type: 'array', items: { $ref: '#/components/schemas/Box' } },
            txts: { type: 'array', items: { type: 'string' } },
            scores: { type: 'array', items: { type: 'number' } },
            word_results: {
              type: 'array',
              items: { oneOf: [{ type: 'array', items: { $ref: '#/components/schemas/WordResult' } }, { type: 'null' }] },
            },
            lang_rec: { type: 'string' },
            elapse_list: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
            elapse: { type: 'number' },
            structured: { $ref: '#/components/schemas/StructuredResult' },
          },
        },
        EmptyResult: {
          type: 'object',
          required: ['request_id', 'type', 'img', 'boxes', 'txts', 'scores', 'word_results', 'elapse_list', 'elapse'],
          properties: {
            request_id: { type: 'string' },
            type: { type: 'string', const: 'empty' },
            img: { type: 'null' },
            boxes: { type: 'null' },
            txts: { type: 'null' },
            scores: { type: 'null' },
            word_results: { type: 'array', maxItems: 0 },
            elapse_list: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
            elapse: { type: 'number' },
            structured: { $ref: '#/components/schemas/StructuredResult' },
          },
        },
        StructuredResult: {
          type: 'object',
          required: ['values', 'evidence', 'timings'],
          properties: {
            values: {
              type: 'object',
              additionalProperties: { oneOf: [{ type: 'string' }, { type: 'null' }] },
            },
            evidence: {
              type: 'object',
              additionalProperties: { type: 'array', items: { type: 'integer', minimum: 0 } },
              description: 'OCR line indexes supporting each extracted value.',
            },
            timings: { $ref: '#/components/schemas/StructuredTimings' },
          },
        },
        StructuredTimings: {
          type: 'object',
          required: ['queue', 'modelLoad', 'generation', 'total'],
          properties: {
            queue: { type: 'number', minimum: 0, description: 'Seconds spent waiting behind earlier extraction calls.' },
            modelLoad: { type: 'number', minimum: 0, description: 'Seconds spent lazily importing Transformers.js and loading the model; zero after initialization.' },
            generation: { type: 'number', minimum: 0, description: 'Seconds spent tokenizing the prompt and generating the response.' },
            total: { type: 'number', minimum: 0, description: 'Seconds from extract() start to parsed result, including queue wait.' },
          },
        },
        ProblemDetail: {
          type: 'object',
          required: ['type', 'title', 'status', 'detail', 'code', 'request_id'],
          properties: {
            type: { type: 'string' },
            title: { type: 'string' },
            status: { type: 'integer' },
            detail: { type: 'string' },
            code: { type: 'string' },
            instance: { type: 'string' },
            request_id: { type: 'string' },
          },
        },
      },
    },
    paths: {
      '/health/live': {
        get: {
          operationId: 'live',
          responses: {
            200: {
              description: 'Process is alive.',
              content: { 'application/json': { schema: { $ref: '#/components/schemas/HealthStatus' } } },
            },
          },
        },
      },
      '/health/ready': {
        get: {
          operationId: 'ready',
          responses: {
            200: {
              description: 'Models are ready.',
              content: { 'application/json': { schema: { $ref: '#/components/schemas/HealthStatus' } } },
            },
            503: {
              description: 'Models are not ready.',
              content: { 'application/json': { schema: { $ref: '#/components/schemas/HealthStatus' } } },
            },
          },
        },
      },
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
                    file: {
                      type: 'string',
                      format: 'binary',
                      description: `Image file, up to ${maxFileSize} bytes and ${maxImagePixels} decoded pixels.`,
                    },
                    options: { $ref: '#/components/schemas/OcrOptions' },
                  },
                  additionalProperties: false,
                },
                encoding: {
                  file: { contentType: 'image/*' },
                  options: { contentType: 'application/json' },
                },
              },
            },
          },
          responses: {
            200: {
              description: 'OCR result in RapidOCR field naming.',
              content: { 'application/json': { schema: { $ref: '#/components/schemas/OcrResponse' } } },
            },
            400: problemResponse('Invalid multipart data or options.'),
            404: problemResponse('The API route was not found.'),
            413: problemResponse('Image bytes, decoded pixels, or options exceed their configured size limits.'),
            415: problemResponse('Unsupported request or image media type.'),
            422: problemResponse('Image data could not be decoded.'),
            429: {
              ...problemResponse('Configured OCR concurrency is full.'),
              headers: { 'Retry-After': { schema: { type: 'string' }, description: 'Minimum delay before retrying the request.' } },
            },
            500: problemResponse('An unexpected OCR server error occurred.'),
            503: problemResponse('OCR models are not ready.'),
          },
        },
      },
    },
  }
}

function problemResponse(description: string): Record<string, unknown> {
  return {
    description,
    content: { 'application/problem+json': { schema: { $ref: '#/components/schemas/ProblemDetail' } } },
  }
}
