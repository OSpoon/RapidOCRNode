import type { ModelSelection, ModelStage, OCRVersion } from '../../src/models/registry.js'
import type { ExecutionProviderConfig } from '../../src/shared/types.js'
import type { WordBoxLineReference } from './word-box-reference.js'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { env } from 'node:process'
import { listSupportedBackends } from 'onnxruntime-node'
import sharp from 'sharp'
import { beforeAll, describe, expect, it } from 'vitest'
import { parse } from 'yaml'
import { TEST_FIXTURE_ASSETS } from '../../model-assets.mjs'
import { downloadVerifiedAsset } from '../../src/models/download.js'
import { resolveModelRoute } from '../../src/models/registry.js'
import { expectWordBoxesMatchReference } from './word-box-reference.js'

interface UpstreamRoute {
  model_key: string
  supported_langs: string[]
}

interface ModelRouteCase {
  modelKey: string
  modelType: string
  ocrVersion: OCRVersion
  selection: ModelSelection
  stage: ModelStage
}

interface StageReference {
  additionalSamples?: Array<{ boxes?: number[][][] | null, classification?: Array<[string, number]>, fixture: string, scores?: number[] | null }>
  boxes?: number[][][] | null
  classification?: Array<[string, number]>
  fixture: string
  scores?: number[] | null
  texts?: string[]
}

interface FullPageLineReference {
  box: number[][]
  confidence: number
  text: string
}

interface FullPageReference {
  fixture: string
  lines: FullPageLineReference[]
  modelKey: string
  version: OCRVersion
}

interface ProviderCase {
  enabled: boolean
  executionProviders: readonly ExecutionProviderConfig[]
  name: string
  scoreTolerance: number
}

const upstream = parse(await readFile(new URL('../fixtures/rapidocr-v3.10.0-default_models.yaml', import.meta.url), 'utf8')) as {
  model_routes: Record<string, Record<string, unknown>>
}
const modelRouteCases = collectModelRouteCases(upstream.model_routes)
const modelsDirectory = resolve(env.RAPIDOCR_MODEL_DIR || 'models')
const fixturePaths = new Map(TEST_FIXTURE_ASSETS.map(asset => [asset.file, resolve(modelsDirectory, asset.file)]))
const supportedProviders = new Set(listSupportedBackends().map(backend => backend.name.toLowerCase()))
const modelProviders: ProviderCase[] = [
  { enabled: true, executionProviders: ['cpu'], name: 'ONNX Runtime CPU', scoreTolerance: 0.03 },
  ...(env.RAPIDOCR_TEST_WEBGPU === '1' ? [{ enabled: supportedProviders.has('webgpu'), executionProviders: ['webgpu', 'cpu'] as const, name: 'ONNX Runtime WebGPU', scoreTolerance: 0.03 }] : []),
  ...(env.RAPIDOCR_TEST_COREML === '1' ? [{ enabled: supportedProviders.has('coreml'), executionProviders: [{ name: 'coreml', coreMlFlags: 0x018 }, 'cpu'] as const, name: 'ONNX Runtime CoreML', scoreTolerance: 0.04 }] : []),
  ...(env.RAPIDOCR_TEST_CUDA === '1' ? [{ enabled: supportedProviders.has('cuda'), executionProviders: [{ name: 'cuda', deviceId: readIntegerEnv('RAPIDOCR_CUDA_DEVICE_ID', 0) }, 'cpu'] as const, name: 'ONNX Runtime CUDA', scoreTolerance: 0.04 }] : []),
  ...(env.RAPIDOCR_TEST_DIRECTML === '1' ? [{ enabled: supportedProviders.has('dml'), executionProviders: [{ name: 'dml', deviceId: readIntegerEnv('RAPIDOCR_DIRECTML_DEVICE_ID', 0) }, 'cpu'] as const, name: 'ONNX Runtime DirectML', scoreTolerance: 0.04 }] : []),
]
const fullPageProviders: ProviderCase[] = [
  modelProviders[0]!,
  ...modelProviders.filter(provider => provider.name.includes('WebGPU') || provider.name.includes('CoreML')),
]
const modelReferences = JSON.parse(await readFile(new URL('../fixtures/rapidocr-v3.10.0-model-reference.json', import.meta.url), 'utf8')) as {
  models: Record<string, StageReference>
  upstream: string
}
const recognitionFullPageReferences = JSON.parse(await readFile(new URL('../fixtures/rapidocr-v3.10.0-recognition-fullpages.expected.json', import.meta.url), 'utf8')) as {
  models: Record<string, FullPageReference>
  upstream: string
}
const onnxWordBoxReferences = JSON.parse(await readFile(new URL('../fixtures/onnx-word-boxes.expected.json', import.meta.url), 'utf8')) as {
  engine: string
  models: Record<string, { fixture: string, lines: WordBoxLineReference[] }>
  upstream: string
}
const onnxSingleCharacterWordBoxReferences = JSON.parse(await readFile(new URL('../fixtures/onnx-single-char-word-boxes.expected.json', import.meta.url), 'utf8')) as {
  models: Record<string, { fixture: string, lines: WordBoxLineReference[] }>
}
const providerModelCases = modelProviders.flatMap(provider => provider.enabled
  ? modelRouteCases.map(route => ({ provider, route }))
  : [])
let api: typeof import('../../src/index.js')

beforeAll(async () => {
  const packageName: string = 'rapidocr-node'
  api = await import(packageName)
  await Promise.all(TEST_FIXTURE_ASSETS.map(asset => downloadVerifiedAsset(asset, modelsDirectory)))
  await Promise.all([
    ['ch_en_num.jpg', 'ch_en_num-sharp.png'],
    ['en_rec.jpg', 'en_rec-sharp.png'],
    ['japan.jpg', 'japan-sharp.png'],
  ].map(([source, target]) => sharp(resolve(modelsDirectory, source!)).png().toFile(resolve(modelsDirectory, target!))))
  await sharp(resolve(modelsDirectory, 'text_cls.jpg')).rotate(180).png().toFile(resolve(modelsDirectory, 'text_cls-rotated.png'))
  await sharp(resolve(modelsDirectory, 'ch_doc_server.png')).extract({ left: 18, top: 0, width: 140, height: 74 }).png().toFile(resolve(modelsDirectory, 'ch_doc_server-line.png'))
}, 180_000)

describe('oNNX Runtime model routes', () => {
  it('matches all 42 distinct ONNX routes declared by upstream', () => {
    expect(modelRouteCases).toHaveLength(42)
    expect(modelRouteCases.filter(route => route.stage === 'det')).toHaveLength(9)
    expect(modelRouteCases.filter(route => route.stage === 'cls')).toHaveLength(3)
    expect(modelRouteCases.filter(route => route.stage === 'rec')).toHaveLength(30)
    expect(modelReferences.upstream).toBe('RapidOCR v3.10.0')
  })

  it('requires each requested execution provider to exist in this ONNX Runtime build', () => {
    for (const provider of modelProviders.filter(provider => provider.name !== 'ONNX Runtime CPU'))
      expect(provider.enabled, `${provider.name} was requested but is not available`).toBe(true)
  })

  it.each(providerModelCases)('$provider.name $route.ocrVersion $route.stage $route.modelKey', async ({ provider, route }) => {
    const referenceKey = `${route.stage}/${route.ocrVersion}/${route.modelKey}`
    const reference = modelReferences.models[referenceKey]
    if (!reference)
      throw new Error(`Missing upstream reference for ${referenceKey}.`)
    const modelAsset = resolveModelRoute(route.stage, route.selection)
    const modelPath = await downloadVerifiedAsset(modelAsset.asset, modelsDirectory)
    const sampleInputs = [
      { boxes: reference.boxes, classification: reference.classification, fixture: reference.fixture, scores: reference.scores },
      ...(reference.additionalSamples ?? []),
    ]

    if (route.stage === 'det') {
      const detector = await api.TextDetector.create(modelPath, 1, {}, provider.executionProviders)
      try {
        for (const sample of sampleInputs) {
          const result = await detector.detect(resolveFixture(sample.fixture))
          expect(result.img.width).toBeGreaterThan(0)
          expectBoxesMatchReference(result.boxes, sample.boxes, `${referenceKey} ${sample.fixture}`)
          expectScoresMatchReference(result.scores, sample.scores, provider.scoreTolerance)
        }
      }
      finally {
        await detector.close()
      }
      return
    }

    if (route.stage === 'cls') {
      const shape = route.ocrVersion === 'PP-OCRv5' ? [3, 80, 160] as const : [3, 48, 192] as const
      const classifier = await api.TextClassifier.create(modelPath, 1, provider.executionProviders, undefined, undefined, undefined, shape)
      try {
        for (const sample of sampleInputs) {
          if (!sample.classification)
            throw new Error(`Missing classifier reference for ${referenceKey} on ${sample.fixture}.`)
          const results = await classifier.classifyLines([resolveFixture(sample.fixture)])
          expect(results).toHaveLength(sample.classification.length)
          for (const [index, expected] of sample.classification.entries()) {
            expect(results[index]?.angle).toBe(Number(expected[0]))
            expect(Math.abs(results[index]!.confidence - expected[1])).toBeLessThanOrEqual(provider.scoreTolerance)
          }
        }
      }
      finally {
        await classifier.close()
      }
      return
    }

    if (!modelAsset.dictionary)
      throw new Error(`Missing upstream dictionary for ${route.modelKey}.`)
    const dictionaryPath = await downloadVerifiedAsset(modelAsset.dictionary, modelsDirectory)
    const recognizer = await api.TextRecognizer.create({
      dictionaryPath,
      executionProviders: provider.executionProviders,
      intraOpNumThreads: 1,
      modelPath,
      modelSelection: route.selection,
    })
    try {
      const result = await recognizer.recognizeLine(resolveFixture(reference.fixture))
      expect(result.text.trim().length).toBeGreaterThan(0)
      expect(result.text).toBe(reference.texts?.[0])
      expect(reference.scores).toHaveLength(1)
      expect(Math.abs(result.confidence - reference.scores![0]!)).toBeLessThanOrEqual(provider.scoreTolerance)
    }
    finally {
      await recognizer.close()
    }
  }, 180_000)
})

describe('oNNX Runtime full-page recognition', () => {
  const routes = modelRouteCases.filter(route => route.stage === 'rec')

  it('pins a complete upstream reference for all 30 recognition routes', () => {
    expect(recognitionFullPageReferences.upstream).toBe('RapidOCR v3.10.0')
    expect(Object.keys(recognitionFullPageReferences.models).sort()).toEqual(
      routes.map(route => `rec/${route.ocrVersion}/${route.modelKey}`).sort(),
    )
  })

  for (const provider of fullPageProviders) {
    for (const route of routes) {
      const referenceKey = `rec/${route.ocrVersion}/${route.modelKey}`
      it.skipIf(!provider.enabled)(`${provider.name} ${route.ocrVersion}/${route.modelKey}`, async () => {
        const reference = recognitionFullPageReferences.models[referenceKey]
        if (!reference)
          throw new Error(`Missing full-page reference for ${referenceKey}.`)
        const ocr = await api.RapidOCR.create({
          classifierModel: { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
          detectionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
          executionProviders: provider.executionProviders,
          intraOpNumThreads: 1,
          modelDirectory: modelsDirectory,
          recognitionModel: route.selection,
        })
        try {
          const result = await ocr.recognize(resolveFixture(reference.fixture))
          expect(result.type).toBe('ocr')
          if (result.type !== 'ocr')
            throw new TypeError(`Expected full OCR output for ${referenceKey}.`)
          expect(result.lines).toHaveLength(reference.lines.length)
          for (const [index, actual] of result.lines.entries()) {
            const expected = reference.lines[index]!
            expect(actual.text, `line ${index}`).toBe(expected.text)
            expect(Math.abs(actual.confidence - expected.confidence)).toBeLessThanOrEqual(provider.scoreTolerance)
            expect(actual.box).toHaveLength(expected.box.length)
            for (const [pointIndex, point] of actual.box.entries()) {
              expect(Math.abs(point.x - expected.box[pointIndex]![0]!)).toBeLessThanOrEqual(2)
              expect(Math.abs(point.y - expected.box[pointIndex]![1]!)).toBeLessThanOrEqual(2)
            }
          }
        }
        finally {
          await ocr.close()
        }
      }, 180_000)
    }
  }
})

describe('oNNX Runtime word boxes', () => {
  const routeKeys = ['rec/PP-OCRv4/ch_PP-OCRv4_rec_mobile', 'rec/PP-OCRv4/en_PP-OCRv4_rec_mobile']

  it('matches upstream word and single-character box references', async () => {
    expect(onnxWordBoxReferences.upstream).toBe('RapidOCR v3.10.0')
    expect(onnxWordBoxReferences.engine).toBe('ONNX Runtime CPU')
    for (const routeKey of routeKeys) {
      const reference = onnxWordBoxReferences.models[routeKey]
      const characterReference = onnxSingleCharacterWordBoxReferences.models[routeKey]
      const route = modelRouteCases.find(candidate => `${candidate.stage}/${candidate.ocrVersion}/${candidate.modelKey}` === routeKey)
      if (!reference || !characterReference || !route)
        throw new Error(`Missing word-box reference or model route for ${routeKey}.`)
      const ocr = await api.RapidOCR.create({
        classifierModel: { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
        detectionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
        intraOpNumThreads: 1,
        modelDirectory: modelsDirectory,
        recognitionModel: route.selection,
      })
      try {
        const input = resolveFixture(reference.fixture)
        const words = await ocr.recognize(input, { returnWordBoxes: true })
        expect(words.type).toBe('ocr')
        if (words.type !== 'ocr')
          throw new TypeError(`Expected full OCR output for ${routeKey}.`)
        expectWordBoxesMatchReference(words.lines, reference.lines)
        const characters = await ocr.recognize(input, { returnSingleCharBoxes: true, returnWordBoxes: true })
        expect(characters.type).toBe('ocr')
        if (characters.type !== 'ocr')
          throw new TypeError(`Expected full OCR output for ${routeKey}.`)
        expectWordBoxesMatchReference(characters.lines, characterReference.lines)
      }
      finally {
        await ocr.close()
      }
    }
  }, 180_000)
})

function collectModelRouteCases(routes: Record<string, Record<string, unknown>>): ModelRouteCase[] {
  const unique = new Map<string, ModelRouteCase>()
  for (const [ocrVersion, config] of Object.entries(routes) as Array<[OCRVersion, Record<string, unknown>]>) {
    for (const stage of ['det', 'cls', 'rec'] as const) {
      const task = config[stage]
      if (typeof task !== 'object' || task === null)
        continue
      for (const [modelType, rawRoutes] of Object.entries(task)) {
        if (!Array.isArray(rawRoutes))
          continue
        for (const rawRoute of rawRoutes as UpstreamRoute[]) {
          const langType = rawRoute.supported_langs[0]
          if (!langType)
            throw new Error(`Upstream route ${rawRoute.model_key} has no supported language.`)
          const selection = { langType, modelType, ocrVersion }
          const resolved = resolveModelRoute(stage, selection)
          if (resolved.modelKey !== rawRoute.model_key)
            throw new Error(`Node route mismatch for ${ocrVersion}/${stage}/${modelType}/${langType}.`)
          unique.set(`${stage}/${ocrVersion}/${rawRoute.model_key}`, { modelKey: rawRoute.model_key, modelType, ocrVersion, selection, stage })
        }
      }
    }
  }
  return [...unique.values()].sort((left, right) => `${left.stage}/${left.ocrVersion}/${left.modelKey}`.localeCompare(`${right.stage}/${right.ocrVersion}/${right.modelKey}`))
}

function resolveFixture(file: string): string {
  return fixturePaths.get(file) ?? resolve(modelsDirectory, file)
}

function expectBoxesMatchReference(actual: readonly (readonly { x: number, y: number }[])[], expected: number[][][] | null | undefined, context: string): void {
  if (expected === undefined)
    throw new Error(`Missing detector boxes for ${context}.`)
  const expectedBoxes = expected ?? []
  expect(actual, context).toHaveLength(expectedBoxes.length)
  for (const [boxIndex, box] of actual.entries()) {
    expect(box).toHaveLength(expectedBoxes[boxIndex]!.length)
    for (const [pointIndex, point] of box.entries()) {
      expect(Math.abs(point.x - expectedBoxes[boxIndex]![pointIndex]![0]!)).toBeLessThanOrEqual(2)
      expect(Math.abs(point.y - expectedBoxes[boxIndex]![pointIndex]![1]!)).toBeLessThanOrEqual(2)
    }
  }
}

function expectScoresMatchReference(actual: readonly number[], expected: number[] | null | undefined, tolerance: number): void {
  if (expected === undefined)
    throw new Error('Missing detector scores in the upstream reference.')
  const actualSorted = [...actual].sort((left, right) => left - right)
  const expectedSorted = [...(expected ?? [])].sort((left, right) => left - right)
  expect(actualSorted).toHaveLength(expectedSorted.length)
  for (const [index, score] of actualSorted.entries())
    expect(Math.abs(score - expectedSorted[index]!)).toBeLessThanOrEqual(tolerance)
}

function readIntegerEnv(name: string, fallback: number): number {
  const value = Number(env[name] ?? fallback)
  if (!Number.isInteger(value) || value < 0)
    throw new RangeError(`${name} must be a non-negative integer.`)
  return value
}
