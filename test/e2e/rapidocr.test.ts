import type { RapidOCRResult } from '../../src/shared/types.js'
import type { KnownRuntimeTextDifference } from '../helpers/known-runtime-differences.js'
import { Buffer } from 'node:buffer'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { arch, env } from 'node:process'
import { fileURLToPath } from 'node:url'
import { InferenceSession, listSupportedBackends } from 'onnxruntime-node'
import sharp from 'sharp'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { TEST_FIXTURE_ASSETS } from '../../model-assets.mjs'
import { downloadVerifiedAsset } from '../../src/models/download.js'
import { getKnownRecognitionFullPageTextDifferences } from '../helpers/known-runtime-differences.js'

type PackageApi = typeof import('../../src/index.js')
type RapidOCRInstance = Awaited<ReturnType<PackageApi['RapidOCR']['create']>>
type TextRecognizerInstance = Awaited<ReturnType<PackageApi['TextRecognizer']['create']>>
type TextDetectorInstance = Awaited<ReturnType<PackageApi['TextDetector']['create']>>
type TextClassifierInstance = Awaited<ReturnType<PackageApi['TextClassifier']['create']>>

interface ExpectedLine {
  box: number[][]
  confidence: number
  text: string
}

interface ExpectedOutput {
  lines: ExpectedLine[]
  upstream: string
}

interface UpstreamFullPageReference {
  image: string
  lines: ExpectedLine[]
}

interface UpstreamFullPageReferences {
  languages: Record<string, UpstreamFullPageReference>
  classifierEnabledLanguages: Record<string, UpstreamFullPageReference>
}

interface UpstreamPPv4FullPageReferences {
  languages: Record<string, UpstreamFullPageReference>
}

interface UpstreamPPv6FullPageReferences {
  languages: Record<string, UpstreamFullPageReference>
}

interface UpstreamRecognitionFullPageReference {
  fixture: string
  language: string
  lines: ExpectedLine[]
  modelKey: string
  modelType: 'medium' | 'mobile' | 'server' | 'small' | 'tiny'
  version: 'PP-OCRv4' | 'PP-OCRv5' | 'PP-OCRv6'
}

const knownChineseVehiclePageTextDifferences: Record<string, readonly KnownRuntimeTextDifference[]> = arch === 'arm64'
  ? {}
  : {
      'multi_PP-OCRv6_rec_small': [
        { confidenceTolerance: 0.04, index: 8, nodeText: '5. 抵押权人姓名/名称：', upstreamText: '5. 抵押权人姓名/名称' },
      ],
      'multi_PP-OCRv6_rec_tiny': [
        { confidenceTolerance: 0.04, index: 4, nodeText: '变更e记期：201名06-29', upstreamText: '变更e记日期：201名06-29' },
      ],
    }

interface UpstreamLanguageCase {
  langType: string
  ocrVersion?: 'PP-OCRv4' | 'PP-OCRv5'
  image: string
  text: string
  fullPage?: boolean
}

const upstreamLanguageCases: readonly UpstreamLanguageCase[] = [
  { langType: 'ar', ocrVersion: 'PP-OCRv4', image: 'arabic.png', text: 'هبوخ اه زورما' },
  { langType: 'cyrillic', ocrVersion: 'PP-OCRv4', image: 'cyrillic.png', text: 'Bpeмe je лeno данac.' },
  { langType: 'devanagari', ocrVersion: 'PP-OCRv4', image: 'devanagari_rec.png', text: 'आजमौसमअछाहै' },
  { langType: 'ta', ocrVersion: 'PP-OCRv4', image: 'ta.png', text: 'இனைனக்கவானிைலெராம்பநல்லாஇஞக்க.' },
  { langType: 'te', ocrVersion: 'PP-OCRv4', image: 'te.png', text: 'ఈరేజహాతావరణంచాలా బాగుంద.' },
  { langType: 'latin', ocrVersion: 'PP-OCRv4', image: 'latin.jpg', text: 'Alphabetmin mundo hodie requentsime adhibitm est alphabem Latinm.' },
  { langType: 'cyrillic', image: 'cyrillic.png', text: 'Време је лепо данас.' },
  { langType: 'devanagari', image: 'devanagari_rec.png', text: 'आज मौसम अच्छा है।' },
  { langType: 'ta', image: 'ta.png', text: 'இன்னைக்கு வானிலை ராம்ப நல்லா இருக்கு.' },
  { langType: 'te', image: 'te.png', text: 'ఈరోజ వాతావరణం చాలా బాగుంది.' },
  { langType: 'el', image: 'el_rec.jpg', text: 'Ωραίος καιρός σήμερα.' },
  { langType: 'th', image: 'th_rec.jpg', text: 'การแพทย์แผนจีนช่วยให้เด็กสมองพิการในคาซัคสถานกลับมาเดินได้' },
  { langType: 'en', ocrVersion: 'PP-OCRv4', image: 'en_rec.jpg', text: 'To facilitate the shot type analysis in videos, we collect MovieShots, a large-scale' },
  { langType: 'en', ocrVersion: 'PP-OCRv5', image: 'en_rec.jpg', text: 'To facilitate the shot type analysis in videos, we collect MovieShots, a large-scale' },
  { langType: 'korean', ocrVersion: 'PP-OCRv4', image: 'korean.jpg', text: '베이징차오양', fullPage: true },
  { langType: 'korean', ocrVersion: 'PP-OCRv5', image: 'korean.jpg', text: '베이징 차오양,', fullPage: true },
  { langType: 'latin', image: 'latin.jpg', text: 'Alphabetum in mundo hodie frequentissie adhibitum est alphabetum Latinum.' },
  { langType: 'eslav', image: 'eslav.jpg', text: 'Славянские языки — большая языковая семья.' },
]

const upstreamFullPageReferences = JSON.parse(await readFile(
  new URL('../fixtures/rapidocr-v3.10.0-language-pages.expected.json', import.meta.url),
  'utf8',
)) as UpstreamFullPageReferences
const upstreamFullPageCases = Object.entries(upstreamFullPageReferences.languages)
  .map(([langType, reference]) => ({ langType, ...reference }))
const upstreamClassifierFullPageCases = Object.entries(upstreamFullPageReferences.classifierEnabledLanguages)
  .map(([langType, reference]) => ({ langType, ...reference }))
const upstreamPPv4FullPageReferences = JSON.parse(await readFile(
  new URL('../fixtures/ppocrv4-language-pages.expected.json', import.meta.url),
  'utf8',
)) as UpstreamPPv4FullPageReferences
const upstreamPPv4FullPageCases = Object.entries(upstreamPPv4FullPageReferences.languages)
  .map(([langType, reference]) => ({ langType, ...reference }))
const upstreamPPv6FullPageReferences = JSON.parse(await readFile(
  new URL('../fixtures/ppocrv6-language-pages.expected.json', import.meta.url),
  'utf8',
)) as UpstreamPPv6FullPageReferences
const upstreamPPv6FullPageCases = Object.entries(upstreamPPv6FullPageReferences.languages)
  .map(([langType, reference]) => ({ langType, ...reference }))
const chineseChtFullPageReference = JSON.parse(await readFile(
  new URL('../fixtures/chinese-cht-synthetic.expected.json', import.meta.url),
  'utf8',
)) as UpstreamFullPageReference
const chineseChtSyntheticImagePath = fileURLToPath(new URL('../fixtures/chinese-cht-synthetic.png', import.meta.url))
const upstreamRecognitionFullPageReferences = JSON.parse(await readFile(
  new URL('../fixtures/rapidocr-v3.10.0-recognition-fullpages.expected.json', import.meta.url),
  'utf8',
)) as { models: Record<string, UpstreamRecognitionFullPageReference> }
const upstreamRecognitionFullPageCases = Object.values(upstreamRecognitionFullPageReferences.models)
const chineseRouteFullPageReferences = JSON.parse(await readFile(
  new URL('../fixtures/check_return_word_len-recognition-routes.expected.json', import.meta.url),
  'utf8',
)) as { models: Record<string, UpstreamRecognitionFullPageReference> }
const chineseRouteFullPageCases = Object.values(chineseRouteFullPageReferences.models)

const modelsDirectory = resolve(env.RAPIDOCR_MODEL_DIR || 'models')
const packageName: string = 'rapidocr-node'
const supportedExecutionProviders = listSupportedBackends().map(backend => backend.name.toLowerCase())
const hasCoreML = supportedExecutionProviders.includes('coreml')
const hasTensorRT = supportedExecutionProviders.includes('tensorrt')
const hasWebGPU = supportedExecutionProviders.includes('webgpu')
const hasCUDA = supportedExecutionProviders.includes('cuda')
const hasDirectML = supportedExecutionProviders.includes('dml')
const testCUDA = env.RAPIDOCR_TEST_CUDA === '1'
const testDirectML = env.RAPIDOCR_TEST_DIRECTML === '1'
const testTensorRT = env.RAPIDOCR_TEST_TENSORRT === '1'
const testWebGPU = env.RAPIDOCR_TEST_WEBGPU === '1'
const cudaDeviceId = readNonNegativeIntegerEnv('RAPIDOCR_CUDA_DEVICE_ID')
const directMLDeviceId = readNonNegativeIntegerEnv('RAPIDOCR_DIRECTML_DEVICE_ID')
const samplePath = resolve(modelsDirectory, 'ch_en_num.jpg')
const expected = JSON.parse(await readFile(new URL('../fixtures/ch_en_num.expected.json', import.meta.url), 'utf8')) as ExpectedOutput
const verticalPagePath = fileURLToPath(new URL('../fixtures/vertical-chinese.png', import.meta.url))
const verticalJapanesePagePath = fileURLToPath(new URL('../fixtures/vertical-japanese.png', import.meta.url))
const verticalPageExpected = JSON.parse(await readFile(
  new URL('../fixtures/vertical-chinese.expected.json', import.meta.url),
  'utf8',
)) as ExpectedOutput
const verticalJapaneseExpected = JSON.parse(await readFile(
  new URL('../fixtures/vertical-japanese.expected.json', import.meta.url),
  'utf8',
)) as ExpectedOutput
const blankBoundaryExpected = JSON.parse(await readFile(
  new URL('../fixtures/blank-boundary.expected.json', import.meta.url),
  'utf8',
)) as { cases: Array<{ height: number, texts: string[], type: 'empty', width: number }>, upstream: string }
const lowResolutionExpected = JSON.parse(await readFile(new URL('../fixtures/ch_en_num_162x215.expected.json', import.meta.url), 'utf8')) as { lines: ExpectedLine[] }
const documentExpected = JSON.parse(await readFile(new URL('../fixtures/ch_doc_server.expected.json', import.meta.url), 'utf8')) as ExpectedOutput
const registrationDocumentExpected = JSON.parse(await readFile(new URL('../fixtures/check_return_word_len.expected.json', import.meta.url), 'utf8')) as ExpectedOutput
let ocr: RapidOCRInstance | undefined
let englishRecognizer: TextRecognizerInstance | undefined
let arabicRecognizer: TextRecognizerInstance | undefined
let recognizer: TextRecognizerInstance | undefined
let detector: TextDetectorInstance | undefined
let classifier: TextClassifierInstance | undefined
const routedOcrInstances: RapidOCRInstance[] = []

beforeAll(async () => {
  const api = await import(packageName) as PackageApi
  const modelPaths = await api.downloadModels({ modelDirectory: modelsDirectory })
  await Promise.all(TEST_FIXTURE_ASSETS.map(asset => downloadVerifiedAsset(asset, modelsDirectory)))
  await Promise.all([
    ['ch_en_num.jpg', 'ch_en_num-sharp.png'],
    ['japan.jpg', 'japan-sharp.png'],
  ].map(([source, target]) => sharp(resolve(modelsDirectory, source!)).png().toFile(resolve(modelsDirectory, target!))))
  ocr = await api.RapidOCR.create({
    ...modelPaths,
    intraOpNumThreads: 1,
  })
  recognizer = await api.TextRecognizer.create({
    modelPath: modelPaths.recognitionModelPath,
    dictionaryPath: modelPaths.dictionaryPath,
    intraOpNumThreads: 1,
  })
  detector = await api.TextDetector.create(modelPaths.detectionModelPath, 1)
  classifier = await api.TextClassifier.create(modelPaths.classifierModelPath, 1)

  const englishSelection = { langType: 'en', modelType: 'mobile', ocrVersion: 'PP-OCRv5' } as const
  const englishModelPaths = await api.downloadModels({
    modelDirectory: modelsDirectory,
    recognitionModel: englishSelection,
  })
  englishRecognizer = await api.TextRecognizer.create({
    dictionaryPath: englishModelPaths.dictionaryPath,
    intraOpNumThreads: 1,
    modelPath: englishModelPaths.recognitionModelPath,
    modelSelection: englishSelection,
  })

  const arabicSelection = { langType: 'ar', modelType: 'mobile', ocrVersion: 'PP-OCRv5' } as const
  const arabicModelPaths = await api.downloadModels({
    modelDirectory: modelsDirectory,
    recognitionModel: arabicSelection,
  })
  arabicRecognizer = await api.TextRecognizer.create({
    dictionaryPath: arabicModelPaths.dictionaryPath,
    intraOpNumThreads: 1,
    modelPath: arabicModelPaths.recognitionModelPath,
    modelSelection: arabicSelection,
  })

  for (const ocrVersion of ['PP-OCRv4', 'PP-OCRv5'] as const) {
    const detectionModel = { langType: 'ch', modelType: 'mobile', ocrVersion } as const
    const recognitionModel = { langType: 'ch', modelType: 'mobile', ocrVersion } as const
    const classifierModel = { langType: 'ch', modelType: 'mobile', ocrVersion } as const
    const paths = await api.downloadModels({
      classifierModel,
      detectionModel,
      modelDirectory: modelsDirectory,
      recognitionModel,
    })
    routedOcrInstances.push(await api.RapidOCR.create({
      ...paths,
      classifierModel,
      detectionModel,
      intraOpNumThreads: 1,
      recognitionModel,
    }))
  }
}, 300_000)

afterAll(async () => {
  await ocr?.close()
  await recognizer?.close()
  await englishRecognizer?.close()
  await arabicRecognizer?.close()
  await detector?.close()
  await classifier?.close()
  await Promise.all(routedOcrInstances.map(instance => instance.close()))
})

describe('rapidocr with pinned upstream models', () => {
  it('matches the upstream output when all caller-created ONNX sessions are injected', async () => {
    const api = await import(packageName) as PackageApi
    const modelPaths = await api.downloadModels({ modelDirectory: modelsDirectory })
    const sessions = await Promise.all([
      InferenceSession.create(modelPaths.detectionModelPath, { executionProviders: ['cpu'], intraOpNumThreads: 1 }),
      InferenceSession.create(modelPaths.classifierModelPath, { executionProviders: ['cpu'], intraOpNumThreads: 1 }),
      InferenceSession.create(modelPaths.recognitionModelPath, { executionProviders: ['cpu'], intraOpNumThreads: 1 }),
    ])
    const [detectionSession, classifierSession, recognitionSession] = sessions
    if (!detectionSession || !classifierSession || !recognitionSession)
      throw new Error('Unable to create the pinned ONNX Runtime sessions.')
    let instance: RapidOCRInstance | undefined
    try {
      instance = await api.RapidOCR.create({
        autoDownloadModels: false,
        classifierSession,
        detectionSession,
        dictionaryPath: modelPaths.dictionaryPath,
        recognitionSession,
      })
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines)
    }
    finally {
      await instance?.close()
      await Promise.all(sessions.map(session => session.release()))
    }
  }, 120_000)

  it.each(['tiny', 'medium'] as const)('matches the upstream PP-OCRv6 %s detection and recognition route', async (modelType) => {
    const api = await import(packageName) as PackageApi
    const modelSelection = { langType: 'ch', modelType, ocrVersion: 'PP-OCRv6' } as const
    const reference = JSON.parse(await readFile(
      new URL(`../fixtures/ch_en_num.ppocrv6-${modelType}.expected.json`, import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const instance = await api.RapidOCR.create({
      detectionModel: modelSelection,
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: modelSelection,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, reference.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches upstream PP-OCRv4 and PP-OCRv5 mobile outputs on a real Chinese page image', async () => {
    for (const [index, instance] of routedOcrInstances.entries()) {
      const reference = JSON.parse(await readFile(
        new URL(`../fixtures/ppocrv${index + 4}-mobile.expected.json`, import.meta.url),
        'utf8',
      )) as ExpectedOutput
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      expect(result.lines.length).toBeGreaterThan(0)
      expectLinesMatchReference(result.lines, reference.lines)
    }
  })

  it.skipIf(!hasCoreML)('matches the pinned real-model output with CoreML text recognition', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      recognitionExecutionProviders: ['coreml', 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.skipIf(!hasCoreML)('matches the pinned full-page output with CoreML across all OCR stages', async () => {
    const api = await import(packageName) as PackageApi
    const coreML = { name: 'coreml', coreMlFlags: 0x010 } as const
    const instance = await api.RapidOCR.create({
      classifierExecutionProviders: [coreML, 'cpu'],
      detectionExecutionProviders: [coreML, 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionExecutionProviders: [coreML, 'cpu'],
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, 'ch_en_num-sharp.png'))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.skipIf(!testWebGPU || !hasWebGPU)('matches the pinned real-model output with WebGPU across all OCR stages', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      executionProviders: ['webgpu', 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('requires requested ONNX Runtime hardware providers to exist in this build', () => {
    if (testCUDA)
      expect(hasCUDA, 'RAPIDOCR_TEST_CUDA was set, but CUDA is absent from this ONNX Runtime build.').toBe(true)
    if (testDirectML)
      expect(hasDirectML, 'RAPIDOCR_TEST_DIRECTML was set, but DirectML is absent from this ONNX Runtime build.').toBe(true)
    if (testTensorRT)
      expect(hasTensorRT, 'RAPIDOCR_TEST_TENSORRT was set, but TensorRT is absent from this ONNX Runtime build.').toBe(true)
    if (testWebGPU)
      expect(hasWebGPU, 'RAPIDOCR_TEST_WEBGPU was set, but WebGPU is absent from this ONNX Runtime build.').toBe(true)
  })

  it.skipIf(!testCUDA || !hasCUDA)('matches the pinned full-page output with CUDA across all OCR stages', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      executionProviders: [{ name: 'cuda', deviceId: cudaDeviceId }, 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines, 0.04)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.skipIf(!testDirectML || !hasDirectML)('matches the pinned full-page output with DirectML across all OCR stages', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      executionProviders: [{ name: 'dml', deviceId: directMLDeviceId }, 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines, 0.04)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.skipIf(!testTensorRT || !hasTensorRT)('matches the pinned real-model output with the ONNX Runtime TensorRT provider', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      executionProviders: ['tensorrt', 'cuda', 'cpu'],
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, expected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('recognizes and reorders real Arabic text for display like upstream RapidOCR', async () => {
    const result = await arabicRecognizer!.recognizeLine(resolve(modelsDirectory, 'arabic.png'))
    expect(result.text).toBe('امروز هوا خوبه')
  })

  it.each(upstreamLanguageCases)('matches upstream $ocrVersion $langType recognition on the pinned real image', async (testCase) => {
    const api = await import(packageName) as PackageApi
    const modelSelection = {
      langType: testCase.langType,
      modelType: 'mobile',
      ocrVersion: testCase.ocrVersion ?? 'PP-OCRv5',
    } as const
    const imagePath = resolve(modelsDirectory, testCase.image)

    if (testCase.fullPage) {
      const instance = await api.RapidOCR.create({
        intraOpNumThreads: 1,
        modelDirectory: modelsDirectory,
        recognitionModel: modelSelection,
      })
      try {
        const result = await instance.recognize(imagePath)
        expect(result.type).toBe('ocr')
        if (result.type !== 'ocr')
          throw new TypeError('Expected a full OCR result.')
        expect(result.lines[0]?.text).toBe(testCase.text)
      }
      finally {
        await instance.close()
      }
      return
    }

    const paths = await api.downloadModels({ modelDirectory: modelsDirectory, recognitionModel: modelSelection })
    const instance = await api.TextRecognizer.create({
      dictionaryPath: paths.dictionaryPath,
      intraOpNumThreads: 1,
      modelPath: paths.recognitionModelPath,
      modelSelection,
    })
    try {
      const result = await instance.recognizeLine(imagePath)
      expect(result.text).toBe(testCase.text)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.each(upstreamFullPageCases)('matches the upstream PP-OCRv5 $langType full-page output with classification disabled', async (testCase) => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: testCase.langType, modelType: 'mobile', ocrVersion: 'PP-OCRv5' },
      useClassifier: false,
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, testCase.image))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, testCase.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.each(upstreamClassifierFullPageCases)('matches the upstream PP-OCRv5 $langType full-page output with the default classifier', async (testCase) => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: testCase.langType, modelType: 'mobile', ocrVersion: 'PP-OCRv5' },
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, testCase.image))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, testCase.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.each(upstreamPPv4FullPageCases)('matches the upstream PP-OCRv4 $langType full-page output with the default classifier', async (testCase) => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: testCase.langType, modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, testCase.image))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, testCase.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.each(upstreamPPv6FullPageCases)('matches the upstream PP-OCRv6 small $langType full-page route', async (testCase) => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: testCase.langType, modelType: 'small', ocrVersion: 'PP-OCRv6' },
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, testCase.image))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, testCase.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('keeps multilingual PP-OCRv6 small page references for English, Japanese, and Latin routes', () => {
    expect(Object.keys(upstreamPPv6FullPageReferences.languages).sort()).toEqual(['en', 'japan', 'la'])
    expect(upstreamPPv6FullPageCases.every(testCase => testCase.lines.length > 0)).toBe(true)
  })

  it('matches the upstream PP-OCRv4 traditional Chinese route on a traditional Chinese full page', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: 'chinese_cht', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
    })
    try {
      const result = await instance.recognize(chineseChtSyntheticImagePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, chineseChtFullPageReference.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it.each(upstreamRecognitionFullPageCases)(
    'matches the upstream $version $modelType $language full-page route ($modelKey)',
    async (testCase) => {
      const api = await import(packageName) as PackageApi
      const instance = await api.RapidOCR.create({
        intraOpNumThreads: 1,
        modelDirectory: modelsDirectory,
        recognitionModel: {
          langType: testCase.language,
          modelType: testCase.modelType,
          ocrVersion: testCase.version,
        },
      })
      try {
        const result = await instance.recognize(resolve(modelsDirectory, testCase.fixture))
        expect(result.type).toBe('ocr')
        if (result.type !== 'ocr')
          throw new TypeError('Expected a full OCR result.')
        expectLinesMatchReference(
          result.lines,
          testCase.lines,
          0.03,
          getKnownRecognitionFullPageTextDifferences(testCase.modelKey, testCase.fixture),
        )
      }
      finally {
        await instance.close()
      }
    },
    120_000,
  )

  it.each(chineseRouteFullPageCases)(
    'matches a second upstream Chinese page on $version $modelType ($modelKey)',
    async (testCase) => {
      const api = await import(packageName) as PackageApi
      const instance = await api.RapidOCR.create({
        classifierModel: { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
        detectionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
        executionProviders: ['cpu'],
        intraOpNumThreads: 1,
        modelDirectory: modelsDirectory,
        recognitionModel: {
          langType: testCase.language,
          modelType: testCase.modelType,
          ocrVersion: testCase.version,
        },
      })
      try {
        const result = await instance.recognize(resolve(modelsDirectory, testCase.fixture))
        expect(result.type).toBe('ocr')
        if (result.type !== 'ocr')
          throw new TypeError('Expected a full OCR result for the second page.')
        expectLinesMatchReference(
          result.lines,
          testCase.lines,
          0.04,
          knownChineseVehiclePageTextDifferences[testCase.modelKey] ?? [],
        )
      }
      finally {
        await instance.close()
      }
    },
    120_000,
  )

  it('matches the upstream PP-OCRv4 Japanese output on a shared lossless raster', async () => {
    const api = await import(packageName) as PackageApi
    const japaneseExpected = JSON.parse(await readFile(
      new URL('../fixtures/japan-decoded-png.expected.json', import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const japaneseRaster = await sharp(resolve(modelsDirectory, 'japan.jpg')).png().toBuffer()
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: 'japan', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
    })
    try {
      const result = await instance.recognize(japaneseRaster)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, japaneseExpected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches the Japanese reference when decoding the original JPEG input', async () => {
    const api = await import(packageName) as PackageApi
    const japaneseExpected = JSON.parse(await readFile(
      new URL('../fixtures/japan-original-jpeg.expected.json', import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: 'japan', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, 'japan.jpg'))
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, japaneseExpected.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches the upstream PP-OCRv4 Chinese server-recognition route', async () => {
    const api = await import(packageName) as PackageApi
    const mobile = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' } as const
    const server = { langType: 'ch', modelType: 'server', ocrVersion: 'PP-OCRv4' } as const
    const paths = await api.downloadModels({
      classifierModel: mobile,
      detectionModel: mobile,
      modelDirectory: modelsDirectory,
      recognitionModel: server,
    })
    const instance = await api.RapidOCR.create({
      ...paths,
      classifierModel: mobile,
      detectionModel: mobile,
      intraOpNumThreads: 1,
      recognitionModel: server,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expect(result.lines[0]?.text).toBe('正品促销')
    }
    finally {
      await instance.close()
    }
  }, 180_000)

  it('matches the upstream PP-OCRv4 server detector output with both model outputs', async () => {
    const api = await import(packageName) as PackageApi
    const reference = JSON.parse(await readFile(
      new URL('../fixtures/ppocrv4-server-det.expected.json', import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const mobile = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' } as const
    const server = { langType: 'ch', modelType: 'server', ocrVersion: 'PP-OCRv4' } as const
    const paths = await api.downloadModels({
      classifierModel: mobile,
      detectionModel: server,
      modelDirectory: modelsDirectory,
      recognitionModel: mobile,
    })
    const instance = await api.RapidOCR.create({
      ...paths,
      classifierModel: mobile,
      detectionModel: server,
      intraOpNumThreads: 1,
      recognitionModel: mobile,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, reference.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches the upstream PP-OCRv5 Chinese server page output', async () => {
    const api = await import(packageName) as PackageApi
    const reference = JSON.parse(await readFile(
      new URL('../fixtures/ppocrv5-server.expected.json', import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const mobile = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv5' } as const
    const server = { langType: 'ch', modelType: 'server', ocrVersion: 'PP-OCRv5' } as const
    const paths = await api.downloadModels({
      classifierModel: mobile,
      detectionModel: mobile,
      modelDirectory: modelsDirectory,
      recognitionModel: server,
    })
    const instance = await api.RapidOCR.create({
      ...paths,
      classifierModel: mobile,
      detectionModel: mobile,
      intraOpNumThreads: 1,
      recognitionModel: server,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, reference.lines)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches the upstream PP-OCRv4 Chinese server recognizer page output', async () => {
    const api = await import(packageName) as PackageApi
    const reference = JSON.parse(await readFile(
      new URL('../fixtures/ppocrv4-server-recognizer.expected.json', import.meta.url),
      'utf8',
    )) as ExpectedOutput
    const classifier = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' } as const
    const detector = { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' } as const
    const recognizer = { langType: 'ch', modelType: 'server', ocrVersion: 'PP-OCRv4' } as const
    const paths = await api.downloadModels({
      classifierModel: classifier,
      detectionModel: detector,
      modelDirectory: modelsDirectory,
      recognitionModel: recognizer,
    })
    const instance = await api.RapidOCR.create({
      ...paths,
      classifierModel: classifier,
      detectionModel: detector,
      intraOpNumThreads: 1,
      recognitionModel: recognizer,
    })
    try {
      const result = await instance.recognize(samplePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result.')
      expectLinesMatchReference(result.lines, reference.lines)
    }
    finally {
      await instance.close()
    }
  }, 180_000)

  it.each(['mobile', 'server'] as const)('matches the upstream PP-OCRv5 %s orientation classifier', async (modelType) => {
    const api = await import(packageName) as PackageApi
    const selection = { langType: 'ch', modelType, ocrVersion: 'PP-OCRv5' } as const
    const paths = await api.downloadModels({ classifierModel: selection, modelDirectory: modelsDirectory })
    const instance = await api.TextClassifier.create(paths.classifierModelPath, 1)
    try {
      const result = await instance.classifyLines([resolve(modelsDirectory, 'text_cls.jpg')])
      expect(result[0]?.angle).toBe(180)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('matches the upstream sample text, reading order, boxes, and confidence within tolerance', async () => {
    const result = await ocr!.recognize(samplePath)
    expect(result.type).toBe('ocr')
    if (result.type !== 'ocr')
      throw new TypeError('Expected a full OCR result.')
    expectLinesMatchReference(result.lines, expected.lines)
    expect(result.img.width).toBeGreaterThan(0)
    expect(result.img.height).toBeGreaterThan(0)
    expect(result.boxes).toEqual(result.lines.map(line => line.box.map(point => [point.x, point.y])))
    expect(result.boxPoints).toEqual(result.lines.map(line => line.box))
    expect(result.txts).toEqual(result.lines.map(line => line.text))
    expect(result.scores).toEqual(result.lines.map(line => line.confidence))
    expect(result.wordResults).toEqual(result.lines.map(line => line.words ?? []))
    expect(result.elapse_list).toEqual(result.elapseList)
    expect(result.word_results).toHaveLength(result.lines.length)
    expect(result.elapseList).toHaveLength(3)
    expect(result.elapse).toBeCloseTo(result.elapseList.reduce((sum, value) => sum + value, 0), 8)
    expect(result.toJSON()).toHaveLength(expected.lines.length)
    expect(result.toJSON()?.[0]).toEqual({
      box: expected.lines[0]!.box,
      score: expect.any(Number),
      txt: expected.lines[0]!.text,
    })
    expect(result.toMarkdown()).toContain(expected.lines[0]!.text)
    const renderedPath = resolve(modelsDirectory, 'rapidocr-result.png')
    const rendered = await result.vis(renderedPath)
    expect(rendered?.format).toBe('BGR')
    const renderedMetadata = await sharp(Buffer.from(rendered!.data), {
      raw: { width: rendered!.width, height: rendered!.height, channels: 3 },
    }).metadata()
    const sourceMetadata = await sharp(samplePath).metadata()
    expect(renderedMetadata.width).toBe(sourceMetadata.width! * 2)
    expect(renderedMetadata.height).toBe(sourceMetadata.height)
    expect(await readFile(renderedPath)).toEqual(await result.visBuffer(renderedPath))
    expectTimings(result)
    expect(result.timings.classification).toBeGreaterThan(0)
  })

  it('matches a second upstream document page, including a box at the image edge', async () => {
    const result = await ocr!.recognize(resolve(modelsDirectory, 'ch_doc_server.png'))

    expectLinesMatchReference(result.lines, documentExpected.lines)
  })

  it('matches upstream on a dense Chinese vehicle-registration scan', async () => {
    const result = await ocr!.recognize(resolve(modelsDirectory, 'check_return_word_len.jpeg'))

    expectLinesMatchReference(
      result.lines,
      registrationDocumentExpected.lines,
      0.04,
      knownChineseVehiclePageTextDifferences['multi_PP-OCRv6_rec_small'] ?? [],
    )
  }, 120_000)

  it('matches the pinned Python output for a real-model vertical Chinese page', async () => {
    const result = await ocr!.recognize(verticalPagePath)
    expect(result.type).toBe('ocr')
    if (result.type !== 'ocr')
      throw new TypeError('Expected a full OCR result for vertical text.')

    expectLinesMatchReference(result.lines, verticalPageExpected.lines)
    for (const line of result.lines) {
      const width = Math.max(...line.box.map(point => point.x)) - Math.min(...line.box.map(point => point.x))
      const height = Math.max(...line.box.map(point => point.y)) - Math.min(...line.box.map(point => point.y))
      expect(height).toBeGreaterThan(width)
    }
  })

  it('matches the pinned Python output for a real-model vertical Japanese page', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      classifierModel: { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
      detectionModel: { langType: 'multi', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
      recognitionModel: { langType: 'japan', modelType: 'mobile', ocrVersion: 'PP-OCRv4' },
    })
    try {
      const result = await instance.recognize(verticalJapanesePagePath)
      expect(result.type).toBe('ocr')
      if (result.type !== 'ocr')
        throw new TypeError('Expected a full OCR result for vertical Japanese text.')

      expectLinesMatchReference(result.lines, verticalJapaneseExpected.lines)
      expect(result.lines.some((line) => {
        const width = Math.max(...line.box.map(point => point.x)) - Math.min(...line.box.map(point => point.x))
        const height = Math.max(...line.box.map(point => point.y)) - Math.min(...line.box.map(point => point.y))
        return height > width
      })).toBe(true)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('applies upstream detector and text-score thresholds per call', async () => {
    const noBoxes = await ocr!.recognize(samplePath, { boxThreshold: 0.99 })
    expect(noBoxes.type).toBe('empty')
    expect(noBoxes.lines).toEqual([])
    expect(noBoxes.timings.detection).toBeGreaterThan(0)
    expect(noBoxes.timings.recognition).toBe(0)

    const noHighScores = await ocr!.recognize(samplePath, { textScoreThreshold: 1 })
    expect(noHighScores.lines.length).toBeLessThan(expected.lines.length)
    expect(noHighScores.lines.every(line => line.confidence === 1)).toBe(true)
    expect(noHighScores.timings.recognition).toBeGreaterThan(0)
  })

  it('persists upstream update_params settings between full-page calls', async () => {
    const api = await import(packageName) as PackageApi
    const instance = await api.RapidOCR.create({
      intraOpNumThreads: 1,
      modelDirectory: modelsDirectory,
    })
    try {
      const initial = await instance.recognize(samplePath)
      expect(initial.type).toBe('ocr')

      instance.updateParams({ useClassifier: false, returnWordBoxes: true })
      const withoutClassifier = await instance.recognize(samplePath)
      expect(withoutClassifier.type).toBe('ocr')
      if (withoutClassifier.type !== 'ocr')
        throw new TypeError('Expected full OCR output after persistent update.')
      expect(withoutClassifier.timings.classification).toBe(0)
      expect(withoutClassifier.lines[0]?.words?.length).toBeGreaterThan(0)

      instance.updateParams({ useRecognition: false })
      expect((await instance.recognize(samplePath)).type).toBe('detection')

      instance.updateParams({ useDetection: false, useRecognition: true, returnWordBoxes: false })
      expect((await instance.recognize(samplePath)).type).toBe('recognition')

      instance.updateParams({ useDetection: true, boxThreshold: 1 })
      expect((await instance.recognize(samplePath)).type).toBe('empty')

      instance.updateParams({ boxThreshold: 0.5, textScoreThreshold: 1 })
      const filtered = await instance.recognize(samplePath)
      expect(filtered.lines.length).toBeLessThan(expected.lines.length)
      expect(filtered.lines.every(line => line.confidence === 1)).toBe(true)
    }
    finally {
      await instance.close()
    }
  }, 120_000)

  it('returns Chinese character boxes when requested', async () => {
    const result = await ocr!.recognize(samplePath, { returnWordBoxes: true })
    const firstLine = result.lines[0]!

    expect(firstLine.text).toBe('正品促销')
    expect(firstLine.words?.map(word => word.text)).toEqual(['正', '品', '促', '销'])
    expect(firstLine.words?.every(word => word.box.length === 4 && word.confidence > 0)).toBe(true)
  })

  it('preserves custom upstream classifier labels on a real orientation model', async () => {
    const api = await import(packageName) as PackageApi
    const classifierModel = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv4' } as const
    const paths = await api.downloadModels({ classifierModel, modelDirectory: modelsDirectory })
    const instance = await api.RapidOCR.create({
      ...paths,
      autoDownloadModels: false,
      classifierModel,
      intraOpNumThreads: 1,
      params: { 'Cls.label_list': ['upright', 'inverted 180'] },
      useDetection: false,
      useRecognition: false,
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, 'text_cls.jpg'))
      expect(result.type).toBe('classification')
      if (result.type !== 'classification')
        throw new TypeError('Expected a classification-only result.')
      expect(result.clsRes[0]?.[0]).toBe('inverted 180')
      expect(result.results[0]?.angle).toBe(180)
    }
    finally {
      await instance.close()
    }
  })

  it('uses the PP-OCRv5 classifier input shape when upstream YAML cls_image_shape is customized', async () => {
    const api = await import(packageName) as PackageApi
    const classifierModel = { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv5' } as const
    const paths = await api.downloadModels({ classifierModel, modelDirectory: modelsDirectory })
    const instance = await api.RapidOCR.create({
      ...paths,
      autoDownloadModels: false,
      classifierModel,
      params: {
        'Cls.ocr_version': 'PP-OCRv5',
        'Cls.cls_image_shape': [3, 64, 256],
      },
      useDetection: false,
      useRecognition: false,
    })
    try {
      const result = await instance.recognize(resolve(modelsDirectory, 'text_cls.jpg'))
      expect(result.type).toBe('classification')
      if (result.type !== 'classification')
        throw new TypeError('Expected a classification-only result.')
      expect(result.clsRes).toHaveLength(1)
      expect(Number.isFinite(result.clsRes[0]?.[1])).toBe(true)
    }
    finally {
      await instance.close()
    }
  })

  it('maps character boxes back through an upside-down classifier crop', async () => {
    const upsideDown = await sharp(samplePath).rotate(180).png().toBuffer()
    const result = await ocr!.recognize(upsideDown, { returnWordBoxes: true })
    expect(result.type).toBe('ocr')
    if (result.type !== 'ocr')
      throw new TypeError('Expected a full OCR result.')

    const line = result.lines.find(item => item.text === '正品促销')
    expect(line?.words?.map(word => word.text)).toEqual(['正', '品', '促', '销'])
    const xs = line!.box.map(point => point.x)
    const ys = line!.box.map(point => point.y)
    for (const word of line!.words!) {
      expect(word.box.every(point => point.x >= Math.min(...xs) - 1 && point.x <= Math.max(...xs) + 1)).toBe(true)
      expect(word.box.every(point => point.y >= Math.min(...ys) - 1 && point.y <= Math.max(...ys) + 1)).toBe(true)
      expect(Math.abs(word.box[0].x - word.box[1].x) + Math.abs(word.box[0].y - word.box[1].y)).toBeGreaterThan(0)
    }
  })

  it('returns detection-only and classification-only outputs when stages are disabled', async () => {
    const detection = await ocr!.recognize(samplePath, { useRecognition: false, useClassifier: false })
    expect(detection.type).toBe('detection')
    if (detection.type !== 'detection')
      throw new TypeError('Expected a detection-only result.')
    expect(detection.boxes).toHaveLength(expected.lines.length)
    expect(detection.scores).toHaveLength(detection.boxes.length)
    expect(detection.img.width).toBeGreaterThan(0)
    expect(Object.keys(detection)).not.toContain('img')
    expect(detection.elapseList[0]).toBe(detection.timings.detection)
    expect(detection.timings.recognition).toBe(0)
    const detectionVisualization = await detection.vis()
    expect((await sharp(Buffer.from(detectionVisualization!.data), {
      raw: { width: detectionVisualization!.width, height: detectionVisualization!.height, channels: 3 },
    }).metadata()).width).toBe(detection.img.width)

    const classification = await ocr!.recognize(resolve(modelsDirectory, 'text_cls.jpg'), {
      useDetection: false,
      useRecognition: false,
      useClassifier: true,
    })
    expect(classification.type).toBe('classification')
    if (classification.type !== 'classification')
      throw new TypeError('Expected a classification-only result.')
    expect(classification.results).toEqual([expect.objectContaining({ angle: 180 })])
    expect(classification.clsRes).toEqual([['180', expect.any(Number)]])
    expect(classification.imgList).toHaveLength(1)
    expect(classification.img.width).toBeGreaterThan(0)
    expect(Object.keys(classification)).not.toContain('imgList')
    expect(classification.cls_res).toEqual(classification.clsRes)
    expect(classification.img_list).toHaveLength(1)
    const classificationVisualization = await classification.vis()
    expect((await sharp(Buffer.from(classificationVisualization!.data), {
      raw: { width: classificationVisualization!.width, height: classificationVisualization!.height, channels: 3 },
    }).metadata()).width).toBe(classification.img.width * 2)
  })

  it('matches the upstream low-resolution regression after ties-to-even detector resizing', async () => {
    const lowResolution = await sharp(samplePath)
      .resize(162, 215, { fit: 'fill' })
      .jpeg({ quality: 80 })
      .toBuffer()
    const result = await ocr!.recognize(lowResolution)

    expectLinesMatchReference(result.lines, lowResolutionExpected.lines, 0.04)
  })

  it('accepts encoded image bytes and reuses the loaded model sessions', async () => {
    const bytes = await readFile(samplePath)
    const result = await ocr!.recognize(bytes)

    expectLinesMatchReference(result.lines, expected.lines)
    expectTimings(result)
  })

  it('runs real models with raw BGR and grayscale buffers', async () => {
    const rgbImage = await sharp(samplePath)
      .removeAlpha()
      .toColourspace('srgb')
      .raw()
      .toBuffer({ resolveWithObject: true })
    const bgr = Buffer.from(rgbImage.data)
    for (let offset = 0; offset < bgr.length; offset += 3) {
      const red = bgr[offset]!
      bgr[offset] = bgr[offset + 2]!
      bgr[offset + 2] = red
    }
    const grayImage = await sharp(samplePath).greyscale().raw().toBuffer({ resolveWithObject: true })

    const bgrResult = await ocr!.recognize({
      data: bgr,
      format: 'BGR24',
      height: rgbImage.info.height,
      width: rgbImage.info.width,
    })
    const grayResult = await ocr!.recognize({
      data: grayImage.data,
      format: 'GRAY8',
      height: grayImage.info.height,
      width: grayImage.info.width,
    })

    expect(bgrResult.lines[0]!.text).toBe('正品促销')
    expect(grayResult.lines.some(line => /[\u3400-\u9FFF]/u.test(line.text))).toBe(true)
  })

  it('classifies an upside-down text line before recognition when detection is disabled', async () => {
    const result = await ocr!.recognize(resolve(modelsDirectory, 'text_cls.jpg'), { useDetection: false })

    expect(result.type).toBe('recognition')
    if (result.type !== 'recognition')
      throw new TypeError('Expected a recognition-only result.')
    expect(result.results).toHaveLength(1)
    expect(result.results[0]!.text).toBe('怪我咯')
    expect(result.imgs).toHaveLength(1)
    expect(result.txts).toEqual(result.results.map(item => item.text))
    expect(result.scores).toEqual(result.results.map(item => item.confidence))
    expect(Object.keys(result)).not.toContain('imgs')
    expect(result.timings.detection).toBe(0)
    expect(result.timings.classification).toBeGreaterThan(0)
    const recognitionVisualization = await result.vis()
    expect((await sharp(Buffer.from(recognitionVisualization!.data), {
      raw: { width: recognitionVisualization!.width, height: recognitionVisualization!.height, channels: 3 },
    }).metadata()).width).toBe(result.imgs[0]!.width * 2)
  })

  it('recognizes an already-cropped line with the real PP-OCRv6 model', async () => {
    const result = await recognizer!.recognizeLine(resolve(modelsDirectory, 'text_rec.jpg'))

    expect(result.text).toBe('韩国小馆')
    expect(result.confidence).toBeGreaterThan(0.95)
  })

  it('recognizes Latin text with the routed PP-OCRv5 English model and dictionary', async () => {
    const crop = await sharp(samplePath)
      .extract({ left: 245, top: 224, width: 67, height: 38 })
      .png()
      .toBuffer()
    const result = await englishRecognizer!.recognizeLine(crop, { returnWordPositions: true })

    expect(result.text).toBe('10.0')
    expect(result.confidence).toBeGreaterThan(0.5)
    expect(result.wordPositions?.map(position => position.text)).toEqual(['10.0'])
  })

  it('exposes standalone detection and orientation-classification APIs', async () => {
    const detections = await detector!.detect(samplePath)
    expect(detections).toHaveLength(expected.lines.length)
    expect(detections[0]!.score).toBeGreaterThan(0.5)
    expect(detections.img.width).toBeGreaterThan(0)
    expect(detections.boxes).toEqual(detections.map(detection => detection.box.map(point => [point.x, point.y])))
    expect(detections.scores).toEqual(detections.map(detection => detection.score))
    expect(detections.elapse).toBeGreaterThan(0)
    expect(detections.elapse).toBeLessThan(60)
    expect(Object.keys(detections)).not.toContain('img')
    expect((await detections.vis())?.format).toBe('BGR')

    const orientations = await classifier!.classifyLines([resolve(modelsDirectory, 'text_cls.jpg')])
    expect(orientations).toEqual([
      expect.objectContaining({ angle: 180 }),
    ])
    expect(orientations[0]!.confidence).toBeGreaterThan(0.9)
    expect(orientations.imgList).toHaveLength(1)
    expect(orientations.clsRes).toEqual([['180', orientations[0]!.confidence]])
    expect(orientations.elapse).toBeGreaterThan(0)
    expect(orientations.elapse).toBeLessThan(60)
  })

  it('returns upstream stage metadata from standalone text recognition', async () => {
    const result = await recognizer!.recognizeLines([resolve(modelsDirectory, 'text_rec.jpg')])

    expect(result).toHaveLength(1)
    expect(result[0]!.text).toBe('韩国小馆')
    expect(result.imgs).toHaveLength(1)
    expect(result.txts).toEqual(result.map(line => line.text))
    expect(result.scores).toEqual(result.map(line => line.confidence))
    expect(result.wordResults).toEqual([undefined])
    expect(result.elapse).toBeGreaterThan(0)
    expect(result.elapse).toBeLessThan(60)
    expect(Object.keys(result)).not.toContain('imgs')
    expect((await result.vis())?.format).toBe('BGR')
  })

  it('runs the DB slow contour-scoring mode against the real PP-OCRv6 detector', async () => {
    const slowDetector = await (await import(packageName) as PackageApi).TextDetector.create(
      resolve(modelsDirectory, 'PP-OCRv6_det_small.onnx'),
      1,
      { scoreMode: 'slow' },
    )
    try {
      const detections = await slowDetector.detect(samplePath)
      expect(detections.length).toBeGreaterThan(0)
      expect(detections.every(detection => detection.score >= 0.5)).toBe(true)
    }
    finally {
      await slowDetector.close()
    }
  })

  it('maps boxes back to source coordinates after maximum-side image resizing', async () => {
    const enlarged = await sharp(samplePath).resize(1615, 2150).jpeg().toBuffer()
    const result = await ocr!.recognize(enlarged)

    expect(result.lines).toHaveLength(expected.lines.length)
    expect(result.lines[0]!.text).toBe('正品促销')
    expect(result.lines.at(-1)!.text).toBe('假一赔十')
    expect(result.lines[1]!.box[0].x).toBeGreaterThan(300)
    expect(result.lines[1]!.box[0].x).toBeLessThan(360)
    expect(result.lines[1]!.box[0].y).toBeGreaterThan(430)
    expect(result.lines[1]!.box[0].y).toBeLessThan(500)
  })

  it('returns no text lines for a blank raw RGB image', async () => {
    const result = await ocr!.recognize({
      data: new Uint8Array(320 * 320 * 3).fill(255),
      height: 320,
      width: 320,
    })

    expect(result.lines).toEqual([])
    expect(result.timings.classification).toBe(0)
    expect(result.timings.recognition).toBe(0)
  })

  it('matches Python empty results at minimum and maximum preprocessing dimensions', async () => {
    expect(blankBoundaryExpected.upstream).toBe('RapidOCR v3.10.0')
    for (const expectedCase of blankBoundaryExpected.cases) {
      const result = await ocr!.recognize({
        data: new Uint8Array(expectedCase.width * expectedCase.height * 3).fill(255),
        height: expectedCase.height,
        width: expectedCase.width,
      })
      expect(result.type).toBe(expectedCase.type)
      expect(result.lines.map(line => line.text)).toEqual(expectedCase.texts)
    }
  }, 120_000)

  it('waits for active inference before closing and rejects new work afterward', async () => {
    const pending = ocr!.recognize(samplePath)
    const closing = ocr!.close()

    await expect(pending).resolves.toHaveProperty('lines.length', expected.lines.length)
    await expect(closing).resolves.toBeUndefined()
    await expect(ocr!.close()).resolves.toBeUndefined()
    await expect(ocr!.recognize(samplePath)).rejects.toThrow(/closed/)
  })
})

function expectTimings(result: RapidOCRResult): void {
  expect(result.timings.detection).toBeGreaterThanOrEqual(0)
  expect(result.timings.classification).toBeGreaterThanOrEqual(0)
  expect(result.timings.recognition).toBeGreaterThanOrEqual(0)
  // These upstream-compatible fields use seconds, not performance.now() milliseconds.
  expect(result.elapse).toBeLessThan(60)
  expect(Object.values(result.timings).every(Number.isFinite)).toBe(true)
}

function readNonNegativeIntegerEnv(name: string): number {
  const value = env[name]
  if (value === undefined)
    return 0
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new RangeError(`${name} must be a non-negative integer.`)
  return parsed
}

function expectLinesMatchReference(
  actualLines: RapidOCRResult['lines'],
  referenceLines: ExpectedLine[],
  confidenceTolerance = 0.03,
  knownTextDifferences: readonly KnownRuntimeTextDifference[] = [],
): void {
  expect(actualLines).toHaveLength(referenceLines.length)

  for (const [lineIndex, actual] of actualLines.entries()) {
    const reference = referenceLines[lineIndex]!
    const knownTextDifference = knownTextDifferences.find(difference => difference.index === lineIndex)
    if (knownTextDifference) {
      expect(reference.text).toBe(knownTextDifference.upstreamText)
      expect(actual.text).toBe(knownTextDifference.nodeText)
    }
    else {
      expect(normalizeCurrencyGlyph(actual.text)).toBe(normalizeCurrencyGlyph(reference.text))
    }
    expect(Math.abs(actual.confidence - reference.confidence))
      .toBeLessThanOrEqual(knownTextDifference?.confidenceTolerance ?? confidenceTolerance)
    for (const [pointIndex, point] of actual.box.entries()) {
      const expectedPoint = reference.box[pointIndex]!
      expect(Math.abs(point.x - expectedPoint[0]!)).toBeLessThanOrEqual(2)
      expect(Math.abs(point.y - expectedPoint[1]!)).toBeLessThanOrEqual(2)
    }
  }
}

function normalizeCurrencyGlyph(text: string): string {
  return text.replaceAll('￥', '¥')
}
