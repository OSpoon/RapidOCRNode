import type { DecodedRgbImage } from '../image/image.js'
import type { ImageInput, RapidOCRCallOptions, RapidOCREmptyResult, RapidOCROptions, RapidOCRResult, RapidOCRUpdateParams, RecognizedLine } from '../shared/types.js'
/*
 * TypeScript adaptation of the RapidOCR v3.10.0 end-to-end pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DetectorPostprocessOptions } from './detection/detector.js'
import type { VisualizationSettings } from './output/visualization.js'
import { performance } from 'node:perf_hooks'
import { resolveRapidOCROptions } from '../config/rapidocr.js'
import { cropPerspective } from '../image/crop.js'
import { addVerticalPadding, decodeRgbImage, roundToEven } from '../image/image.js'
import { resizeRgbImageOpenCv } from '../image/opencv.js'
import { resolveModelPaths } from '../models/download.js'
import { resolveModelRoute } from '../models/registry.js'
import { createRapidOCRLogger } from '../shared/logger.js'
import { TextClassifier } from './classification/classifier.js'
import { TextDetector, validateDetectorPostprocessOptions } from './detection/detector.js'
import { toRapidOCRBoxes, toRapidOCRImage, toRapidOCRWordInfo, toRapidOCRWordResults } from './output/compat.js'
import { addEmptyExportMethods, addOcrExportMethods } from './output/export.js'
import { addStageVisualization } from './output/metadata.js'
import { renderDetectionStage, renderLineStage } from './output/visualization.js'
import { TextRecognizer } from './recognition/recognizer.js'
import { mapWordPositionsToBox } from './recognition/word-boxes.js'

const MAX_IMAGE_SIDE = 2000
const MIN_IMAGE_SIDE = 30
// OpenCV resize variants can move borderline classifier scores by about 0.0043.
const CLASSIFIER_RETRY_MARGIN = 0.005

interface ImagePreprocessOptions {
  maxSideLength: number
  minSideLength: number
  usePreprocessImage: boolean
}

interface VerticalPaddingOptions {
  minHeight: number
  useVerticalPadding: boolean
  widthHeightRatio: number
}

/** Reusable end-to-end PP-OCR pipeline for page and document images. */
export class RapidOCR {
  private detector: TextDetector | undefined
  private detectorPromise: Promise<TextDetector> | undefined
  private recognizer: TextRecognizer | undefined
  private recognizerPromise: Promise<TextRecognizer> | undefined
  private classifier: TextClassifier | undefined
  private classifierPromise: Promise<TextClassifier> | undefined
  private activeCalls = 0
  private closePromise: Promise<void> | undefined
  private idleResolver: (() => void) | undefined
  private state: 'open' | 'closing' | 'closed' = 'open'
  private textScoreThreshold: number
  private useDetectionByDefault: boolean
  private useClassifierByDefault: boolean
  private useRecognitionByDefault: boolean
  private readonly classifierThreshold: number
  private returnWordBoxesByDefault: boolean
  private returnSingleCharBoxesByDefault: boolean
  private readonly executionProviderWarnings = new Set<string>()
  private readonly logger: ReturnType<typeof createRapidOCRLogger>

  private constructor(
    private readonly options: RapidOCROptions,
    private readonly detectorOptions: DetectorPostprocessOptions,
    private readonly imagePreprocessOptions: ImagePreprocessOptions,
    private readonly verticalPaddingOptions: VerticalPaddingOptions,
    private readonly visualizationSettings: VisualizationSettings,
  ) {
    this.logger = createRapidOCRLogger(options.logLevel ?? 'info')
    this.textScoreThreshold = options.textScoreThreshold ?? 0.5
    this.useDetectionByDefault = options.useDetection ?? true
    this.useClassifierByDefault = options.useClassifier ?? true
    this.useRecognitionByDefault = options.useRecognition ?? true
    this.classifierThreshold = options.classifierThreshold ?? 0.9
    this.returnWordBoxesByDefault = options.returnWordBoxes ?? false
    this.returnSingleCharBoxesByDefault = options.returnSingleCharBoxes ?? false
  }

  /** Validates configuration; stage models are downloaded and loaded on first use. */
  static async create(inputOptions: RapidOCROptions = {}): Promise<RapidOCR> {
    const options = await resolveRapidOCROptions(inputOptions)
    const textScoreThreshold = options.textScoreThreshold ?? 0.5
    const imagePreprocessOptions = {
      maxSideLength: options.maxSideLength ?? MAX_IMAGE_SIDE,
      minSideLength: options.minSideLength ?? MIN_IMAGE_SIDE,
      usePreprocessImage: options.usePreprocessImage ?? true,
    }
    const verticalPaddingOptions = {
      minHeight: options.minHeight ?? MIN_IMAGE_SIDE,
      useVerticalPadding: options.useVerticalPadding ?? true,
      widthHeightRatio: options.widthHeightRatio ?? 8,
    }
    if (!Number.isFinite(textScoreThreshold) || textScoreThreshold < 0 || textScoreThreshold > 1) {
      throw new RangeError('textScoreThreshold must be between 0 and 1.')
    }
    if (options.classifierThreshold !== undefined
      && (!Number.isFinite(options.classifierThreshold) || options.classifierThreshold < 0 || options.classifierThreshold > 1)) {
      throw new RangeError('classifierThreshold must be between 0 and 1.')
    }
    if (options.classifierLabels !== undefined
      && (!Array.isArray(options.classifierLabels) || options.classifierLabels.length !== 2 || options.classifierLabels.some(label => typeof label !== 'string'))) {
      throw new TypeError('classifierLabels must contain exactly two strings.')
    }
    if (options.intraOpNumThreads !== undefined
      && (!Number.isInteger(options.intraOpNumThreads) || options.intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }
    if (options.interOpNumThreads !== undefined
      && (!Number.isInteger(options.interOpNumThreads) || options.interOpNumThreads < 1)) {
      throw new RangeError('interOpNumThreads must be a positive integer.')
    }
    for (const [stage, providers] of Object.entries({
      fallback: options.executionProviders,
      detection: options.detectionExecutionProviders,
      recognition: options.recognitionExecutionProviders,
      classification: options.classifierExecutionProviders,
    })) {
      if (providers !== undefined && providers.length === 0) {
        throw new RangeError(`${stage} executionProviders must contain at least one provider.`)
      }
    }
    if (options.batchSize !== undefined && (!Number.isInteger(options.batchSize) || options.batchSize < 1)) {
      throw new RangeError('batchSize must be a positive integer.')
    }
    if (options.classifierBatchSize !== undefined && (!Number.isInteger(options.classifierBatchSize) || options.classifierBatchSize < 1)) {
      throw new RangeError('classifierBatchSize must be a positive integer.')
    }
    validateImagePreprocessOptions(imagePreprocessOptions, verticalPaddingOptions)

    const detectorOptions: DetectorPostprocessOptions = {
      ...(options.detectionThreshold === undefined ? {} : { threshold: options.detectionThreshold }),
      ...(options.boxThreshold === undefined ? {} : { boxThreshold: options.boxThreshold }),
      ...(options.unclipRatio === undefined ? {} : { unclipRatio: options.unclipRatio }),
      ...(options.maxCandidates === undefined ? {} : { maxCandidates: options.maxCandidates }),
      ...(options.useDilation === undefined ? {} : { useDilation: options.useDilation }),
      ...(options.scoreMode === undefined ? {} : { scoreMode: options.scoreMode }),
      ...(options.limitSideLength === undefined ? {} : { limitSideLength: options.limitSideLength }),
      ...(options.limitType === undefined ? {} : { limitType: options.limitType }),
      ...(options.detectionMean === undefined ? {} : { mean: options.detectionMean }),
      ...(options.detectionStd === undefined ? {} : { std: options.detectionStd }),
    }
    validateDetectorPostprocessOptions(detectorOptions)

    return new RapidOCR(
      options,
      detectorOptions,
      imagePreprocessOptions,
      verticalPaddingOptions,
      {
        autoDownloadFonts: options.autoDownloadFonts,
        fontPath: options.fontPath,
        langType: options.recognitionModel?.langType ?? 'ch',
        modelDirectory: options.modelDirectory,
      },
    )
  }

  /** Loads the models enabled by this instance without processing an image. */
  async warmup(): Promise<void> {
    this.assertOpen()
    this.activeCalls++
    try {
      const results = await Promise.allSettled([
        ...(this.useDetectionByDefault ? [this.getDetector()] : []),
        ...(this.useClassifierByDefault ? [this.getClassifier()] : []),
        ...(this.useRecognitionByDefault ? [this.getRecognizer()] : []),
      ])
      const failure = results.find(result => result.status === 'rejected')
      if (failure?.status === 'rejected')
        throw failure.reason
    }
    finally {
      this.activeCalls--
      if (this.activeCalls === 0 && this.state === 'closing') {
        this.idleResolver?.()
        this.idleResolver = undefined
      }
    }
  }

  /** Persists upstream-compatible stage, output, and score settings for later calls. */
  updateParams(options: RapidOCRUpdateParams): void {
    this.assertOpen()
    if (this.activeCalls > 0) {
      throw new TypeError('updateParams() must be called between active recognize() calls.')
    }
    if (options === null || typeof options !== 'object' || Array.isArray(options)) {
      throw new TypeError('updateParams() expects an object of supported RapidOCR options.')
    }

    const supportedKeys: readonly (keyof RapidOCRUpdateParams)[] = [
      'boxThreshold',
      'returnSingleCharBoxes',
      'returnWordBoxes',
      'textScoreThreshold',
      'unclipRatio',
      'useClassifier',
      'useDetection',
      'useRecognition',
    ]
    for (const key of Object.keys(options)) {
      if (!supportedKeys.includes(key as keyof RapidOCRUpdateParams))
        throw new TypeError(`Unknown RapidOCR update parameter: ${key}.`)
    }

    validateCallOptions(options)
    for (const key of ['useClassifier', 'useDetection', 'useRecognition', 'returnWordBoxes', 'returnSingleCharBoxes'] as const) {
      if (options[key] !== undefined && typeof options[key] !== 'boolean')
        throw new TypeError(`${key} must be a boolean.`)
    }

    if (options.boxThreshold !== undefined)
      this.detectorOptions.boxThreshold = options.boxThreshold
    if (options.unclipRatio !== undefined)
      this.detectorOptions.unclipRatio = options.unclipRatio
    if (options.textScoreThreshold !== undefined)
      this.textScoreThreshold = options.textScoreThreshold
    if (options.useDetection !== undefined)
      this.useDetectionByDefault = options.useDetection
    if (options.useClassifier !== undefined)
      this.useClassifierByDefault = options.useClassifier
    if (options.useRecognition !== undefined)
      this.useRecognitionByDefault = options.useRecognition
    if (options.returnWordBoxes !== undefined)
      this.returnWordBoxesByDefault = options.returnWordBoxes
    if (options.returnSingleCharBoxes !== undefined)
      this.returnSingleCharBoxesByDefault = options.returnSingleCharBoxes
  }

  /** Detects, crops, optionally classifies orientation, and recognizes text. */
  async recognize(input: ImageInput, callOptions: RapidOCRCallOptions = {}): Promise<RapidOCRResult> {
    this.assertOpen()
    validateCallOptions(callOptions)
    const useDetection = callOptions.useDetection ?? this.useDetectionByDefault
    const useClassifier = callOptions.useClassifier ?? this.useClassifierByDefault
    const useRecognition = callOptions.useRecognition ?? this.useRecognitionByDefault
    const returnWordBoxes = callOptions.returnWordBoxes ?? this.returnWordBoxesByDefault
    const returnSingleCharBoxes = callOptions.returnSingleCharBoxes ?? this.returnSingleCharBoxesByDefault
    this.activeCalls++
    try {
      const source = await decodeRgbImage(input)
      const scaled = await resizeForDetection(source, this.imagePreprocessOptions)
      const padded = useDetection && this.verticalPaddingOptions.useVerticalPadding
        ? addVerticalPadding(scaled.image, this.verticalPaddingOptions.widthHeightRatio, this.verticalPaddingOptions.minHeight)
        : { image: scaled.image, top: 0 }
      const detectionImage = padded.image

      const detector = useDetection ? await this.getDetector() : undefined
      const detectionOutput = useDetection
        ? await detector!.detect(detectionImage, {
            ...(callOptions.boxThreshold === undefined ? {} : { boxThreshold: callOptions.boxThreshold }),
            ...(callOptions.unclipRatio === undefined ? {} : { unclipRatio: callOptions.unclipRatio }),
          })
        : undefined
      const detections = detectionOutput ?? [{ box: fullImageBox(scaled.image.width, scaled.image.height), score: 1 }]
      const detection = detectionOutput?.elapse ?? 0
      if (detections.length === 0) {
        return createEmptyResult({ classification: 0, detection, recognition: 0 })
      }

      let crops = useDetection
        ? await Promise.all(detections.map(item => cropPerspective(detectionImage, item.box)))
        : [scaled.image]
      const wordMappingCrops = crops

      let classification = 0
      let classificationResults: Array<{ angle: 0 | 180, confidence: number }> | undefined
      let classificationLabels: Array<[string, number]> | undefined
      if (useClassifier) {
        const classifier = await this.getClassifier()
        const classificationStart = performance.now()
        const classified = await classifier.classifyAndRotateDetailed(crops)
        crops = classified.images
        classificationResults = classified.results
        classificationLabels = classified.clsRes
        classification = (performance.now() - classificationStart) / 1000
      }

      if (!useRecognition) {
        const timings = { classification, detection, recognition: 0 }
        if (useDetection) {
          const boxes = detections.map(item => mapBoxToSource(item.box, scaled.scaleX, scaled.scaleY, source.width, source.height, padded.top))
          const scores = detections.map(item => item.score)
          return addStageVisualization(makeImageFieldsNonEnumerable({
            type: 'detection',
            ...timingFields(timings),
            img: toRapidOCRImage(source),
            boxes: toRapidOCRBoxes(boxes),
            boxPoints: boxes,
            scores,
            lines: [],
          }), savePath => renderDetectionStage(source, boxes, scores, savePath))
        }
        if (classificationResults && classificationLabels) {
          return addStageVisualization(makeImageFieldsNonEnumerable({
            type: 'classification',
            ...timingFields(timings),
            img: toRapidOCRImage(source),
            imgList: crops,
            img_list: crops.map(toRapidOCRImage),
            lines: [],
            clsRes: classificationLabels,
            cls_res: classificationLabels,
            results: classificationResults,
          }), savePath => renderLineStage(
            crops,
            classificationLabels.map(([label]) => label),
            classificationResults.map(result => result.confidence),
            this.visualizationSettings,
            savePath,
          ))
        }
        return createEmptyResult(timings)
      }

      const recognizer = await this.getRecognizer()
      const textScoreThreshold = callOptions.textScoreThreshold ?? this.textScoreThreshold
      const recognitionOptions = {
        returnSingleCharBoxes,
        returnWordPositions: returnWordBoxes || returnSingleCharBoxes,
      }
      const recognized = await recognizer.recognizeLines(crops, recognitionOptions)
      let recognition = recognized.elapse
      const retriedUnrotated = new Set<number>()
      const uncertainRotations = classificationResults?.flatMap((orientation, index) => {
        const recognizedLine = recognized[index]
        const wasRotated = orientation.angle === 180 && orientation.confidence > this.classifierThreshold
        const nearThreshold = orientation.confidence <= this.classifierThreshold + CLASSIFIER_RETRY_MARGIN
        return useDetection && wasRotated && nearThreshold && recognizedLine
          ? [index]
          : []
      }) ?? []

      if (uncertainRotations.length > 0) {
        const retryCrops = uncertainRotations.map(index => wordMappingCrops[index]!)
        const retried = await recognizer.recognizeLines(retryCrops, recognitionOptions)
        recognition += retried.elapse
        for (const [retryIndex, lineIndex] of uncertainRotations.entries()) {
          const current = recognized[lineIndex]
          const alternative = retried[retryIndex]
          if (current && alternative && alternative.confidence > textScoreThreshold && alternative.confidence > current.confidence) {
            recognized[lineIndex] = alternative
            retriedUnrotated.add(lineIndex)
          }
        }
      }

      if (!useDetection) {
        const results = recognized.filter(result => result.text.trim().length > 0)
        return addStageVisualization(makeImageFieldsNonEnumerable({
          type: 'recognition',
          ...timingFields({ classification, detection, recognition }),
          img: toRapidOCRImage(source),
          imgs: crops.map(toRapidOCRImage),
          imgsRgb: crops,
          lines: [],
          results,
          txts: results.map(result => result.text),
          scores: results.map(result => result.confidence),
          word_results: results.map(result => toRapidOCRWordInfo(result.wordPositions)),
          lang_rec: this.langRec,
        }), savePath => renderLineStage(
          crops,
          results.map(result => result.text),
          results.map(result => result.confidence),
          this.visualizationSettings,
          savePath,
        ))
      }

      const lines: RecognizedLine[] = []
      for (let index = 0; index < detections.length; index++) {
        const result = recognized[index]
        const detectionResult = detections[index]
        if (result && detectionResult && result.confidence >= textScoreThreshold) {
          const { wordPositions, ...recognitionResult } = result
          const orientation = classificationResults?.[index]
          const lineBox = mapBoxToSource(detectionResult.box, scaled.scaleX, scaled.scaleY, source.width, source.height, padded.top)
          const wordCrop = wordMappingCrops[index]
          const words = (returnWordBoxes || returnSingleCharBoxes) && wordCrop
            ? await mapWordPositionsToBox(
                lineBox,
                wordCrop.width,
                wordCrop.height,
                result.text,
                wordPositions ?? [],
                orientation?.angle === 180
                && orientation.confidence > this.classifierThreshold
                && !retriedUnrotated.has(index),
              )
            : undefined
          lines.push({
            ...recognitionResult,
            box: lineBox,
            ...(words === undefined ? {} : { words }),
          })
        }
      }

      const timings = { classification, detection, recognition }
      if (lines.length === 0)
        return createEmptyResult(timings)

      return addOcrExportMethods(
        {
          type: 'ocr',
          ...timingFields(timings),
          img: toRapidOCRImage(source),
          lines,
          boxes: toRapidOCRBoxes(lines.map(line => line.box)),
          boxPoints: lines.map(line => line.box),
          txts: lines.map(line => line.text),
          scores: lines.map(line => line.confidence),
          wordResults: lines.map(line => line.words ?? []),
          word_results: toRapidOCRWordResults(lines),
          lang_rec: this.langRec,
        },
        source,
        this.visualizationSettings,
      )
    }
    finally {
      this.activeCalls--
      if (this.activeCalls === 0 && this.state === 'closing') {
        this.idleResolver?.()
        this.idleResolver = undefined
      }
    }
  }

  /** Releases native model sessions after active recognition calls finish. */
  close(): Promise<void> {
    if (this.closePromise)
      return this.closePromise
    this.state = 'closing'
    this.closePromise = (async () => {
      if (this.activeCalls > 0) {
        await new Promise<void>((resolve) => {
          this.idleResolver = resolve
        })
      }
      const releases = [
        ...(this.recognizer ? [this.recognizer.close()] : []),
        ...(this.classifier ? [this.classifier.close()] : []),
        ...(this.detector ? [this.detector.close()] : []),
      ]
      const results = await Promise.allSettled(releases)
      this.state = 'closed'
      const failed = results.find(result => result.status === 'rejected')
      if (failed?.status === 'rejected')
        throw failed.reason
    })()
    return this.closePromise
  }

  private assertOpen(): void {
    if (this.state !== 'open') {
      throw new TypeError(`RapidOCR is ${this.state} and cannot start another recognition call.`)
    }
  }

  private get langRec(): string {
    const selection = this.options.recognitionModel
    const version = selection?.ocrVersion ?? 'PP-OCRv6'
    const modelType = selection?.modelType ?? (version === 'PP-OCRv6' ? 'small' : 'mobile')
    return `${selection?.langType ?? 'ch'}_${modelType}`
  }

  private logExecutionProviderFallback(message: string): void {
    if (this.executionProviderWarnings.has(message))
      return
    this.executionProviderWarnings.add(message)
    this.logger('warning', message)
  }

  private async getDetector(): Promise<TextDetector> {
    if (this.detector)
      return this.detector
    if (!this.detectorPromise) {
      const pending = this.createDetector()
      this.detectorPromise = pending
      try {
        return await pending
      }
      finally {
        if (this.detectorPromise === pending)
          this.detectorPromise = undefined
      }
    }
    return this.detectorPromise
  }

  private async createDetector(): Promise<TextDetector> {
    if (this.options.detectionSession) {
      this.logger('debug', 'Using caller-provided detection session.')
      const detector = await TextDetector.createWithSession(this.options.detectionSession, this.detectorOptions)
      this.detector = detector
      return detector
    }
    const assets = this.options.detectionModelPath === undefined
      ? { detectionModelPath: resolveModelRoute('det', this.options.detectionModel).asset }
      : {}
    const paths = await resolveModelPaths({
      paths: this.options.detectionModelPath === undefined ? {} : { detectionModelPath: this.options.detectionModelPath },
      assets,
      required: ['detectionModelPath'],
      ...(this.options.modelDirectory === undefined ? {} : { modelDirectory: this.options.modelDirectory }),
      ...(this.options.autoDownloadModels === undefined ? {} : { autoDownloadModels: this.options.autoDownloadModels }),
    })
    const detector = await TextDetector.create(
      paths.detectionModelPath,
      this.options.intraOpNumThreads,
      this.detectorOptions,
      this.options.detectionExecutionProviders ?? this.options.executionProviders ?? ['cpu'],
      this.options.interOpNumThreads,
      this.options.enableCpuMemArena,
      message => this.logExecutionProviderFallback(message),
    )
    this.logger('info', `Using detection model ${paths.detectionModelPath}`)
    this.detector = detector
    return detector
  }

  private async getClassifier(): Promise<TextClassifier> {
    if (this.classifier)
      return this.classifier
    if (!this.classifierPromise) {
      const pending = this.createClassifier()
      this.classifierPromise = pending
      try {
        return await pending
      }
      finally {
        if (this.classifierPromise === pending)
          this.classifierPromise = undefined
      }
    }
    return this.classifierPromise
  }

  private async createClassifier(): Promise<TextClassifier> {
    if (this.options.classifierSession) {
      this.logger('debug', 'Using caller-provided classifier session.')
      const classifier = await TextClassifier.createWithSession(this.options.classifierSession, {
        ...(this.options.classifierThreshold === undefined ? {} : { rotateThreshold: this.options.classifierThreshold }),
        ...(this.options.classifierBatchSize === undefined ? {} : { batchSize: this.options.classifierBatchSize }),
        ...(this.options.classifierImageShape === undefined ? {} : { imageShape: this.options.classifierImageShape }),
        ...(this.options.classifierLabels === undefined ? {} : { labelList: this.options.classifierLabels }),
      })
      this.classifier = classifier
      return classifier
    }
    const assets = this.options.classifierModelPath === undefined
      ? { classifierModelPath: resolveModelRoute('cls', this.options.classifierModel ?? { ocrVersion: 'PP-OCRv4' }).asset }
      : {}
    const paths = await resolveModelPaths({
      paths: this.options.classifierModelPath === undefined ? {} : { classifierModelPath: this.options.classifierModelPath },
      assets,
      required: ['classifierModelPath'],
      ...(this.options.modelDirectory === undefined ? {} : { modelDirectory: this.options.modelDirectory }),
      ...(this.options.autoDownloadModels === undefined ? {} : { autoDownloadModels: this.options.autoDownloadModels }),
    })
    const classifier = await TextClassifier.create(
      paths.classifierModelPath,
      this.options.intraOpNumThreads,
      this.options.classifierExecutionProviders ?? this.options.executionProviders ?? ['cpu'],
      this.options.interOpNumThreads,
      this.options.classifierThreshold,
      this.options.classifierBatchSize,
      this.options.classifierImageShape,
      this.options.enableCpuMemArena,
      message => this.logExecutionProviderFallback(message),
      this.options.classifierLabels,
    )
    this.logger('info', `Using classifier model ${paths.classifierModelPath}`)
    this.classifier = classifier
    return classifier
  }

  private async getRecognizer(): Promise<TextRecognizer> {
    if (this.recognizer)
      return this.recognizer
    if (!this.recognizerPromise) {
      const pending = TextRecognizer.create({
        ...(this.options.recognitionModelPath === undefined ? {} : { modelPath: this.options.recognitionModelPath }),
        ...(this.options.dictionaryPath === undefined ? {} : { dictionaryPath: this.options.dictionaryPath }),
        ...(this.options.batchSize === undefined ? {} : { batchSize: this.options.batchSize }),
        ...(this.options.intraOpNumThreads === undefined ? {} : { intraOpNumThreads: this.options.intraOpNumThreads }),
        ...(this.options.interOpNumThreads === undefined ? {} : { interOpNumThreads: this.options.interOpNumThreads }),
        ...(this.options.recognitionSession !== undefined || this.options.enableCpuMemArena === undefined
          ? {}
          : { enableCpuMemArena: this.options.enableCpuMemArena }),
        ...(this.options.recognitionImageShape === undefined ? {} : { imageShape: this.options.recognitionImageShape }),
        ...(this.options.modelDirectory === undefined ? {} : { modelDirectory: this.options.modelDirectory }),
        ...(this.options.autoDownloadModels === undefined ? {} : { autoDownloadModels: this.options.autoDownloadModels }),
        ...(this.options.recognitionSession === undefined ? {} : { session: this.options.recognitionSession }),
        modelSelection: this.options.recognitionModel,
        ...(this.options.recognitionSession !== undefined
          ? {}
          : { executionProviders: this.options.recognitionExecutionProviders ?? this.options.executionProviders ?? ['cpu'] }),
      }, message => this.logExecutionProviderFallback(message))
      if (this.options.recognitionSession) {
        this.logger('debug', 'Using caller-provided recognition session.')
      }
      else {
        const modelLabel = this.options.recognitionModelPath
          ?? resolveModelRoute('rec', this.options.recognitionModel).asset.file
        this.logger('info', `Using recognition model ${modelLabel}`)
      }
      this.recognizerPromise = pending
      try {
        const recognizer = await pending
        this.recognizer = recognizer
        return recognizer
      }
      finally {
        if (this.recognizerPromise === pending)
          this.recognizerPromise = undefined
      }
    }
    return this.recognizerPromise
  }
}

function validateCallOptions(options: RapidOCRCallOptions): void {
  if (options.boxThreshold !== undefined
    && (!Number.isFinite(options.boxThreshold) || options.boxThreshold < 0 || options.boxThreshold > 1)) {
    throw new RangeError('boxThreshold must be between 0 and 1.')
  }
  if (options.unclipRatio !== undefined && (!Number.isFinite(options.unclipRatio) || options.unclipRatio < 0)) {
    throw new RangeError('unclipRatio must be a finite non-negative number.')
  }
  if (options.textScoreThreshold !== undefined
    && (!Number.isFinite(options.textScoreThreshold) || options.textScoreThreshold < 0 || options.textScoreThreshold > 1)) {
    throw new RangeError('textScoreThreshold must be between 0 and 1.')
  }
}

function timingFields(timings: { classification: number, detection: number, recognition: number }): {
  elapse: number
  elapseList: [number, number, number]
  elapse_list: [number, number, number]
  timings: { classification: number, detection: number, recognition: number }
} {
  const elapseList: [number, number, number] = [timings.detection, timings.classification, timings.recognition]
  return { elapse: elapseList.reduce((total, value) => total + value, 0), elapseList, elapse_list: elapseList, timings }
}

function createEmptyResult(timings: { classification: number, detection: number, recognition: number }): RapidOCREmptyResult {
  return addEmptyExportMethods({
    type: 'empty' as const,
    ...timingFields(timings),
    img: null,
    lines: [] as [],
    boxes: null,
    boxPoints: null,
    txts: null,
    scores: null,
    wordResults: [] as [],
    word_results: [] as [],
  })
}

function makeImageFieldsNonEnumerable<T extends object>(result: T): T {
  for (const key of ['img', 'imgList', 'img_list', 'imgs', 'imgsRgb']) {
    if (Object.hasOwn(result, key))
      Object.defineProperty(result, key, { enumerable: false })
  }
  return result
}

export async function resizeForDetection(
  image: DecodedRgbImage,
  options: ImagePreprocessOptions = {
    maxSideLength: MAX_IMAGE_SIDE,
    minSideLength: MIN_IMAGE_SIDE,
    usePreprocessImage: true,
  },
): Promise<{
  image: DecodedRgbImage
  scaleX: number
  scaleY: number
}> {
  if (!options.usePreprocessImage)
    return { image, scaleX: 1, scaleY: 1 }

  const maxSide = Math.max(image.width, image.height)
  let current = image
  let scaleX = 1
  let scaleY = 1

  if (maxSide > options.maxSideLength) {
    const ratio = options.maxSideLength / maxSide
    const resized = await resizeWithinBounds(current, ratio)
    scaleX = image.width / resized.width
    scaleY = image.height / resized.height
    current = resized
  }

  const minSide = Math.min(current.width, current.height)
  if (minSide < options.minSideLength) {
    const ratio = options.minSideLength / minSide
    const resized = await resizeWithinBounds(current, ratio)
    scaleX *= current.width / resized.width
    scaleY *= current.height / resized.height
    current = resized
  }

  return { image: current, scaleX, scaleY }
}

async function resizeWithinBounds(image: DecodedRgbImage, ratio: number): Promise<DecodedRgbImage> {
  const height = roundTo32(Math.trunc(image.height * ratio))
  const width = roundTo32(Math.trunc(image.width * ratio))
  if (height === image.height && width === image.width) {
    return image
  }
  const data = await resizeRgbImageOpenCv(image, width, height)
  return { data, height, width }
}

function roundTo32(value: number): number {
  return Math.max(32, roundToEven(value / 32) * 32)
}

function mapBoxToSource(
  box: RecognizedLine['box'],
  scaleX: number,
  scaleY: number,
  width: number,
  height: number,
  paddingTop = 0,
): RecognizedLine['box'] {
  return [
    mapPointToSource(box[0], scaleX, scaleY, width, height, paddingTop),
    mapPointToSource(box[1], scaleX, scaleY, width, height, paddingTop),
    mapPointToSource(box[2], scaleX, scaleY, width, height, paddingTop),
    mapPointToSource(box[3], scaleX, scaleY, width, height, paddingTop),
  ]
}

function fullImageBox(width: number, height: number): RecognizedLine['box'] {
  const right = Math.max(0, width - 1)
  const bottom = Math.max(0, height - 1)
  return [
    { x: 0, y: 0 },
    { x: right, y: 0 },
    { x: right, y: bottom },
    { x: 0, y: bottom },
  ]
}

function mapPointToSource(
  point: RecognizedLine['box'][number],
  scaleX: number,
  scaleY: number,
  width: number,
  height: number,
  paddingTop: number,
): RecognizedLine['box'][number] {
  return {
    x: Math.max(0, Math.min(width, roundToEven(point.x * scaleX))),
    y: Math.max(0, Math.min(height, roundToEven((point.y - paddingTop) * scaleY))),
  }
}

function validateImagePreprocessOptions(
  image: ImagePreprocessOptions,
  padding: VerticalPaddingOptions,
): void {
  if (!Number.isFinite(image.minSideLength) || image.minSideLength < 1) {
    throw new RangeError('minSideLength must be a positive finite number.')
  }
  if (!Number.isFinite(image.maxSideLength) || image.maxSideLength < image.minSideLength) {
    throw new RangeError('maxSideLength must be finite and at least minSideLength.')
  }
  if (!Number.isFinite(padding.minHeight) || padding.minHeight < 1) {
    throw new RangeError('minHeight must be a positive finite number.')
  }
  if (!Number.isFinite(padding.widthHeightRatio)
    || (padding.widthHeightRatio <= 0 && padding.widthHeightRatio !== -1)) {
    throw new RangeError('widthHeightRatio must be positive or -1.')
  }
}
