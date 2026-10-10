/*
 * TypeScript adaptation of the RapidOCR v3.10.0 end-to-end pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from './image.js'
import type { ImageInput, RapidOCRCallOptions, RapidOCROptions, RapidOCRResult, RecognizedLine } from './types.js'
import { performance } from 'node:perf_hooks'
import { TextClassifier } from './classifier.js'
import { cropPerspective } from './crop.js'
import { resizeRgbImageOpenCv, TextDetector } from './detector.js'
import { decodeRgbImage, resizeRgbImage, roundToEven } from './image.js'
import { resolveModelPaths } from './models.js'
import { TextRecognizer } from './recognizer.js'

const MAX_IMAGE_SIDE = 2000
const MIN_IMAGE_SIDE = 30

/** Reusable end-to-end PP-OCR pipeline for page and document images. */
export class RapidOCR {
  private activeCalls = 0
  private closePromise: Promise<void> | undefined
  private idleResolver: (() => void) | undefined
  private state: 'open' | 'closing' | 'closed' = 'open'

  private constructor(
    private readonly detector: TextDetector,
    private readonly recognizer: TextRecognizer,
    private readonly classifier: TextClassifier | undefined,
    private readonly textScoreThreshold: number,
  ) {}

  /** Loads all model sessions once so they can be shared across recognize calls. */
  static async create(options: RapidOCROptions = {}): Promise<RapidOCR> {
    const textScoreThreshold = options.textScoreThreshold ?? 0.5
    if (!Number.isFinite(textScoreThreshold) || textScoreThreshold < 0 || textScoreThreshold > 1) {
      throw new RangeError('textScoreThreshold must be between 0 and 1.')
    }
    if (options.intraOpNumThreads !== undefined
      && (!Number.isInteger(options.intraOpNumThreads) || options.intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }
    if (options.batchSize !== undefined && (!Number.isInteger(options.batchSize) || options.batchSize < 1)) {
      throw new RangeError('batchSize must be a positive integer.')
    }

    let detector: TextDetector | undefined
    let classifier: TextClassifier | undefined
    let recognizer: TextRecognizer | undefined
    try {
      const required: Array<'detectionModelPath' | 'recognitionModelPath' | 'dictionaryPath' | 'classifierModelPath'> = [
        'detectionModelPath',
        'recognitionModelPath',
        'dictionaryPath',
      ]
      if (options.classifierModelPath !== undefined || options.useClassifier === true)
        required.push('classifierModelPath')
      const modelPaths = await resolveModelPaths({
        paths: {
          ...(options.detectionModelPath === undefined ? {} : { detectionModelPath: options.detectionModelPath }),
          ...(options.recognitionModelPath === undefined ? {} : { recognitionModelPath: options.recognitionModelPath }),
          ...(options.dictionaryPath === undefined ? {} : { dictionaryPath: options.dictionaryPath }),
          ...(options.classifierModelPath === undefined ? {} : { classifierModelPath: options.classifierModelPath }),
        },
        required,
        ...(options.modelDirectory === undefined ? {} : { modelDirectory: options.modelDirectory }),
        ...(options.autoDownloadModels === undefined ? {} : { autoDownloadModels: options.autoDownloadModels }),
      })

      detector = await TextDetector.create(modelPaths.detectionModelPath, options.intraOpNumThreads)
      if (options.classifierModelPath !== undefined || options.useClassifier === true) {
        classifier = await TextClassifier.create(modelPaths.classifierModelPath, options.intraOpNumThreads)
      }
      recognizer = await TextRecognizer.create({
        modelPath: modelPaths.recognitionModelPath,
        dictionaryPath: modelPaths.dictionaryPath,
        ...(options.batchSize === undefined ? {} : { batchSize: options.batchSize }),
        ...(options.intraOpNumThreads === undefined ? {} : { intraOpNumThreads: options.intraOpNumThreads }),
      })
      return new RapidOCR(detector, recognizer, classifier, textScoreThreshold)
    }
    catch (error) {
      await Promise.allSettled([
        ...(recognizer ? [recognizer.close()] : []),
        ...(classifier ? [classifier.close()] : []),
        ...(detector ? [detector.close()] : []),
      ])
      throw error
    }
  }

  /** Detects, crops, optionally classifies orientation, and recognizes text. */
  async recognize(input: ImageInput, callOptions: RapidOCRCallOptions = {}): Promise<RapidOCRResult> {
    this.assertOpen()
    this.activeCalls++
    try {
      const source = await decodeRgbImage(input)
      const scaled = await resizeForDetection(source)

      const detectionStart = performance.now()
      const useDetection = callOptions.useDetection ?? true
      const detections = useDetection
        ? await this.detector.detect(scaled.image)
        : [{ box: fullImageBox(scaled.image.width, scaled.image.height), score: 1 }]
      const detection = useDetection ? performance.now() - detectionStart : 0
      if (detections.length === 0) {
        return { lines: [], timings: { classification: 0, detection, recognition: 0 } }
      }

      let crops = useDetection
        ? await Promise.all(detections.map(item => cropPerspective(scaled.image, item.box)))
        : [scaled.image]
      if (scaled.scaleX !== 1 || scaled.scaleY !== 1) {
        crops = await Promise.all(crops.map(async (crop) => {
          const width = Math.max(1, roundToEven(crop.width * scaled.scaleX))
          const height = Math.max(1, roundToEven(crop.height * scaled.scaleY))
          return { data: await resizeRgbImage(crop, width, height), height, width }
        }))
      }

      let classification = 0
      if (this.classifier && (callOptions.useClassifier ?? true)) {
        const classificationStart = performance.now()
        crops = await this.classifier.classifyAndRotate(crops)
        classification = performance.now() - classificationStart
      }

      const recognitionStart = performance.now()
      const recognized = await this.recognizer.recognizeLines(crops)
      const recognition = performance.now() - recognitionStart
      const lines: RecognizedLine[] = []
      for (let index = 0; index < detections.length; index++) {
        const result = recognized[index]
        const detectionResult = detections[index]
        if (result && detectionResult && result.confidence >= this.textScoreThreshold) {
          lines.push({
            ...result,
            box: mapBoxToSource(detectionResult.box, scaled.scaleX, scaled.scaleY, source.width, source.height),
          })
        }
      }

      return { lines, timings: { classification, detection, recognition } }
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
      const releases = [this.recognizer.close(), ...(this.classifier ? [this.classifier.close()] : []), this.detector.close()]
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
}

export async function resizeForDetection(image: DecodedRgbImage): Promise<{
  image: DecodedRgbImage
  scaleX: number
  scaleY: number
}> {
  const maxSide = Math.max(image.width, image.height)
  let current = image
  let scaleX = 1
  let scaleY = 1

  if (maxSide > MAX_IMAGE_SIDE) {
    const ratio = MAX_IMAGE_SIDE / maxSide
    const resized = await resizeWithinBounds(current, ratio)
    scaleX = image.width / resized.width
    scaleY = image.height / resized.height
    current = resized
  }

  const minSide = Math.min(current.width, current.height)
  if (minSide < MIN_IMAGE_SIDE) {
    const ratio = MIN_IMAGE_SIDE / minSide
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
): RecognizedLine['box'] {
  return [
    mapPointToSource(box[0], scaleX, scaleY, width, height),
    mapPointToSource(box[1], scaleX, scaleY, width, height),
    mapPointToSource(box[2], scaleX, scaleY, width, height),
    mapPointToSource(box[3], scaleX, scaleY, width, height),
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
): RecognizedLine['box'][number] {
  return {
    x: Math.max(0, Math.min(width, roundToEven(point.x * scaleX))),
    y: Math.max(0, Math.min(height, roundToEven(point.y * scaleY))),
  }
}
