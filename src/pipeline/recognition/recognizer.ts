import type { DecodedRgbImage } from '../../image/image.js'
/*
 * TypeScript adaptation of the RapidOCR v3.10.0 CTC text-recognition pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { ImageInput, RapidOCRImage, RapidOCRStageVisualization, RapidOCRWordInfo, RawRgbImage, RecognitionResult, TextRecognitionCallOptions, TextRecognizerOptions, WordPosition } from '../../shared/types.js'
import type { VisualizationSettings } from '../output/visualization.js'
import { readFile } from 'node:fs/promises'
import { performance } from 'node:perf_hooks'
import * as ort from 'onnxruntime-node'
import { createInferenceSession } from '../../backends/inference.js'
import { decodeRgbImage } from '../../image/image.js'
import { resizeRgbImageOpenCv } from '../../image/opencv.js'
import { resolveModelPaths, selectModelAssets } from '../../models/download.js'
import { resolveModelRoute } from '../../models/registry.js'
import { toRapidOCRImage, toRapidOCRWordInfo } from '../output/compat.js'
import { addArrayMetadata, addStageVisualization } from '../output/metadata.js'
import { renderLineStage } from '../output/visualization.js'
import { decodeCtcOutput, readDictionary } from './ctc.js'
import { reorderTextForDisplay } from './rtl.js'

const IMAGE_HEIGHT = 48
const IMAGE_WIDTH = 320
const CHANNELS = 3

function getLangRec(selection: TextRecognizerOptions['modelSelection']): string {
  const version = selection?.ocrVersion ?? 'PP-OCRv6'
  const modelType = selection?.modelType ?? (version === 'PP-OCRv6' ? 'small' : 'mobile')
  return `${selection?.langType ?? 'ch'}_${modelType}`
}

export interface TextRecognitionOutput extends Array<RecognitionResult>, RapidOCRStageVisualization {
  imgs: RapidOCRImage[]
  imgsRgb: RawRgbImage[]
  txts: string[]
  scores: number[]
  wordResults: Array<WordPosition[] | undefined>
  word_results: Array<RapidOCRWordInfo | null>
  lang_rec: string
  elapse: number
}

/** Runs a PP-OCR compatible CTC recognition model on cropped text lines. */
export class TextRecognizer {
  private activeCalls = 0
  private closePromise: Promise<void> | undefined
  private idleResolver: (() => void) | undefined
  private state: 'open' | 'closing' | 'closed' = 'open'

  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly ownsSession: boolean,
    private readonly characters: readonly string[],
    private readonly batchSize: number,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedBatchSize: number | undefined,
    private readonly fixedWidth: number | undefined,
    private readonly imageHeight: number,
    private readonly imageWidth: number,
    private readonly isArabic: boolean,
    private readonly langRec: string,
    private readonly visualizationSettings: VisualizationSettings,
  ) {}

  /** Loads the ONNX session and matching character dictionary once for reuse. */
  static async create(
    options: TextRecognizerOptions = {},
    onExecutionProviderFallback?: (message: string) => void,
  ): Promise<TextRecognizer> {
    const batchSize = options.batchSize ?? 6
    const [channels, imageHeight, imageWidth] = options.imageShape ?? [CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError('batchSize must be a positive integer.')
    }
    if (channels !== CHANNELS || !Number.isInteger(imageHeight) || imageHeight < 1 || !Number.isInteger(imageWidth) || imageWidth < 1) {
      throw new RangeError('imageShape must contain three values: 3, a positive height, and a positive width.')
    }

    if (options.intraOpNumThreads !== undefined
      && (!Number.isInteger(options.intraOpNumThreads) || options.intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }
    if (options.interOpNumThreads !== undefined
      && (!Number.isInteger(options.interOpNumThreads) || options.interOpNumThreads < 1)) {
      throw new RangeError('interOpNumThreads must be a positive integer.')
    }
    if (options.executionProviders !== undefined && options.executionProviders.length === 0) {
      throw new RangeError('executionProviders must contain at least one provider.')
    }
    if (options.session !== undefined
      && (options.executionProviders !== undefined
        || options.intraOpNumThreads !== undefined
        || options.interOpNumThreads !== undefined
        || options.enableCpuMemArena !== undefined)) {
      throw new TypeError('ONNX Runtime session options must be configured before passing session to TextRecognizer.')
    }
    const modelAssets = selectModelAssets({
      recognitionModel: options.modelSelection,
    }, false)
    const requiredModelPaths = options.session === undefined
      ? ['recognitionModelPath', 'dictionaryPath'] as const
      : ['dictionaryPath'] as const
    const modelPaths = await resolveModelPaths({
      paths: {
        ...(options.modelPath === undefined ? {} : { recognitionModelPath: options.modelPath }),
        ...(options.dictionaryPath === undefined ? {} : { dictionaryPath: options.dictionaryPath }),
      },
      assets: modelAssets,
      required: requiredModelPaths,
      ...(options.modelDirectory === undefined ? {} : { modelDirectory: options.modelDirectory }),
      ...(options.autoDownloadModels === undefined ? {} : { autoDownloadModels: options.autoDownloadModels }),
    })
    const recognitionModelPath = 'recognitionModelPath' in modelPaths ? modelPaths.recognitionModelPath : undefined
    if (options.session === undefined && recognitionModelPath === undefined) {
      throw new Error('Unable to resolve model path: recognitionModelPath')
    }

    const dictionaryContents = await readFile(modelPaths.dictionaryPath, 'utf8')
    const characters = readDictionary(dictionaryContents)
    const ownsSession = options.session === undefined
    const session = options.session ?? await createInferenceSession(recognitionModelPath!, {
      executionProviders: options.executionProviders ?? ['cpu'],
      ...(options.intraOpNumThreads === undefined
        ? {}
        : { intraOpNumThreads: options.intraOpNumThreads }),
      ...(options.interOpNumThreads === undefined
        ? {}
        : { interOpNumThreads: options.interOpNumThreads }),
      ...(options.enableCpuMemArena === undefined ? {} : { enableCpuMemArena: options.enableCpuMemArena }),
      ...(onExecutionProviderFallback === undefined ? {} : { onExecutionProviderFallback }),
    })

    try {
      const input = session.inputMetadata[0]
      const output = session.outputMetadata[0]
      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]

      if (session.inputMetadata.length !== 1
        || session.outputMetadata.length !== 1
        || !input
        || !output
        || !inputName
        || !outputName
        || !input.isTensor
        || !output.isTensor) {
        throw new Error('Recognition model must have one tensor input and one tensor output.')
      }

      if (input.type !== 'float32') {
        throw new Error(`Recognition model input must use float32; received ${input.type}.`)
      }

      if (output.type !== 'float32' || output.shape.length !== 3) {
        throw new Error('Recognition model output must be a float32 [batch, time, classes] tensor.')
      }

      if (input.shape.length !== 4
        || input.shape[1] !== CHANNELS
        || (typeof input.shape[2] === 'number' && input.shape[2] !== imageHeight)) {
        throw new Error(`Recognition model input must have the PP-OCR shape [batch, 3, ${imageHeight}, width].`)
      }

      return new TextRecognizer(
        session,
        ownsSession,
        characters,
        batchSize,
        inputName,
        outputName,
        typeof input.shape[0] === 'number' ? input.shape[0] : undefined,
        typeof input.shape[3] === 'number' ? input.shape[3] : undefined,
        imageHeight,
        imageWidth,
        resolveModelRoute('rec', options.modelSelection).modelKey.startsWith('arabic_'),
        getLangRec(options.modelSelection),
        {
          autoDownloadFonts: options.autoDownloadFonts,
          fontPath: options.fontPath,
          langType: options.modelSelection?.langType,
          modelDirectory: options.modelDirectory,
        },
      )
    }
    catch (error) {
      if (ownsSession)
        await session.release()
      throw error
    }
  }

  /** Recognizes one already-cropped text line. */
  async recognizeLine(input: ImageInput, options: TextRecognitionCallOptions = {}): Promise<RecognitionResult> {
    const [result] = await this.recognizeLines([input], options)
    if (!result) {
      throw new Error('Unable to read the recognition result.')
    }
    return result
  }

  /** Recognizes multiple already-cropped text lines and preserves input order. */
  async recognizeLines(
    inputs: readonly ImageInput[],
    options: TextRecognitionCallOptions = {},
  ): Promise<TextRecognitionOutput> {
    this.assertOpen()
    if (inputs.length === 0) {
      const results = addArrayMetadata([] as RecognitionResult[], {
        elapse: 0,
        imgs: [] as RapidOCRImage[],
        imgsRgb: [] as RawRgbImage[],
        scores: [] as number[],
        txts: [] as string[],
        wordResults: [] as Array<WordPosition[] | undefined>,
        word_results: [] as Array<RapidOCRWordInfo | null>,
        lang_rec: this.langRec,
      }) as TextRecognitionOutput
      return addStageVisualization(results, savePath => renderLineStage([], [], [], this.visualizationSettings, savePath))
    }

    this.activeCalls++
    try {
      const images = await Promise.all(inputs.map(input => decodeRgbImage(input)))
      const started = performance.now()
      const capacity = this.fixedBatchSize ?? this.batchSize
      const orderedImages = images
        .map((image, index) => ({ image, index, ratio: image.width / image.height }))
        .sort((left, right) => left.ratio - right.ratio)
      const resultsByIndex = new Map<number, RecognitionResult>()

      for (let offset = 0; offset < orderedImages.length; offset += capacity) {
        const batch = orderedImages.slice(offset, offset + capacity)
        const output = await this.runBatch(batch.map(item => item.image))
        const maxRatio = Math.max(this.imageWidth / this.imageHeight, ...batch.map(item => item.image.width / item.image.height))
        for (let index = 0; index < batch.length; index++) {
          const item = batch[index]
          if (!item) {
            throw new Error('Unable to map a recognition result back to its input image.')
          }
          resultsByIndex.set(item.index, decodeCtcOutput(output.data, output.dims, index, this.characters, {
            ...options,
            lineWidth: Math.max(1, (output.dims[1] ?? 1) * item.image.width / item.image.height / maxRatio),
          }))
        }
      }

      const recognizedResults = images.map((_, index) => {
        const result = resultsByIndex.get(index)
        if (!result) {
          throw new Error('Recognition result is missing for an input image.')
        }
        return this.isArabic ? { ...result, text: reorderTextForDisplay(result.text) } : result
      })
      const output = addArrayMetadata(recognizedResults, {
        elapse: (performance.now() - started) / 1000,
        imgs: images.map(toRapidOCRImage),
        imgsRgb: images,
        scores: recognizedResults.map(result => result.confidence),
        txts: recognizedResults.map(result => result.text),
        wordResults: recognizedResults.map(result => result.wordPositions),
        word_results: recognizedResults.map(result => toRapidOCRWordInfo(result.wordPositions)),
        lang_rec: this.langRec,
      }) as TextRecognitionOutput
      return addStageVisualization(output, savePath => renderLineStage(
        images,
        recognizedResults.map(result => result.text),
        recognizedResults.map(result => result.confidence),
        this.visualizationSettings,
        savePath,
      ))
    }
    finally {
      this.activeCalls--
      if (this.activeCalls === 0 && this.state === 'closing') {
        this.idleResolver?.()
        this.idleResolver = undefined
      }
    }
  }

  /** Releases the native ONNX session after any active inference calls finish. */
  close(): Promise<void> {
    if (this.closePromise) {
      return this.closePromise
    }

    this.state = 'closing'
    this.closePromise = (async () => {
      if (this.activeCalls > 0) {
        await new Promise<void>((resolve) => {
          this.idleResolver = resolve
        })
      }

      try {
        if (this.ownsSession)
          await this.session.release()
      }
      finally {
        this.state = 'closed'
      }
    })()

    return this.closePromise
  }

  private async runBatch(
    images: readonly DecodedRgbImage[],
  ): Promise<{ data: Float32Array, dims: readonly number[] }> {
    const maxRatio = Math.max(
      this.imageWidth / this.imageHeight,
      ...images.map(image => image.width / image.height),
    )
    const width = this.fixedWidth ?? Math.max(this.imageWidth, Math.trunc(this.imageHeight * maxRatio))
    const batchCount = this.fixedBatchSize ?? images.length
    const inputData = new Float32Array(batchCount * CHANNELS * this.imageHeight * width)

    for (let batchIndex = 0; batchIndex < images.length; batchIndex++) {
      const image = images[batchIndex]
      if (!image) {
        continue
      }

      const resizedWidth = Math.min(
        Math.ceil(this.imageHeight * image.width / image.height),
        width,
      )
      const pixels = await resizeRgbImageOpenCv(image, resizedWidth, this.imageHeight)
      const planeSize = this.imageHeight * width
      const imageOffset = batchIndex * CHANNELS * planeSize

      for (let y = 0; y < this.imageHeight; y++) {
        for (let x = 0; x < resizedWidth; x++) {
          const pixelOffset = (y * resizedWidth + x) * CHANNELS
          const targetOffset = y * width + x
          // Sharp and the public raw input use RGB; RapidOCR's OpenCV pipeline feeds BGR.
          inputData[imageOffset + targetOffset] = normalizePixel(pixels[pixelOffset + 2])
          inputData[imageOffset + planeSize + targetOffset] = normalizePixel(pixels[pixelOffset + 1])
          inputData[imageOffset + planeSize * 2 + targetOffset] = normalizePixel(pixels[pixelOffset])
        }
      }
    }

    const tensor = new ort.Tensor('float32', inputData, [batchCount, CHANNELS, this.imageHeight, width])
    const outputs = await this.session.run({ [this.inputName]: tensor })
    const output = outputs[this.outputName]
    if (!(output instanceof ort.Tensor) || output.type !== 'float32') {
      throw new TypeError('Recognition model output must be a float32 tensor.')
    }

    const data = output.data
    if (!(data instanceof Float32Array)) {
      throw new TypeError('Recognition model output must store float32 values in CPU memory.')
    }

    return { data, dims: output.dims }
  }

  private assertOpen(): void {
    if (this.state !== 'open') {
      throw new TypeError(`TextRecognizer is ${this.state} and cannot start another inference call.`)
    }
  }
}

function normalizePixel(pixel: number | undefined): number {
  if (pixel === undefined) {
    throw new Error('Image decoder returned an incomplete RGB buffer.')
  }
  return pixel / 127.5 - 1
}
