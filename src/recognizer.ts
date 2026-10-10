import type { DecodedRgbImage } from './image.js'
/*
 * TypeScript adaptation of the RapidOCR v3.10.0 CTC text-recognition pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { ImageInput, RecognitionResult, TextRecognizerOptions } from './types.js'
import { readFile } from 'node:fs/promises'
import * as ort from 'onnxruntime-node'
import { decodeCtcOutput, readDictionary } from './ctc.js'
import { decodeRgbImage, resizeRgbImage } from './image.js'
import { resolveModelPaths } from './models.js'

const IMAGE_HEIGHT = 48
const IMAGE_WIDTH = 320
const CHANNELS = 3

/** Runs a PP-OCR compatible CTC recognition model on cropped text lines. */
export class TextRecognizer {
  private activeCalls = 0
  private closePromise: Promise<void> | undefined
  private idleResolver: (() => void) | undefined
  private state: 'open' | 'closing' | 'closed' = 'open'

  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly characters: readonly string[],
    private readonly batchSize: number,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedBatchSize: number | undefined,
    private readonly fixedWidth: number | undefined,
  ) {}

  /** Loads the ONNX session and matching character dictionary once for reuse. */
  static async create(options: TextRecognizerOptions = {}): Promise<TextRecognizer> {
    const batchSize = options.batchSize ?? 6
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError('batchSize must be a positive integer.')
    }

    if (options.intraOpNumThreads !== undefined
      && (!Number.isInteger(options.intraOpNumThreads) || options.intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }

    const modelPaths = await resolveModelPaths({
      paths: {
        ...(options.modelPath === undefined ? {} : { recognitionModelPath: options.modelPath }),
        ...(options.dictionaryPath === undefined ? {} : { dictionaryPath: options.dictionaryPath }),
      },
      required: ['recognitionModelPath', 'dictionaryPath'],
      ...(options.modelDirectory === undefined ? {} : { modelDirectory: options.modelDirectory }),
      ...(options.autoDownloadModels === undefined ? {} : { autoDownloadModels: options.autoDownloadModels }),
    })

    const dictionaryContents = await readFile(modelPaths.dictionaryPath, 'utf8')
    const characters = readDictionary(dictionaryContents)
    const session = await ort.InferenceSession.create(modelPaths.recognitionModelPath, {
      executionProviders: ['cpu'],
      ...(options.intraOpNumThreads === undefined
        ? {}
        : { intraOpNumThreads: options.intraOpNumThreads }),
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

      if (input.shape.length !== 4 || input.shape[1] !== CHANNELS || input.shape[2] !== IMAGE_HEIGHT) {
        throw new Error('Recognition model input must have the PP-OCR shape [batch, 3, 48, width].')
      }

      return new TextRecognizer(
        session,
        characters,
        batchSize,
        inputName,
        outputName,
        typeof input.shape[0] === 'number' ? input.shape[0] : undefined,
        typeof input.shape[3] === 'number' ? input.shape[3] : undefined,
      )
    }
    catch (error) {
      await session.release()
      throw error
    }
  }

  /** Recognizes one already-cropped text line. */
  async recognizeLine(input: ImageInput): Promise<RecognitionResult> {
    const [result] = await this.recognizeLines([input])
    if (!result) {
      throw new Error('Unable to read the recognition result.')
    }
    return result
  }

  /** Recognizes multiple already-cropped text lines and preserves input order. */
  async recognizeLines(inputs: readonly ImageInput[]): Promise<RecognitionResult[]> {
    this.assertOpen()
    if (inputs.length === 0) {
      return []
    }

    this.activeCalls++
    try {
      const images = await Promise.all(inputs.map(input => decodeRgbImage(input)))
      const capacity = this.fixedBatchSize ?? this.batchSize
      const orderedImages = images
        .map((image, index) => ({ image, index, ratio: image.width / image.height }))
        .sort((left, right) => left.ratio - right.ratio)
      const results = new Map<number, RecognitionResult>()

      for (let offset = 0; offset < orderedImages.length; offset += capacity) {
        const batch = orderedImages.slice(offset, offset + capacity)
        const output = await this.runBatch(batch.map(item => item.image))
        for (let index = 0; index < batch.length; index++) {
          const item = batch[index]
          if (!item) {
            throw new Error('Unable to map a recognition result back to its input image.')
          }
          results.set(item.index, decodeCtcOutput(output.data, output.dims, index, this.characters))
        }
      }

      return images.map((_, index) => {
        const result = results.get(index)
        if (!result) {
          throw new Error('Recognition result is missing for an input image.')
        }
        return result
      })
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
      IMAGE_WIDTH / IMAGE_HEIGHT,
      ...images.map(image => image.width / image.height),
    )
    const width = this.fixedWidth ?? Math.max(IMAGE_WIDTH, Math.trunc(IMAGE_HEIGHT * maxRatio))
    const batchCount = this.fixedBatchSize ?? images.length
    const inputData = new Float32Array(batchCount * CHANNELS * IMAGE_HEIGHT * width)

    for (let batchIndex = 0; batchIndex < images.length; batchIndex++) {
      const image = images[batchIndex]
      if (!image) {
        continue
      }

      const resizedWidth = Math.min(
        Math.ceil(IMAGE_HEIGHT * image.width / image.height),
        width,
      )
      const pixels = await resizeRgbImage(image, resizedWidth, IMAGE_HEIGHT)
      const planeSize = IMAGE_HEIGHT * width
      const imageOffset = batchIndex * CHANNELS * planeSize

      for (let y = 0; y < IMAGE_HEIGHT; y++) {
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

    const tensor = new ort.Tensor('float32', inputData, [batchCount, CHANNELS, IMAGE_HEIGHT, width])
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
