/*
 * TypeScript adaptation of the RapidOCR v3.10.0 orientation-classification pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from './image.js'
import * as ort from 'onnxruntime-node'
import { rotate180 } from './crop.js'
import { resizeRgbImage } from './image.js'

const IMAGE_HEIGHT = 48
const IMAGE_WIDTH = 192
const CHANNELS = 3
const CLASSIFICATION_BATCH_SIZE = 6
const ROTATE_THRESHOLD = 0.9

export class TextClassifier {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedBatchSize: number | undefined,
    private readonly fixedWidth: number | undefined,
  ) {}

  static async create(modelPath: string, intraOpNumThreads?: number): Promise<TextClassifier> {
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu'],
      ...(intraOpNumThreads === undefined ? {} : { intraOpNumThreads }),
    })

    try {
      const input = session.inputMetadata[0]
      const output = session.outputMetadata[0]
      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      if (session.inputMetadata.length !== 1
        || session.outputMetadata.length !== 1
        || !input?.isTensor
        || !output?.isTensor
        || !inputName
        || !outputName
        || input.type !== 'float32'
        || output.type !== 'float32'
        || input.shape.length !== 4
        || input.shape[1] !== CHANNELS
        || (typeof input.shape[2] === 'number' && input.shape[2] !== IMAGE_HEIGHT)
        || output.shape.length !== 2
        || (typeof output.shape[1] === 'number' && output.shape[1] < 2)) {
        throw new Error('Classifier model must have one float32 [batch, 3, 48, width] input and one float32 output.')
      }

      return new TextClassifier(
        session,
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

  async classifyAndRotate(images: readonly DecodedRgbImage[]): Promise<DecodedRgbImage[]> {
    const results = [...images]
    const capacity = this.fixedBatchSize ?? CLASSIFICATION_BATCH_SIZE
    const ordered = images
      .map((image, index) => ({ image, index, ratio: image.width / image.height }))
      .sort((left, right) => left.ratio - right.ratio)

    for (let offset = 0; offset < ordered.length; offset += capacity) {
      const batch = ordered.slice(offset, offset + capacity)
      const { data, valuesPerItem } = await this.classifyBatch(batch)
      for (let index = 0; index < batch.length; index++) {
        const item = batch[index]!
        if (isUpsideDown(data, index, valuesPerItem)) {
          results[item.index] = rotate180(item.image)
        }
      }
    }

    return results
  }

  async close(): Promise<void> {
    await this.session.release()
  }

  private async classifyBatch(batch: readonly { image: DecodedRgbImage, index: number, ratio: number }[]): Promise<{ data: Float32Array, valuesPerItem: number }> {
    const width = this.fixedWidth ?? IMAGE_WIDTH
    const batchSize = this.fixedBatchSize ?? batch.length
    const planeSize = IMAGE_HEIGHT * width
    const inputData = new Float32Array(batchSize * CHANNELS * planeSize)
    for (let index = 0; index < batch.length; index++) {
      const image = batch[index]!.image
      const resizedWidth = Math.min(Math.ceil(IMAGE_HEIGHT * image.width / image.height), width)
      const pixels = await resizeRgbImage(image, resizedWidth, IMAGE_HEIGHT)
      writeNormalizedPixels(inputData, pixels, index, resizedWidth, width, planeSize)
    }

    const tensor = new ort.Tensor('float32', inputData, [batchSize, CHANNELS, IMAGE_HEIGHT, width])
    const outputs = await this.session.run({ [this.inputName]: tensor })
    const output = outputs[this.outputName]
    if (!(output instanceof ort.Tensor) || output.type !== 'float32' || !(output.data instanceof Float32Array)) {
      throw new TypeError('Classifier model output must be a float32 tensor in CPU memory.')
    }
    const valuesPerItem = output.data.length / batchSize
    if (!Number.isInteger(valuesPerItem) || valuesPerItem < 2) {
      throw new Error('Classifier model output must contain at least two scores per image.')
    }
    return { data: output.data, valuesPerItem }
  }
}

function writeNormalizedPixels(
  target: Float32Array,
  pixels: Uint8Array,
  batchIndex: number,
  resizedWidth: number,
  width: number,
  planeSize: number,
): void {
  const imageOffset = batchIndex * CHANNELS * planeSize
  for (let y = 0; y < IMAGE_HEIGHT; y++) {
    for (let x = 0; x < resizedWidth; x++) {
      const pixelOffset = (y * resizedWidth + x) * CHANNELS
      const targetOffset = y * width + x
      target[imageOffset + targetOffset] = normalize(pixels[pixelOffset + 2])
      target[imageOffset + planeSize + targetOffset] = normalize(pixels[pixelOffset + 1])
      target[imageOffset + planeSize * 2 + targetOffset] = normalize(pixels[pixelOffset])
    }
  }
}

function isUpsideDown(data: Float32Array, index: number, valuesPerItem: number): boolean {
  let bestClass = 0
  let bestScore = Number.NEGATIVE_INFINITY
  const start = index * valuesPerItem
  for (let classIndex = 0; classIndex < valuesPerItem; classIndex++) {
    const score = data[start + classIndex] ?? Number.NEGATIVE_INFINITY
    if (score > bestScore) {
      bestScore = score
      bestClass = classIndex
    }
  }
  return bestClass === 1 && bestScore > ROTATE_THRESHOLD
}

function normalize(pixel: number | undefined): number {
  if (pixel === undefined)
    throw new Error('Image decoder returned an incomplete RGB buffer.')
  return pixel / 127.5 - 1
}
