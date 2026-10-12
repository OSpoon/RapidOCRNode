/*
 * TypeScript adaptation of the RapidOCR v3.10.0 orientation-classification pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from '../../image/image.js'
import type { ExecutionProviderConfig, ImageInput, RapidOCRImage, RapidOCRStageVisualization, RawRgbImage } from '../../shared/types.js'
import { performance } from 'node:perf_hooks'
import * as ort from 'onnxruntime-node'
import { createInferenceSession } from '../../backends/inference.js'
import { rotate180 } from '../../image/crop.js'
import { decodeRgbImage } from '../../image/image.js'
import { resizeRgbImageOpenCv } from '../../image/opencv.js'
import { toRapidOCRImage } from '../output/compat.js'
import { addArrayMetadata, addStageVisualization } from '../output/metadata.js'
import { renderLineStage } from '../output/visualization.js'

const IMAGE_HEIGHT = 48
const IMAGE_WIDTH = 192
const CHANNELS = 3
const CLASSIFICATION_BATCH_SIZE = 6
const DEFAULT_ROTATE_THRESHOLD = 0.9
const DEFAULT_LABEL_LIST = ['0', '180'] as const

export interface TextOrientationResult {
  angle: 0 | 180
  confidence: number
}

export interface TextClassificationOutput extends Array<TextOrientationResult>, RapidOCRStageVisualization {
  imgList: RawRgbImage[]
  img_list: RapidOCRImage[]
  clsRes: Array<[string, number]>
  cls_res: Array<[string, number]>
  elapse: number
}

export interface TextClassifierSessionOptions {
  /** Override the minimum classification confidence required to rotate a line. */
  rotateThreshold?: number
  /** Maximum number of lines sent to the model in one inference call. */
  batchSize?: number
  /** Model input channels, height, and width. */
  imageShape?: readonly [3, number, number]
  /** Two labels mapped to the classifier model's output classes. */
  labelList?: readonly [string, string]
}

export class TextClassifier {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly ownsSession: boolean,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedBatchSize: number | undefined,
    private readonly fixedHeight: number | undefined,
    private readonly fixedWidth: number | undefined,
    private readonly rotateThreshold: number,
    private readonly batchSize: number,
    private readonly imageHeight: number,
    private readonly imageWidth: number,
    private readonly labelList: readonly [string, string],
  ) {}

  static async create(
    modelPath: string,
    intraOpNumThreads?: number,
    executionProviders: readonly ExecutionProviderConfig[] = ['cpu'],
    interOpNumThreads?: number,
    rotateThreshold = DEFAULT_ROTATE_THRESHOLD,
    batchSize = CLASSIFICATION_BATCH_SIZE,
    imageShape?: readonly [3, number, number],
    enableCpuMemArena?: boolean,
    onExecutionProviderFallback?: (message: string) => void,
    labelList: readonly [string, string] = DEFAULT_LABEL_LIST,
  ): Promise<TextClassifier> {
    const [channels, configuredHeight, configuredWidth] = imageShape ?? [CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError('batchSize must be a positive integer.')
    }
    if (channels !== CHANNELS || !Number.isInteger(configuredHeight) || configuredHeight < 1 || !Number.isInteger(configuredWidth) || configuredWidth < 1) {
      throw new RangeError('imageShape must contain three values: 3, a positive height, and a positive width.')
    }
    if (intraOpNumThreads !== undefined && (!Number.isInteger(intraOpNumThreads) || intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }
    if (interOpNumThreads !== undefined && (!Number.isInteger(interOpNumThreads) || interOpNumThreads < 1)) {
      throw new RangeError('interOpNumThreads must be a positive integer.')
    }
    if (executionProviders.length === 0) {
      throw new RangeError('executionProviders must contain at least one provider.')
    }
    validateRotateThreshold(rotateThreshold)
    const normalizedLabelList = validateLabelList(labelList)
    const session = await createInferenceSession(modelPath, {
      executionProviders,
      ...(intraOpNumThreads === undefined ? {} : { intraOpNumThreads }),
      ...(interOpNumThreads === undefined ? {} : { interOpNumThreads }),
      ...(enableCpuMemArena === undefined ? {} : { enableCpuMemArena }),
      ...(onExecutionProviderFallback === undefined ? {} : { onExecutionProviderFallback }),
    })

    return TextClassifier.createFromSession(
      session,
      true,
      rotateThreshold,
      batchSize,
      configuredHeight,
      configuredWidth,
      imageShape !== undefined,
      false,
      normalizedLabelList,
    )
  }

  /** Wraps a caller-created ONNX Runtime session without taking ownership of it. */
  static async createWithSession(
    session: ort.InferenceSession,
    options: TextClassifierSessionOptions = {},
  ): Promise<TextClassifier> {
    const batchSize = options.batchSize ?? CLASSIFICATION_BATCH_SIZE
    const [channels, imageHeight, imageWidth] = options.imageShape ?? [CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError('batchSize must be a positive integer.')
    }
    if (channels !== CHANNELS || !Number.isInteger(imageHeight) || imageHeight < 1 || !Number.isInteger(imageWidth) || imageWidth < 1) {
      throw new RangeError('imageShape must contain three values: 3, a positive height, and a positive width.')
    }
    const rotateThreshold = options.rotateThreshold ?? DEFAULT_ROTATE_THRESHOLD
    validateRotateThreshold(rotateThreshold)
    const labelList = validateLabelList(options.labelList ?? DEFAULT_LABEL_LIST)
    return TextClassifier.createFromSession(
      session,
      false,
      rotateThreshold,
      batchSize,
      imageHeight,
      imageWidth,
      options.imageShape !== undefined,
      false,
      labelList,
    )
  }

  private static async createFromSession(
    session: ort.InferenceSession,
    ownsSession: boolean,
    rotateThreshold: number,
    batchSize: number,
    configuredHeight: number,
    configuredWidth: number,
    validateConfiguredShape: boolean,
    allowDynamicBatch: boolean,
    labelList: readonly [string, string],
  ): Promise<TextClassifier> {
    const channels = CHANNELS
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
        || input.shape[1] !== channels
        || (validateConfiguredShape && typeof input.shape[2] === 'number' && input.shape[2] !== configuredHeight)
        || (validateConfiguredShape && typeof input.shape[3] === 'number' && input.shape[3] !== configuredWidth)
        || (typeof input.shape[2] === 'number' && input.shape[2] < 1)
        || (typeof input.shape[3] === 'number' && input.shape[3] < 1)
        || output.shape.length !== 2
        || (typeof output.shape[1] === 'number' && output.shape[1] !== 2)) {
        throw new Error('Classifier model must have one float32 [batch, 3, height, width] input and one float32 output.')
      }

      return new TextClassifier(
        session,
        ownsSession,
        inputName,
        outputName,
        allowDynamicBatch ? undefined : typeof input.shape[0] === 'number' ? input.shape[0] : undefined,
        typeof input.shape[2] === 'number' ? input.shape[2] : undefined,
        typeof input.shape[3] === 'number' ? input.shape[3] : undefined,
        rotateThreshold,
        batchSize,
        typeof input.shape[2] === 'number' ? input.shape[2] : configuredHeight,
        typeof input.shape[3] === 'number' ? input.shape[3] : configuredWidth,
        labelList,
      )
    }
    catch (error) {
      if (ownsSession)
        await session.release()
      throw error
    }
  }

  /** Classifies already-cropped text lines and returns their predicted orientation. */
  async classifyLines(inputs: readonly ImageInput[]): Promise<TextClassificationOutput> {
    const images = await Promise.all(inputs.map(input => decodeRgbImage(input)))
    const started = performance.now()
    const { images: rotatedImages, results, clsRes } = await this.classifyAndRotateDetailed(images)
    const output = addArrayMetadata(results, {
      clsRes,
      cls_res: clsRes,
      elapse: (performance.now() - started) / 1000,
      imgList: rotatedImages,
      img_list: rotatedImages.map(toRapidOCRImage),
    }) as TextClassificationOutput
    return addStageVisualization(output, savePath => renderLineStage(
      rotatedImages,
      clsRes.map(([label]) => label),
      clsRes.map(([, confidence]) => confidence),
      {},
      savePath,
    ))
  }

  async classifyAndRotate(images: readonly DecodedRgbImage[]): Promise<DecodedRgbImage[]> {
    return (await this.classifyAndRotateDetailed(images)).images
  }

  async classifyAndRotateDetailed(images: readonly DecodedRgbImage[]): Promise<{
    clsRes: Array<[string, number]>
    images: DecodedRgbImage[]
    results: TextOrientationResult[]
  }> {
    const { classIndices, results } = await this.classifyImages(images)
    const clsRes = classIndices.map((classIndex, index) => [this.labelList[classIndex]!, results[index]!.confidence] as [string, number])
    const rotatedImages = images.map((image, index) => {
      const [label, confidence] = clsRes[index] ?? ['', 0]
      return label.includes('180') && confidence > this.rotateThreshold
        ? rotate180(image)
        : image
    })
    return { clsRes, images: rotatedImages, results }
  }

  async close(): Promise<void> {
    if (this.ownsSession)
      await this.session.release()
  }

  private async classifyImages(images: readonly DecodedRgbImage[]): Promise<{
    classIndices: number[]
    results: TextOrientationResult[]
  }> {
    const results = Array.from({ length: images.length }) as TextOrientationResult[]
    const classIndices = Array.from({ length: images.length }).fill(0) as number[]
    const capacity = this.fixedBatchSize ?? this.batchSize
    const ordered = images
      .map((image, index) => ({ image, index, ratio: image.width / image.height }))
      .sort((left, right) => left.ratio - right.ratio)

    for (let offset = 0; offset < ordered.length; offset += capacity) {
      const batch = ordered.slice(offset, offset + capacity)
      const { data, valuesPerItem } = await this.classifyBatch(batch)
      for (let index = 0; index < batch.length; index++) {
        const item = batch[index]!
        const prediction = getBestClass(data, index, valuesPerItem)
        const label = this.labelList[prediction.index]!
        results[item.index] = {
          angle: label.includes('180') ? 180 : 0,
          confidence: prediction.score,
        }
        classIndices[item.index] = prediction.index
      }
    }

    return { classIndices, results }
  }

  private async classifyBatch(batch: readonly { image: DecodedRgbImage, index: number, ratio: number }[]): Promise<{ data: Float32Array, valuesPerItem: number }> {
    const height = this.fixedHeight ?? this.imageHeight
    const width = this.fixedWidth ?? this.imageWidth
    const batchSize = this.fixedBatchSize ?? batch.length
    const planeSize = height * width
    const inputData = new Float32Array(batchSize * CHANNELS * planeSize)
    for (let index = 0; index < batch.length; index++) {
      const image = batch[index]!.image
      const resizedWidth = Math.min(Math.ceil(height * image.width / image.height), width)
      const pixels = await resizeRgbImageOpenCv(image, resizedWidth, height)
      writeNormalizedPixels(inputData, pixels, index, height, resizedWidth, width, planeSize)
    }

    const tensor = new ort.Tensor('float32', inputData, [batchSize, CHANNELS, height, width])
    const outputs = await this.session.run({ [this.inputName]: tensor })
    const output = outputs[this.outputName]
    if (!(output instanceof ort.Tensor) || output.type !== 'float32' || !(output.data instanceof Float32Array)) {
      throw new TypeError('Classifier model output must be a float32 tensor in CPU memory.')
    }
    const valuesPerItem = output.data.length / batchSize
    if (valuesPerItem !== 2) {
      throw new Error(`Classifier model output must contain two orientation scores per image; received ${valuesPerItem}.`)
    }
    return { data: output.data, valuesPerItem }
  }
}

function validateLabelList(value: readonly [string, string]): readonly [string, string] {
  if (!Array.isArray(value) || value.length !== 2 || value.some(label => typeof label !== 'string')) {
    throw new TypeError('labelList must contain exactly two strings.')
  }
  return [value[0], value[1]]
}

function writeNormalizedPixels(
  target: Float32Array,
  pixels: Uint8Array,
  batchIndex: number,
  height: number,
  resizedWidth: number,
  width: number,
  planeSize: number,
): void {
  const imageOffset = batchIndex * CHANNELS * planeSize
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < resizedWidth; x++) {
      const pixelOffset = (y * resizedWidth + x) * CHANNELS
      const targetOffset = y * width + x
      target[imageOffset + targetOffset] = normalize(pixels[pixelOffset + 2])
      target[imageOffset + planeSize + targetOffset] = normalize(pixels[pixelOffset + 1])
      target[imageOffset + planeSize * 2 + targetOffset] = normalize(pixels[pixelOffset])
    }
  }
}

function getBestClass(data: Float32Array, index: number, valuesPerItem: number): { index: number, score: number } {
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
  return { index: bestClass, score: bestScore }
}

function normalize(pixel: number | undefined): number {
  if (pixel === undefined)
    throw new Error('Image decoder returned an incomplete RGB buffer.')
  return pixel / 127.5 - 1
}

function validateRotateThreshold(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError('classifierThreshold must be between 0 and 1.')
  }
}
