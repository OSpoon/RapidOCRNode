/*
 * TypeScript adaptation of the RapidOCR v3.10.0 DB detector pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { ExecutionProviderConfig, ImageInput, RapidOCRBox, RapidOCRImage, RapidOCRStageVisualization, TextBox } from '../../shared/types.js'
import type { DetectorBox, DetectorPostprocessOptions } from './postprocess.js'
import { performance } from 'node:perf_hooks'
import * as ort from 'onnxruntime-node'
import { createInferenceSession } from '../../backends/inference.js'
import { decodeRgbImage, roundToEven } from '../../image/image.js'
import { resizeRgbImageOpenCv } from '../../image/opencv.js'
import { toRapidOCRBoxes, toRapidOCRImage } from '../output/compat.js'
import { addArrayMetadata, addStageVisualization } from '../output/metadata.js'
import { renderDetectionStage } from '../output/visualization.js'
import { DEFAULT_DETECTOR_SIDE, extractBoxes, getDetectorResizeRatio, validateDetectorPostprocessOptions } from './postprocess.js'

export { extractBoxes, getDetectorResizeRatio, sortBoxesInReadingOrder, validateDetectorPostprocessOptions } from './postprocess.js'
export type { DetectorBox, DetectorPostprocessOptions } from './postprocess.js'

const DETECTOR_MEAN = [0.5, 0.5, 0.5] as const
const DETECTOR_STD = [0.5, 0.5, 0.5] as const
export interface TextDetectionOutput extends Array<DetectorBox>, RapidOCRStageVisualization {
  img: RapidOCRImage
  boxes: RapidOCRBox[]
  boxPoints: TextBox[]
  scores: number[]
  elapse: number
}

interface ProbabilityMap {
  data: Float32Array
  height: number
  width: number
}

function getProbabilityMap(data: Float32Array, dims: readonly number[]): ProbabilityMap {
  if (dims.length === 4 && dims[0] === 1 && dims[1] === 1) {
    const height = dims[2]
    const width = dims[3]
    if (height && width && data.length === height * width) {
      return { data, height, width }
    }
  }
  if (dims.length === 3 && dims[0] === 1) {
    const height = dims[1]
    const width = dims[2]
    if (height && width && data.length === height * width) {
      return { data, height, width }
    }
  }
  throw new Error(`Detection model output must have shape [1, 1, height, width] or [1, height, width]; received [${dims.join(', ')}].`)
}

export class TextDetector {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly ownsSession: boolean,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedHeight: number | undefined,
    private readonly fixedWidth: number | undefined,
    private readonly postprocessOptions: DetectorPostprocessOptions,
  ) {}

  static async create(
    modelPath: string,
    intraOpNumThreads?: number,
    postprocessOptions: DetectorPostprocessOptions = {},
    executionProviders: readonly ExecutionProviderConfig[] = ['cpu'],
    interOpNumThreads?: number,
    enableCpuMemArena?: boolean,
    onExecutionProviderFallback?: (message: string) => void,
  ): Promise<TextDetector> {
    validateDetectorPostprocessOptions(postprocessOptions)
    if (intraOpNumThreads !== undefined && (!Number.isInteger(intraOpNumThreads) || intraOpNumThreads < 1)) {
      throw new RangeError('intraOpNumThreads must be a positive integer.')
    }
    if (interOpNumThreads !== undefined && (!Number.isInteger(interOpNumThreads) || interOpNumThreads < 1)) {
      throw new RangeError('interOpNumThreads must be a positive integer.')
    }
    if (executionProviders.length === 0) {
      throw new RangeError('executionProviders must contain at least one provider.')
    }
    const session = await createInferenceSession(modelPath, {
      executionProviders,
      ...(intraOpNumThreads === undefined ? {} : { intraOpNumThreads }),
      ...(interOpNumThreads === undefined ? {} : { interOpNumThreads }),
      ...(enableCpuMemArena === undefined ? {} : { enableCpuMemArena }),
      ...(onExecutionProviderFallback === undefined ? {} : { onExecutionProviderFallback }),
    })

    return TextDetector.createFromSession(session, true, postprocessOptions)
  }

  /** Wraps a caller-created ONNX Runtime session without taking ownership of it. */
  static async createWithSession(
    session: ort.InferenceSession,
    postprocessOptions: DetectorPostprocessOptions = {},
  ): Promise<TextDetector> {
    return TextDetector.createFromSession(session, false, postprocessOptions)
  }

  private static async createFromSession(
    session: ort.InferenceSession,
    ownsSession: boolean,
    postprocessOptions: DetectorPostprocessOptions,
  ): Promise<TextDetector> {
    validateDetectorPostprocessOptions(postprocessOptions)
    try {
      const input = session.inputMetadata[0]
      const output = session.outputMetadata[0]
      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      if (session.inputMetadata.length !== 1
        || session.outputMetadata.length < 1
        || !input?.isTensor
        || !output?.isTensor
        || !inputName
        || !outputName
        || input.type !== 'float32'
        || output.type !== 'float32'
        || input.shape.length !== 4
        || input.shape[1] !== 3) {
        throw new Error('Detection model must have one float32 [1, 3, height, width] input and at least one float32 output.')
      }

      return new TextDetector(
        session,
        ownsSession,
        inputName,
        outputName,
        typeof input.shape[2] === 'number' ? input.shape[2] : undefined,
        typeof input.shape[3] === 'number' ? input.shape[3] : undefined,
        postprocessOptions,
      )
    }
    catch (error) {
      if (ownsSession)
        await session.release()
      throw error
    }
  }

  async detect(input: ImageInput, overrides: DetectorPostprocessOptions = {}): Promise<TextDetectionOutput> {
    validateDetectorPostprocessOptions(overrides)
    const image = await decodeRgbImage(input)
    const started = performance.now()
    const activeOptions = { ...this.postprocessOptions, ...overrides }
    const ratio = this.getResizeRatio(image.width, image.height, activeOptions)
    const height = this.fixedHeight ?? roundTo32(Math.trunc(image.height * ratio))
    const width = this.fixedWidth ?? roundTo32(Math.trunc(image.width * ratio))
    const resized = await resizeRgbImageOpenCv(image, width, height)
    const planeSize = width * height
    const tensorData = new Float32Array(3 * planeSize)

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const sourceOffset = (y * width + x) * 3
        const targetOffset = y * width + x
        tensorData[targetOffset] = normalize(resized[sourceOffset + 2], activeOptions.mean?.[0] ?? DETECTOR_MEAN[0], activeOptions.std?.[0] ?? DETECTOR_STD[0])
        tensorData[planeSize + targetOffset] = normalize(resized[sourceOffset + 1], activeOptions.mean?.[1] ?? DETECTOR_MEAN[1], activeOptions.std?.[1] ?? DETECTOR_STD[1])
        tensorData[planeSize * 2 + targetOffset] = normalize(resized[sourceOffset], activeOptions.mean?.[2] ?? DETECTOR_MEAN[2], activeOptions.std?.[2] ?? DETECTOR_STD[2])
      }
    }

    const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, height, width])
    const outputs = await this.session.run({ [this.inputName]: inputTensor })
    const output = outputs[this.outputName]
    if (!(output instanceof ort.Tensor) || output.type !== 'float32' || !(output.data instanceof Float32Array)) {
      throw new TypeError('Detection model output must be a float32 tensor in CPU memory.')
    }

    const map = getProbabilityMap(output.data, output.dims)
    const detections = await extractBoxes(map.data, map.width, map.height, image.width, image.height, activeOptions)
    const outputBoxes = detections.map(detection => detection.box)
    const scores = detections.map(detection => detection.score)
    const result = addArrayMetadata(detections, {
      boxes: toRapidOCRBoxes(outputBoxes),
      boxPoints: outputBoxes,
      elapse: (performance.now() - started) / 1000,
      img: toRapidOCRImage(image),
      scores,
    }) as TextDetectionOutput
    return addStageVisualization(result, savePath => renderDetectionStage(image, outputBoxes, scores, savePath))
  }

  async close(): Promise<void> {
    if (this.ownsSession)
      await this.session.release()
  }

  private getResizeRatio(width: number, height: number, options: DetectorPostprocessOptions): number {
    if (this.fixedWidth && this.fixedHeight) {
      return 1
    }
    return getDetectorResizeRatio(width, height, options.limitSideLength ?? DEFAULT_DETECTOR_SIDE, options.limitType ?? 'min')
  }
}

function roundTo32(value: number): number {
  return Math.max(32, roundToEven(value / 32) * 32)
}

function normalize(pixel: number | undefined, mean: number, std: number): number {
  if (pixel === undefined)
    throw new Error('Image decoder returned an incomplete RGB buffer.')
  return (pixel / 255 - mean) / std
}
