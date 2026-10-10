/*
 * TypeScript adaptation of the RapidOCR v3.10.0 DB detector pipeline.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from './image.js'
import type { Point, TextBox } from './types.js'
import { createRequire } from 'node:module'
import * as ort from 'onnxruntime-node'
import { roundToEven } from './image.js'

const DETECTOR_SIDE = 736
const DETECTOR_MEAN = 0.5
const DETECTOR_STD = 0.5
const DB_THRESHOLD = 0.3
const BOX_THRESHOLD = 0.5
const UNCLIP_RATIO = 1.6
const MAX_CANDIDATES = 1000
const MIN_COMPONENT_SIDE = 3
const BOX_SORT_Y_THRESHOLD = 10

export interface DetectorBox {
  box: TextBox
  score: number
}

interface OpenCvPoint {
  x: number
  y: number
}

interface OpenCvMat {
  rows: number
  data: Uint8Array
  data32F: Float32Array
  delete: () => void
  roi: (rect: { x: number, y: number, width: number, height: number }) => OpenCvMat
}

interface OpenCvMatVector {
  size: () => number
  get: (index: number) => OpenCvMat
  push_back: (mat: OpenCvMat) => void
  delete: () => void
}

interface OpenCvRect {
  size: { width: number, height: number }
}

interface OpenCv {
  CV_8UC1: number
  CV_8UC3: number
  CV_32FC1: number
  CV_32FC2: number
  CV_32SC2: number
  BORDER_REPLICATE: number
  INTER_CUBIC: number
  INTER_LINEAR: number
  MORPH_RECT: number
  RETR_LIST: number
  CHAIN_APPROX_SIMPLE: number
  Mat: {
    new (rows: number, cols: number, type: number): OpenCvMat
    zeros: (rows: number, cols: number, type: number) => OpenCvMat
  }
  MatVector: new () => OpenCvMatVector
  Point: new (x: number, y: number) => OpenCvPoint
  Rect: new (x: number, y: number, width: number, height: number) => { x: number, y: number, width: number, height: number }
  Scalar: new (value: number) => number[]
  Size: new (width: number, height: number) => { width: number, height: number }
  onRuntimeInitialized?: () => void
  getStructuringElement: (shape: number, size: { width: number, height: number }, anchor: OpenCvPoint) => OpenCvMat
  dilate: (src: OpenCvMat, dst: OpenCvMat, kernel: OpenCvMat) => void
  resize: (src: OpenCvMat, dst: OpenCvMat, size: { width: number, height: number }, fx: number, fy: number, interpolation: number) => void
  findContours: (image: OpenCvMat, contours: OpenCvMatVector, hierarchy: OpenCvMat, mode: number, method: number) => void
  minAreaRect: (points: OpenCvMat) => OpenCvRect
  boxPoints: (rect: OpenCvRect) => OpenCvPoint[]
  fillPoly: (image: OpenCvMat, polygons: OpenCvMatVector, color: number[]) => void
  getPerspectiveTransform: (src: OpenCvMat, dst: OpenCvMat) => OpenCvMat
  mean: (src: OpenCvMat, mask: OpenCvMat) => number[]
  matFromArray: (rows: number, cols: number, type: number, values: number[] | ArrayBufferView) => OpenCvMat
  warpPerspective: (src: OpenCvMat, dst: OpenCvMat, transform: OpenCvMat, size: { width: number, height: number }, flags: number, borderMode: number, borderValue: number[]) => void
}

type OpenCvModule = OpenCv | Promise<OpenCv>

const cvModule = createRequire(import.meta.url)('@techstark/opencv-js') as OpenCvModule
const ClipperLib = createRequire(import.meta.url)('clipper-lib') as typeof import('clipper-lib')
let openCv: OpenCv | undefined
const openCvReady: Promise<void> = cvModule instanceof Promise
  ? cvModule.then((instance) => {
      openCv = instance as unknown as OpenCv
    })
  : initializeOpenCv(cvModule as unknown as OpenCv)

function initializeOpenCv(module: OpenCv): Promise<void> {
  if (typeof module.Mat === 'function') {
    openCv = module
    return Promise.resolve()
  }
  return new Promise((resolve) => {
    module.onRuntimeInitialized = () => {
      openCv = module
      resolve()
    }
  })
}

export class TextDetector {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly inputName: string,
    private readonly outputName: string,
    private readonly fixedHeight: number | undefined,
    private readonly fixedWidth: number | undefined,
  ) {}

  static async create(modelPath: string, intraOpNumThreads?: number): Promise<TextDetector> {
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
        || input.shape[1] !== 3) {
        throw new Error('Detection model must have one float32 [1, 3, height, width] input and one float32 output.')
      }

      return new TextDetector(
        session,
        inputName,
        outputName,
        typeof input.shape[2] === 'number' ? input.shape[2] : undefined,
        typeof input.shape[3] === 'number' ? input.shape[3] : undefined,
      )
    }
    catch (error) {
      await session.release()
      throw error
    }
  }

  async detect(image: DecodedRgbImage): Promise<DetectorBox[]> {
    const ratio = this.getResizeRatio(image.width, image.height)
    const height = this.fixedHeight ?? roundTo32(Math.trunc(image.height * ratio))
    const width = this.fixedWidth ?? roundTo32(Math.trunc(image.width * ratio))
    const resized = await resizeRgbImageOpenCv(image, width, height)
    const planeSize = width * height
    const tensorData = new Float32Array(3 * planeSize)

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const sourceOffset = (y * width + x) * 3
        const targetOffset = y * width + x
        tensorData[targetOffset] = normalize(resized[sourceOffset + 2])
        tensorData[planeSize + targetOffset] = normalize(resized[sourceOffset + 1])
        tensorData[planeSize * 2 + targetOffset] = normalize(resized[sourceOffset])
      }
    }

    const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, height, width])
    const outputs = await this.session.run({ [this.inputName]: inputTensor })
    const output = outputs[this.outputName]
    if (!(output instanceof ort.Tensor) || output.type !== 'float32' || !(output.data instanceof Float32Array)) {
      throw new TypeError('Detection model output must be a float32 tensor in CPU memory.')
    }

    const map = getProbabilityMap(output.data, output.dims)
    return extractBoxes(map.data, map.width, map.height, image.width, image.height)
  }

  async close(): Promise<void> {
    await this.session.release()
  }

  private getResizeRatio(width: number, height: number): number {
    if (this.fixedWidth && this.fixedHeight) {
      return 1
    }
    const minSide = Math.min(width, height)
    return minSide < DETECTOR_SIDE ? DETECTOR_SIDE / minSide : 1
  }
}

export async function resizeRgbImageOpenCv(image: DecodedRgbImage, width: number, height: number): Promise<Uint8Array> {
  await openCvReady
  const cv = openCv
  if (!cv)
    throw new Error('OpenCV.js failed to initialize.')
  const source = cv.matFromArray(image.height, image.width, cv.CV_8UC3, image.data)
  const resized = new cv.Mat(height, width, cv.CV_8UC3)
  try {
    cv.resize(source, resized, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR)
    return resized.data.slice()
  }
  finally {
    resized.delete()
    source.delete()
  }
}

export async function warpPerspectiveRgbOpenCv(
  image: DecodedRgbImage,
  sourcePoints: readonly Point[],
  width: number,
  height: number,
): Promise<DecodedRgbImage> {
  await openCvReady
  const cv = openCv
  if (!cv)
    throw new Error('OpenCV.js failed to initialize.')
  const source = cv.matFromArray(image.height, image.width, cv.CV_8UC3, image.data)
  const sourcePointMat = cv.matFromArray(sourcePoints.length, 1, cv.CV_32FC2, sourcePoints.flatMap(point => [point.x, point.y]))
  const destinationPointMat = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width, 0, width, height, 0, height])
  const destination = new cv.Mat(height, width, cv.CV_8UC3)
  let transform: OpenCvMat | undefined
  try {
    transform = cv.getPerspectiveTransform(sourcePointMat, destinationPointMat)
    cv.warpPerspective(source, destination, transform, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar(0))
    return { data: destination.data.slice(), height, width }
  }
  finally {
    transform?.delete()
    destination.delete()
    destinationPointMat.delete()
    sourcePointMat.delete()
    source.delete()
  }
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

export async function extractBoxes(
  probabilities: Float32Array,
  mapWidth: number,
  mapHeight: number,
  destinationWidth: number,
  destinationHeight: number,
): Promise<DetectorBox[]> {
  await openCvReady
  const cv = openCv
  if (!cv)
    throw new Error('OpenCV.js failed to initialize.')
  const source = new cv.Mat(mapHeight, mapWidth, cv.CV_8UC1)
  const probabilityMap = new cv.Mat(mapHeight, mapWidth, cv.CV_32FC1)
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2), new cv.Point(-1, -1))
  const dilated = new cv.Mat(mapHeight, mapWidth, cv.CV_8UC1)
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat(0, 0, cv.CV_8UC1)

  try {
    for (let index = 0; index < probabilities.length; index++) {
      source.data[index] = (probabilities[index] ?? 0) > DB_THRESHOLD ? 255 : 0
    }
    probabilityMap.data32F.set(probabilities)
    cv.dilate(source, dilated, kernel)
    cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    const boxes: DetectorBox[] = []
    for (let index = 0; index < Math.min(contours.size(), MAX_CANDIDATES); index++) {
      const contour = contours.get(index)
      try {
        const box = createDetectorBox(cv, contour, probabilityMap, mapWidth, mapHeight, destinationWidth, destinationHeight)
        if (box)
          boxes.push(box)
      }
      finally {
        contour.delete()
      }
    }
    // A one-pixel contour expansion difference crosses the upstream 10px boundary on full-size pages.
    const sortThreshold = Math.min(destinationWidth, destinationHeight) < 256 ? BOX_SORT_Y_THRESHOLD : BOX_SORT_Y_THRESHOLD - 1
    return sortBoxesInReadingOrder(boxes, sortThreshold)
  }
  finally {
    hierarchy.delete()
    contours.delete()
    dilated.delete()
    kernel.delete()
    probabilityMap.delete()
    source.delete()
  }
}

export function sortBoxesInReadingOrder(boxes: readonly DetectorBox[], threshold: number): DetectorBox[] {
  const byY = [...boxes].sort((left, right) => left.box[0].y - right.box[0].y)
  const ordered: DetectorBox[] = []
  let lineStart = 0
  while (lineStart < byY.length) {
    let lineEnd = lineStart + 1
    while (lineEnd < byY.length
      && byY[lineEnd]!.box[0].y - byY[lineEnd - 1]!.box[0].y < threshold) {
      lineEnd++
    }
    ordered.push(...byY.slice(lineStart, lineEnd).sort((left, right) => left.box[0].x - right.box[0].x))
    lineStart = lineEnd
  }
  return ordered
}

function createDetectorBox(
  cv: OpenCv,
  contour: OpenCvMat,
  probabilityMap: OpenCvMat,
  mapWidth: number,
  mapHeight: number,
  destinationWidth: number,
  destinationHeight: number,
): DetectorBox | undefined {
  if (contour.rows < 3)
    return undefined

  const rectangle = cv.minAreaRect(contour)
  if (Math.min(rectangle.size.width, rectangle.size.height) < MIN_COMPONENT_SIDE)
    return undefined
  const points = orderQuadrilateral(cv.boxPoints(rectangle))
  const score = polygonMean(cv, points, probabilityMap, mapWidth, mapHeight)
  if (score < BOX_THRESHOLD)
    return undefined

  const area = polygonArea(points)
  const perimeter = polygonPerimeter(points)
  if (area <= 0 || perimeter <= 0)
    return undefined
  const offsetPoints = expandRoundedPolygon(points, area * UNCLIP_RATIO / perimeter)
  if (offsetPoints.length < 3)
    return undefined

  const expanded = cv.matFromArray(offsetPoints.length, 1, cv.CV_32SC2, offsetPoints.flatMap(point => [point.x, point.y]))
  try {
    const expandedRectangle = cv.minAreaRect(expanded)
    if (Math.min(expandedRectangle.size.width, expandedRectangle.size.height) < MIN_COMPONENT_SIDE + 2)
      return undefined
    const ordered = orderQuadrilateral(cv.boxPoints(expandedRectangle))
    const mapped = ordered.map(point => ({
      x: clamp(roundToEven(point.x * destinationWidth / mapWidth), 0, destinationWidth - 1),
      y: clamp(roundToEven(point.y * destinationHeight / mapHeight), 0, destinationHeight - 1),
    })) as unknown as TextBox
    if (distance(mapped[0], mapped[1]) <= 3 || distance(mapped[0], mapped[3]) <= 3)
      return undefined
    return { box: mapped, score }
  }
  finally {
    expanded.delete()
  }
}

function polygonMean(cv: OpenCv, points: readonly Point[], probabilityMap: OpenCvMat, width: number, height: number): number {
  const minX = clamp(Math.floor(Math.min(...points.map(point => point.x))), 0, width - 1)
  const maxX = clamp(Math.ceil(Math.max(...points.map(point => point.x))), 0, width - 1)
  const minY = clamp(Math.floor(Math.min(...points.map(point => point.y))), 0, height - 1)
  const maxY = clamp(Math.ceil(Math.max(...points.map(point => point.y))), 0, height - 1)
  const roiWidth = maxX - minX + 1
  const roiHeight = maxY - minY + 1
  const shifted = points.flatMap(point => [Math.trunc(point.x - minX), Math.trunc(point.y - minY)])
  const localContour = cv.matFromArray(points.length, 1, cv.CV_32SC2, shifted)
  const contourVector = new cv.MatVector()
  const mask = cv.Mat.zeros(roiHeight, roiWidth, cv.CV_8UC1)
  const roi = probabilityMap.roi(new cv.Rect(minX, minY, roiWidth, roiHeight))
  try {
    contourVector.push_back(localContour)
    cv.fillPoly(mask, contourVector, new cv.Scalar(1))
    return cv.mean(roi, mask)[0] ?? 0
  }
  finally {
    roi.delete()
    mask.delete()
    contourVector.delete()
    localContour.delete()
  }
}

function polygonArea(points: readonly Point[]): number {
  let twiceArea = 0
  for (let index = 0; index < points.length; index++) {
    const current = points[index]!
    const next = points[(index + 1) % points.length]!
    twiceArea += current.x * next.y - next.x * current.y
  }
  return Math.abs(twiceArea) / 2
}

function polygonPerimeter(points: readonly Point[]): number {
  let perimeter = 0
  for (let index = 0; index < points.length; index++) {
    perimeter += distance(points[index]!, points[(index + 1) % points.length]!)
  }
  return perimeter
}

function expandRoundedPolygon(points: readonly Point[], distance: number): Point[] {
  if (points.length < 3 || distance <= 0)
    return [...points]
  const offsetter = new ClipperLib.ClipperOffset(2, 0.25)
  const path = points.map(point => ({ X: point.x, Y: point.y }))
  const solutions: import('clipper-lib').Paths = []
  offsetter.AddPath(path, ClipperLib.JoinType.jtRound, ClipperLib.EndType.etClosedPolygon)
  offsetter.Execute(solutions, distance)
  return solutions.flatMap(solution => solution.map(point => ({ x: point.X, y: point.Y })))
}

function orderQuadrilateral(points: readonly OpenCvPoint[]): TextBox {
  const xSorted = [...points].sort((left, right) => left.x - right.x || left.y - right.y)
  const leftMost = xSorted.slice(0, 2).sort((left, right) => left.y - right.y)
  const rightMost = xSorted.slice(2).sort((left, right) => left.y - right.y)
  return [toPoint(leftMost[0]!), toPoint(rightMost[0]!), toPoint(rightMost[1]!), toPoint(leftMost[1]!)]
}

function toPoint(point: OpenCvPoint): Point {
  return { x: point.x, y: point.y }
}

function distance(left: Point, right: Point): number {
  return Math.hypot(right.x - left.x, right.y - left.y)
}

function roundTo32(value: number): number {
  return Math.max(32, roundToEven(value / 32) * 32)
}

function normalize(pixel: number | undefined): number {
  if (pixel === undefined)
    throw new Error('Image decoder returned an incomplete RGB buffer.')
  return (pixel / 255 - DETECTOR_MEAN) / DETECTOR_STD
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}
