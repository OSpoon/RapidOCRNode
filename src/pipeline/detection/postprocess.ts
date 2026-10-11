import type { OpenCv, OpenCvMat, OpenCvPoint } from '../../image/opencv.js'
import type { Point, TextBox } from '../../shared/types.js'
import { createRequire } from 'node:module'
import { roundToEven } from '../../image/image.js'
import { getOpenCv } from '../../image/opencv.js'

/*
 * DB detector probability-map postprocessing adapted from RapidOCR v3.10.0.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */

export const DEFAULT_DETECTOR_SIDE = 736
const DB_THRESHOLD = 0.3
const BOX_THRESHOLD = 0.5
const UNCLIP_RATIO = 1.6
const MAX_CANDIDATES = 1000
const MIN_COMPONENT_SIDE = 3
const BOX_SORT_Y_THRESHOLD = 10

export interface DetectorPostprocessOptions {
  /** DB probability-map threshold. Defaults to 0.3. */
  threshold?: number
  /** Minimum mean text probability inside a detected region. Defaults to 0.5. */
  boxThreshold?: number
  /** DB polygon expansion ratio. Defaults to 1.6. */
  unclipRatio?: number
  /** Maximum number of contours to postprocess. Defaults to 1000. */
  maxCandidates?: number
  /** Dilate the binary probability map before contour extraction. Defaults to true. */
  useDilation?: boolean
  /** Use upstream fast rectangle or slow contour-region mean scoring. Defaults to fast. */
  scoreMode?: 'fast' | 'slow'
  /** Resize limit for dynamic input models. Defaults to 736. */
  limitSideLength?: number
  /** Resize the short side up to the limit or the long side down to the limit. Defaults to min. */
  limitType?: 'min' | 'max'
  /** RGB/BGR model normalization mean, in detector tensor channel order. */
  mean?: readonly [number, number, number]
  /** RGB/BGR model normalization standard deviation, in detector tensor channel order. */
  std?: readonly [number, number, number]
}

export interface DetectorBox {
  box: TextBox
  score: number
}

const ClipperLib = createRequire(import.meta.url)('clipper-lib') as typeof import('clipper-lib')

export function getDetectorResizeRatio(
  width: number,
  height: number,
  limitSideLength = DEFAULT_DETECTOR_SIDE,
  limitType: 'min' | 'max' = 'min',
): number {
  if (!Number.isFinite(limitSideLength) || limitSideLength < 1) {
    throw new RangeError('limitSideLength must be a positive finite number.')
  }
  if (limitType !== 'min' && limitType !== 'max') {
    throw new TypeError('limitType must be "min" or "max".')
  }
  if (limitType === 'min') {
    const minSide = Math.min(width, height)
    return minSide < limitSideLength ? limitSideLength / minSide : 1
  }
  const maxSide = Math.max(width, height)
  return maxSide > limitSideLength ? limitSideLength / maxSide : 1
}

export async function extractBoxes(
  probabilities: Float32Array,
  mapWidth: number,
  mapHeight: number,
  destinationWidth: number,
  destinationHeight: number,
  options: DetectorPostprocessOptions = {},
): Promise<DetectorBox[]> {
  validateDetectorPostprocessOptions(options)
  const threshold = options.threshold ?? DB_THRESHOLD
  const boxThreshold = options.boxThreshold ?? BOX_THRESHOLD
  const unclipRatio = options.unclipRatio ?? UNCLIP_RATIO
  const maxCandidates = options.maxCandidates ?? MAX_CANDIDATES
  const useDilation = options.useDilation ?? true
  const scoreMode = options.scoreMode ?? 'fast'

  const cv = await getOpenCv()
  const source = new cv.Mat(mapHeight, mapWidth, cv.CV_8UC1)
  const probabilityMap = new cv.Mat(mapHeight, mapWidth, cv.CV_32FC1)
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2), new cv.Point(-1, -1))
  const dilated = new cv.Mat(mapHeight, mapWidth, cv.CV_8UC1)
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat(0, 0, cv.CV_8UC1)

  try {
    for (let index = 0; index < probabilities.length; index++) {
      source.data[index] = (probabilities[index] ?? 0) > threshold ? 255 : 0
    }
    probabilityMap.data32F.set(probabilities)
    if (useDilation)
      cv.dilate(source, dilated, kernel)
    else
      dilated.data.set(source.data)
    cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    const boxes: DetectorBox[] = []
    for (let index = 0; index < Math.min(contours.size(), maxCandidates); index++) {
      const contour = contours.get(index)
      try {
        const box = createDetectorBox(
          cv,
          contour,
          probabilityMap,
          mapWidth,
          mapHeight,
          destinationWidth,
          destinationHeight,
          boxThreshold,
          unclipRatio,
          scoreMode,
        )
        if (box)
          boxes.push(box)
      }
      finally {
        contour.delete()
      }
    }
    return sortBoxesInReadingOrder(boxes, BOX_SORT_Y_THRESHOLD)
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
  boxThreshold: number,
  unclipRatio: number,
  scoreMode: 'fast' | 'slow',
): DetectorBox | undefined {
  if (contour.rows < 3)
    return undefined

  const rectangle = cv.minAreaRect(contour)
  if (Math.min(rectangle.size.width, rectangle.size.height) < MIN_COMPONENT_SIDE)
    return undefined
  const points = orderQuadrilateral(cv.boxPoints(rectangle))
  const scorePoints = scoreMode === 'slow' ? getContourPoints(contour) : points
  const score = polygonMean(cv, scorePoints, probabilityMap, mapWidth, mapHeight)
  if (score < boxThreshold)
    return undefined

  const area = polygonArea(points)
  const perimeter = polygonPerimeter(points)
  if (area <= 0 || perimeter <= 0)
    return undefined
  const offsetPoints = expandRoundedPolygon(points, area * unclipRatio / perimeter)
  if (offsetPoints.length < 3)
    return undefined

  const expanded = cv.matFromArray(offsetPoints.length, 1, cv.CV_32SC2, offsetPoints.flatMap(point => [point.x, point.y]))
  try {
    const expandedRectangle = cv.minAreaRect(expanded)
    if (Math.min(expandedRectangle.size.width, expandedRectangle.size.height) < MIN_COMPONENT_SIDE + 2)
      return undefined
    const ordered = orderQuadrilateral(cv.boxPoints(expandedRectangle))
    const mapped = ordered.map(point => ({
      x: mapDetectorCoordinate(point.x, mapWidth, destinationWidth),
      y: mapDetectorCoordinate(point.y, mapHeight, destinationHeight),
    })) as unknown as TextBox
    if (distance(mapped[0], mapped[1]) <= 3 || distance(mapped[0], mapped[3]) <= 3)
      return undefined
    return { box: mapped, score }
  }
  finally {
    expanded.delete()
  }
}

export function mapDetectorCoordinate(value: number, mapLength: number, destinationLength: number): number {
  // Upstream scales OpenCV float32 points through NumPy float32 division and
  // multiplication before applying ties-to-even rounding.
  const scaled = Math.fround(Math.fround(value / mapLength) * destinationLength)
  return clamp(roundToEven(scaled), 0, destinationLength - 1)
}

function getContourPoints(contour: OpenCvMat): Point[] {
  const coordinates = contour.data32S
  return Array.from({ length: contour.rows }, (_, index) => ({
    x: coordinates[index * 2] ?? 0,
    y: coordinates[index * 2 + 1] ?? 0,
  }))
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
  // Pyclipper converts OpenCV's float box points to integer coordinates by truncation.
  const path = points.map(point => ({ X: Math.trunc(point.x), Y: Math.trunc(point.y) }))
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

export function validateDetectorPostprocessOptions(options: DetectorPostprocessOptions): void {
  if (options.scoreMode !== undefined && options.scoreMode !== 'fast' && options.scoreMode !== 'slow') {
    throw new TypeError('scoreMode must be "fast" or "slow".')
  }
  for (const [name, value] of [
    ['threshold', options.threshold],
    ['boxThreshold', options.boxThreshold],
  ] as const) {
    if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 1)) {
      throw new RangeError(`${name} must be between 0 and 1.`)
    }
  }
  if (options.unclipRatio !== undefined && (!Number.isFinite(options.unclipRatio) || options.unclipRatio < 0)) {
    throw new RangeError('unclipRatio must be a finite non-negative number.')
  }
  if (options.maxCandidates !== undefined && (!Number.isInteger(options.maxCandidates) || options.maxCandidates < 1)) {
    throw new RangeError('maxCandidates must be a positive integer.')
  }
  if (options.limitSideLength !== undefined
    && (!Number.isFinite(options.limitSideLength) || options.limitSideLength < 1)) {
    throw new RangeError('limitSideLength must be a positive finite number.')
  }
  if (options.limitType !== undefined && options.limitType !== 'min' && options.limitType !== 'max') {
    throw new TypeError('limitType must be "min" or "max".')
  }
  for (const [name, values] of [['mean', options.mean], ['std', options.std]] as const) {
    if (values !== undefined && (values.length !== 3 || values.some(value => !Number.isFinite(value)))) {
      throw new TypeError(`${name} must contain three finite numbers.`)
    }
  }
  if (options.std?.includes(0)) {
    throw new RangeError('std values must not be zero.')
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}
