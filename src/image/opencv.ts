import type { Point } from '../shared/types.js'
import type { DecodedRgbImage } from './image.js'
import { createRequire } from 'node:module'
import { arch } from 'node:process'
import { warpPerspectiveRgbOpenCv5Fp16 } from './warp.js'

export interface OpenCvPoint {
  x: number
  y: number
}

export interface OpenCvMat {
  rows: number
  data: Uint8Array
  data32F: Float32Array
  data32S: Int32Array
  data64F: Float64Array
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

export interface OpenCv {
  CV_8UC1: number
  CV_8UC3: number
  CV_32FC1: number
  CV_32FC2: number
  CV_32SC2: number
  CV_64F: number
  DECOMP_LU: number
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
  invert: (src: OpenCvMat, dst: OpenCvMat, method: number) => number
  mean: (src: OpenCvMat, mask: OpenCvMat) => number[]
  matFromArray: (rows: number, cols: number, type: number, values: number[] | ArrayBufferView) => OpenCvMat
  warpPerspective: (src: OpenCvMat, dst: OpenCvMat, transform: OpenCvMat, size: { width: number, height: number }, flags: number, borderMode: number, borderValue: number[]) => void
}

type OpenCvModule = OpenCv | Promise<OpenCv>

const cvModule = createRequire(import.meta.url)('@techstark/opencv-js') as OpenCvModule
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

export async function getOpenCv(): Promise<OpenCv> {
  await openCvReady
  if (!openCv)
    throw new Error('OpenCV.js failed to initialize.')
  return openCv
}

export async function resizeRgbImageOpenCv(image: DecodedRgbImage, width: number, height: number): Promise<Uint8Array> {
  const cv = await getOpenCv()
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
  const cv = await getOpenCv()
  const source = cv.matFromArray(image.height, image.width, cv.CV_8UC3, image.data)
  const sourcePointMat = cv.matFromArray(sourcePoints.length, 1, cv.CV_32FC2, sourcePoints.flatMap(point => [point.x, point.y]))
  const destinationPointMat = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, width, 0, width, height, 0, height])
  let transform: OpenCvMat | undefined
  try {
    transform = cv.getPerspectiveTransform(sourcePointMat, destinationPointMat)

    if (arch === 'arm64') {
      const inverseTransform = new cv.Mat(3, 3, cv.CV_64F)
      try {
        if (cv.invert(transform, inverseTransform, cv.DECOMP_LU) !== 0) {
          return warpPerspectiveRgbOpenCv5Fp16(image, width, height, inverseTransform.data64F)
        }
      }
      finally {
        inverseTransform.delete()
      }
    }

    const destination = new cv.Mat(height, width, cv.CV_8UC3)
    try {
      cv.warpPerspective(source, destination, transform, new cv.Size(width, height), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar(0))
      return { data: destination.data.slice(), height, width }
    }
    finally {
      destination.delete()
    }
  }
  finally {
    transform?.delete()
    destinationPointMat.delete()
    sourcePointMat.delete()
    source.delete()
  }
}
