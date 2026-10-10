/*
 * TypeScript adaptation of the RapidOCR v3.10.0 image crop and rotation processing.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from './image.js'
import type { Point, TextBox } from './types.js'
import { warpPerspectiveRgbOpenCv } from './detector.js'

export async function cropPerspective(image: DecodedRgbImage, box: TextBox): Promise<DecodedRgbImage> {
  const [topLeft, topRight, bottomRight, bottomLeft] = box
  const width = Math.max(1, Math.trunc(Math.max(distance(topLeft, topRight), distance(bottomLeft, bottomRight))))
  const height = Math.max(1, Math.trunc(Math.max(distance(topLeft, bottomLeft), distance(topRight, bottomRight))))
  const crop = await warpPerspectiveRgbOpenCv(image, box, width, height)

  return height / width >= 1.5 ? rotateCounterClockwise(crop) : crop
}

export function rotate180(image: DecodedRgbImage): DecodedRgbImage {
  const output = new Uint8Array(image.data.length)
  const pixelCount = image.width * image.height
  for (let index = 0; index < pixelCount; index++) {
    const sourceOffset = index * 3
    const targetOffset = (pixelCount - index - 1) * 3
    output[targetOffset] = image.data[sourceOffset] ?? 0
    output[targetOffset + 1] = image.data[sourceOffset + 1] ?? 0
    output[targetOffset + 2] = image.data[sourceOffset + 2] ?? 0
  }
  return { data: output, height: image.height, width: image.width }
}

function rotateCounterClockwise(image: DecodedRgbImage): DecodedRgbImage {
  const width = image.height
  const height = image.width
  const output = new Uint8Array(image.data.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sourceX = image.width - 1 - y
      const sourceY = x
      const sourceOffset = (sourceY * image.width + sourceX) * 3
      const targetOffset = (y * width + x) * 3
      output[targetOffset] = image.data[sourceOffset] ?? 0
      output[targetOffset + 1] = image.data[sourceOffset + 1] ?? 0
      output[targetOffset + 2] = image.data[sourceOffset + 2] ?? 0
    }
  }
  return { data: output, height, width }
}

function distance(left: Point, right: Point): number {
  return Math.hypot(right.x - left.x, right.y - left.y)
}
