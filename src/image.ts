/*
 * TypeScript adaptation of RapidOCR v3.10.0 image preprocessing and input handling.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { ImageInput, RawRgbImage } from './types.js'
import { Buffer } from 'node:buffer'
import sharp from 'sharp'

export type DecodedRgbImage = RawRgbImage

export function roundToEven(value: number): number {
  const lower = Math.floor(value)
  const fraction = value - lower
  if (fraction < 0.5)
    return lower
  if (fraction > 0.5)
    return lower + 1
  return lower % 2 === 0 ? lower : lower + 1
}

export async function decodeRgbImage(input: ImageInput): Promise<DecodedRgbImage> {
  if (typeof input === 'string') {
    return decodeEncodedImage(input)
  }

  if (input instanceof Uint8Array) {
    return decodeEncodedImage(input)
  }

  validateRawRgbImage(input)
  return input
}

export async function resizeRgbImage(
  image: DecodedRgbImage,
  width: number,
  height: number,
): Promise<Buffer> {
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
    throw new RangeError('Resized image dimensions must be positive integers.')
  }

  const output = Buffer.alloc(width * height * 3)
  const xCoordinates = Array.from({ length: width }, (_, x) => getInterpolationCoordinate(x, image.width, width))
  const yCoordinates = Array.from({ length: height }, (_, y) => getInterpolationCoordinate(y, image.height, height))

  for (let y = 0; y < height; y++) {
    const yCoordinate = yCoordinates[y]!
    for (let x = 0; x < width; x++) {
      const xCoordinate = xCoordinates[x]!
      const topLeft = (yCoordinate.low * image.width + xCoordinate.low) * 3
      const topRight = (yCoordinate.low * image.width + xCoordinate.high) * 3
      const bottomLeft = (yCoordinate.high * image.width + xCoordinate.low) * 3
      const bottomRight = (yCoordinate.high * image.width + xCoordinate.high) * 3
      const target = (y * width + x) * 3

      for (let channel = 0; channel < 3; channel++) {
        const top = interpolate(
          image.data[topLeft + channel] ?? 0,
          image.data[topRight + channel] ?? 0,
          xCoordinate.weight,
        )
        const bottom = interpolate(
          image.data[bottomLeft + channel] ?? 0,
          image.data[bottomRight + channel] ?? 0,
          xCoordinate.weight,
        )
        output[target + channel] = Math.round(interpolate(top, bottom, yCoordinate.weight))
      }
    }
  }

  return output
}

function getInterpolationCoordinate(index: number, sourceLength: number, targetLength: number): {
  high: number
  low: number
  weight: number
} {
  const coordinate = (index + 0.5) * sourceLength / targetLength - 0.5
  const low = Math.max(0, Math.floor(coordinate))
  const weight = low >= sourceLength - 1 ? 0 : Math.max(0, coordinate - low)
  return { high: Math.min(low + 1, sourceLength - 1), low, weight }
}

function interpolate(low: number, high: number, weight: number): number {
  return low + (high - low) * weight
}

async function decodeEncodedImage(input: string | Uint8Array): Promise<DecodedRgbImage> {
  try {
    const { data, info } = await sharp(input, { failOn: 'error' })
      .rotate()
      .toColourspace('srgb')
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true })

    if (info.channels !== 4) {
      throw new Error(`Expected an RGBA image, received ${info.channels} channels.`)
    }

    return { data: convertRgbaToRgb(data), height: info.height, width: info.width }
  }
  catch (cause) {
    throw new Error('Unable to decode image input. Pass an image path, encoded bytes, or raw RGB pixels.', { cause })
  }
}

function convertRgbaToRgb(rgba: Uint8Array): Uint8Array {
  const background = chooseBackground(rgba)
  const rgb = new Uint8Array((rgba.length / 4) * 3)

  for (let sourceIndex = 0, targetIndex = 0; sourceIndex < rgba.length; sourceIndex += 4, targetIndex += 3) {
    const opacity = (rgba[sourceIndex + 3] ?? 0) / 255
    const inverseOpacity = 1 - opacity
    rgb[targetIndex] = Math.trunc((rgba[sourceIndex] ?? 0) * opacity + background * inverseOpacity)
    rgb[targetIndex + 1] = Math.trunc((rgba[sourceIndex + 1] ?? 0) * opacity + background * inverseOpacity)
    rgb[targetIndex + 2] = Math.trunc((rgba[sourceIndex + 2] ?? 0) * opacity + background * inverseOpacity)
  }

  return rgb
}

function chooseBackground(rgba: Uint8Array): number {
  let luminanceSum = 0
  let visiblePixels = 0

  for (let index = 0; index < rgba.length; index += 4) {
    if ((rgba[index + 3] ?? 0) === 0) {
      continue
    }

    const red = rgba[index] ?? 0
    const green = rgba[index + 1] ?? 0
    const blue = rgba[index + 2] ?? 0
    luminanceSum += 0.299 * red + 0.587 * green + 0.114 * blue
    visiblePixels++
  }

  return visiblePixels === 0 || luminanceSum / visiblePixels < 128 ? 255 : 0
}

function validateRawRgbImage(image: RawRgbImage): void {
  if (!Number.isInteger(image.width) || image.width < 1) {
    throw new RangeError('Raw RGB image width must be a positive integer.')
  }

  if (!Number.isInteger(image.height) || image.height < 1) {
    throw new RangeError('Raw RGB image height must be a positive integer.')
  }

  const expectedLength = image.width * image.height * 3
  if (image.data.byteLength !== expectedLength) {
    throw new RangeError(`Raw RGB image data must contain ${expectedLength} bytes.`)
  }
}
