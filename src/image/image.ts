/*
 * TypeScript adaptation of RapidOCR v3.10.0 image preprocessing and input handling.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { ImageInput, RawPixelFormat, RawPixelImage, RawRgbImage } from '../shared/types.js'
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
    if (isHttpUrl(input))
      return decodeRemoteImage(input)
    return decodeEncodedImage(input)
  }

  if (input instanceof Uint8Array) {
    return decodeEncodedImage(input)
  }

  return normalizeRawPixelImage(input)
}

async function decodeRemoteImage(url: string): Promise<DecodedRgbImage> {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) })
  if (!response.ok) {
    throw new Error(`Remote image request failed: HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}`)
  }
  return decodeEncodedImage(Buffer.from(await response.arrayBuffer()))
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//iu.test(value)
}

/** Adds the vertical letterbox padding used by RapidOCR before text detection. */
export function addVerticalPadding(
  image: DecodedRgbImage,
  widthHeightRatio = 8,
  minHeight = 30,
): { image: DecodedRgbImage, top: number } {
  const needsRatioPadding = widthHeightRatio !== -1 && image.width / image.height > widthHeightRatio
  if (image.height > minHeight && !needsRatioPadding)
    return { image, top: 0 }

  const desiredHeight = Math.max(
    widthHeightRatio === -1 ? 0 : Math.trunc(image.width / widthHeightRatio),
    minHeight,
  ) * 2
  const top = Math.trunc(Math.abs(desiredHeight - image.height) / 2)
  const height = image.height + top * 2
  const data = new Uint8Array(image.width * height * 3)
  data.set(image.data, top * image.width * 3)
  return { image: { data, height, width: image.width }, top }
}

async function decodeEncodedImage(input: string | Uint8Array): Promise<DecodedRgbImage> {
  try {
    const { data, info } = await sharp(input, { failOn: 'error', ignoreIcc: true })
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

function normalizeRawPixelImage(image: RawPixelImage | RawRgbImage): DecodedRgbImage {
  validateRawDimensions(image)
  if (!('format' in image)) {
    const expectedLength = image.width * image.height * 3
    if (image.data.byteLength !== expectedLength) {
      throw new RangeError(`Raw RGB image data must contain ${expectedLength} bytes.`)
    }
    return image
  }

  const channels = channelsForFormat(image.format)
  const rowBytes = image.width * channels
  const stride = image.stride ?? rowBytes
  if (!Number.isInteger(stride) || stride < rowBytes) {
    throw new RangeError(`Raw image stride must be an integer at least ${rowBytes}.`)
  }
  const requiredLength = stride * image.height
  if (image.data.byteLength < requiredLength) {
    throw new RangeError(`Raw image buffer must contain at least ${requiredLength} bytes.`)
  }

  const alphaIndex = channels === 4 || channels === 2 ? channels - 1 : -1
  const background = channels === 4 ? chooseRawRgbaBackground(image, stride) : 255
  const rgb = new Uint8Array(image.width * image.height * 3)
  for (let y = 0; y < image.height; y++) {
    const sourceY = image.bottomUp ? image.height - y - 1 : y
    for (let x = 0; x < image.width; x++) {
      const sourceOffset = sourceY * stride + x * channels
      const targetOffset = (y * image.width + x) * 3
      if (channels === 1) {
        const gray = image.data[sourceOffset] ?? 0
        rgb[targetOffset] = gray
        rgb[targetOffset + 1] = gray
        rgb[targetOffset + 2] = gray
        continue
      }
      if (channels === 2) {
        const gray = image.data[sourceOffset] ?? 0
        const alpha = image.data[sourceOffset + alphaIndex] ?? 0
        const value = alpha === 0 ? 255 : Math.min(255, gray + 255 - alpha)
        rgb[targetOffset] = value
        rgb[targetOffset + 1] = value
        rgb[targetOffset + 2] = value
        continue
      }

      const isBgr = image.format === 'BGR' || image.format === 'BGR24' || image.format === 'BGRA' || image.format === 'BGRA32' || image.format === 'BGRX'
      const red = image.data[sourceOffset + (isBgr ? 2 : 0)] ?? 0
      const green = image.data[sourceOffset + 1] ?? 0
      const blue = image.data[sourceOffset + (isBgr ? 0 : 2)] ?? 0
      const alpha = channels === 4 ? (image.data[sourceOffset + alphaIndex] ?? 0) / 255 : 1
      rgb[targetOffset] = Math.trunc(red * alpha + background * (1 - alpha))
      rgb[targetOffset + 1] = Math.trunc(green * alpha + background * (1 - alpha))
      rgb[targetOffset + 2] = Math.trunc(blue * alpha + background * (1 - alpha))
    }
  }

  return { data: rgb, height: image.height, width: image.width }
}

function validateRawDimensions(image: RawRgbImage | RawPixelImage): void {
  if (!Number.isInteger(image.width) || image.width < 1) {
    throw new RangeError('Raw image width must be a positive integer.')
  }

  if (!Number.isInteger(image.height) || image.height < 1) {
    throw new RangeError('Raw image height must be a positive integer.')
  }
}

function channelsForFormat(format: RawPixelFormat): number {
  switch (format) {
    case 'GRAY':
    case 'GRAY8':
      return 1
    case 'GRAY_ALPHA':
    case 'GRAYA':
      return 2
    case 'RGB':
    case 'RGB24':
    case 'BGR':
    case 'BGR24':
      return 3
    case 'RGBA':
    case 'RGBA32':
    case 'RGBX':
    case 'BGRA':
    case 'BGRA32':
    case 'BGRX':
      return 4
    default:
      throw new TypeError(`Unsupported raw image format: ${String(format)}.`)
  }
}

function chooseRawRgbaBackground(image: RawPixelImage, stride: number): number {
  const isBgr = image.format === 'BGRA' || image.format === 'BGRA32' || image.format === 'BGRX'
  let luminance = 0
  let visiblePixels = 0
  for (let y = 0; y < image.height; y++) {
    const sourceY = image.bottomUp ? image.height - y - 1 : y
    for (let x = 0; x < image.width; x++) {
      const offset = sourceY * stride + x * 4
      const alpha = image.data[offset + 3] ?? 0
      if (alpha === 0)
        continue
      const red = image.data[offset + (isBgr ? 2 : 0)] ?? 0
      const green = image.data[offset + 1] ?? 0
      const blue = image.data[offset + (isBgr ? 0 : 2)] ?? 0
      luminance += 0.299 * red + 0.587 * green + 0.114 * blue
      visiblePixels++
    }
  }
  return visiblePixels === 0 || luminance / visiblePixels < 128 ? 255 : 0
}
