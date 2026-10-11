/*
 * OpenCV 5 ARM NEON FP16 bicubic image warp compatibility.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { DecodedRgbImage } from './image.js'

const f32 = Math.fround
const float32View = new DataView(new ArrayBuffer(4))

/** Mirrors OpenCV 5's 8-bit, three-channel FP16 bicubic perspective warp. */
export function warpPerspectiveRgbOpenCv5Fp16(
  image: DecodedRgbImage,
  width: number,
  height: number,
  inverseTransform: ArrayLike<number>,
): DecodedRgbImage {
  if (inverseTransform.length !== 9) {
    throw new RangeError('A perspective transform must contain nine values.')
  }

  const matrix = Float32Array.from(inverseTransform, f32)
  const m0 = matrix[0]!
  const m1 = matrix[1]!
  const m2 = matrix[2]!
  const m3 = matrix[3]!
  const m4 = matrix[4]!
  const m5 = matrix[5]!
  const m6 = matrix[6]!
  const m7 = matrix[7]!
  const m8 = matrix[8]!
  const output = new Uint8Array(width * height * 3)
  const bigWidth = Math.max(image.width, 16)
  const bigHeight = Math.max(image.height, 16)

  for (let y = 0; y < height; y++) {
    const mapX = f32(y * m1 + m2)
    const mapY = f32(y * m4 + m5)
    const mapZ = f32(y * m7 + m8)

    for (let x = 0; x < width; x++) {
      const inverseZ = 1 / (mapZ + m6 * x)
      const sourceX = f32((mapX + m0 * x) * inverseZ)
      const sourceY = f32((mapY + m3 * x) * inverseZ)
      const coordinateX = Math.max(-bigWidth, Math.min(bigWidth * 2, sourceX))
      const coordinateY = Math.max(-bigHeight, Math.min(bigHeight * 2, sourceY))
      const sourceXFloor = Math.floor(coordinateX)
      const sourceYFloor = Math.floor(coordinateY)
      const horizontalWeights = cubicWeights(f32(coordinateX - sourceXFloor))
      const verticalWeights = cubicWeights(f32(coordinateY - sourceYFloor))
      const outputOffset = (y * width + x) * 3

      for (let channel = 0; channel < 3; channel++) {
        let accumulator = 0
        for (let row = 0; row < 4; row++) {
          const sourceYIndex = clamp(sourceYFloor - 1 + row, image.height)
          const sourceXIndex = sourceXFloor - 1
          const offset = (sourceYIndex * image.width + clamp(sourceXIndex, image.width)) * 3 + channel
          const p0 = image.data[offset]!
          const p1 = image.data[(sourceYIndex * image.width + clamp(sourceXIndex + 1, image.width)) * 3 + channel]!
          const p2 = image.data[(sourceYIndex * image.width + clamp(sourceXIndex + 2, image.width)) * 3 + channel]!
          const p3 = image.data[(sourceYIndex * image.width + clamp(sourceXIndex + 3, image.width)) * 3 + channel]!

          let horizontal = fusedHalf(p1, horizontalWeights[1], roundedHalf(p0 * horizontalWeights[0]))
          horizontal = fusedHalf(p2, horizontalWeights[2], horizontal)
          horizontal = fusedHalf(p3, horizontalWeights[3], horizontal)
          accumulator = fusedHalf(horizontal, verticalWeights[row], accumulator)
        }

        output[outputOffset + channel] = clampByte(roundToEven(accumulator))
      }
    }
  }

  return { data: output, height, width }
}

function cubicWeights(value: number): [number, number, number, number] {
  const alpha = roundedHalf(value)
  const a = roundedHalf(-0.75)
  const aPlusTwo = roundedHalf(1.25)
  const negativeAPlusThree = roundedHalf(-2.25)
  const one = 1
  const alphaSquared = roundedHalf(alpha * alpha)
  const oneMinusAlpha = roundedHalf(one - alpha)
  const oneMinusAlphaSquared = roundedHalf(oneMinusAlpha * oneMinusAlpha)
  const weight0 = roundedHalf(a * roundedHalf(alpha * oneMinusAlphaSquared))
  const weight3 = roundedHalf(a * roundedHalf(alphaSquared * oneMinusAlpha))
  const weight1 = fusedHalf(alphaSquared, fusedHalf(aPlusTwo, alpha, negativeAPlusThree), one)
  const weight2 = roundedHalf(roundedHalf(roundedHalf(one - weight0) - weight1) - weight3)
  return [weight0, weight1, weight2, weight3]
}

function roundedHalf(value: number): number {
  float32View.setFloat32(0, value, false)
  const bits = float32View.getUint32(0, false)
  const sign = (bits >>> 16) & 0x8000
  const exponent = (bits >>> 23) & 0xFF
  let mantissa = bits & 0x7FFFFF

  if (exponent === 0xFF) {
    const halfBits = sign | 0x7C00 | (mantissa === 0 ? 0 : 0x0200)
    return halfToNumber(halfBits)
  }

  const halfExponent = exponent - 127 + 15
  if (halfExponent >= 0x1F) {
    return halfToNumber(sign | 0x7C00)
  }
  if (halfExponent <= 0) {
    if (halfExponent < -10) {
      return halfToNumber(sign)
    }

    mantissa |= 0x800000
    const shift = 14 - halfExponent
    let halfMantissa = mantissa >>> shift
    const remainder = mantissa & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
      halfMantissa++
    }
    return halfToNumber(sign | halfMantissa)
  }

  let halfBits = sign | (halfExponent << 10) | (mantissa >>> 13)
  const remainder = mantissa & 0x1FFF
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfBits & 1) !== 0)) {
    halfBits++
  }
  return halfToNumber(halfBits)
}

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1
  const exponent = (bits >>> 10) & 0x1F
  const mantissa = bits & 0x03FF
  if (exponent === 0) {
    return sign * mantissa * 2 ** -24
  }
  if (exponent === 0x1F) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15)
}

function fusedHalf(left: number, right: number, addend: number): number {
  return roundedHalf(left * right + addend)
}

function roundToEven(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (fraction < 0.5) {
    return floor
  }
  if (fraction > 0.5) {
    return floor + 1
  }
  return floor % 2 === 0 ? floor : floor + 1
}

function clamp(value: number, size: number): number {
  return Math.max(0, Math.min(size - 1, value))
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, value))
}
