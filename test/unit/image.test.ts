import sharp from 'sharp'
import { describe, expect, it } from 'vitest'
import { decodeRgbImage, resizeRgbImage, roundToEven } from '../../src/image.js'

describe('roundToEven', () => {
  it('matches NumPy and Python ties-to-even rounding', () => {
    expect([roundToEven(30.5), roundToEven(31.5), roundToEven(-1.5), roundToEven(-2.5)])
      .toEqual([30, 32, -2, -2])
  })
})

describe('decodeRgbImage', () => {
  it('accepts well-formed caller-provided RGB pixels', async () => {
    const image = { data: Uint8Array.of(10, 20, 30), height: 1, width: 1 }
    await expect(decodeRgbImage(image)).resolves.toBe(image)
  })

  it('rejects raw data whose byte length does not match its dimensions', async () => {
    await expect(decodeRgbImage({ data: Uint8Array.of(1, 2), height: 1, width: 1 }))
      .rejects
      .toThrow(/must contain 3 bytes/)
  })

  it('composites fully transparent pixels against a contrasting background', async () => {
    const png = await sharp(Uint8Array.of(20, 20, 20, 0, 0, 0, 0, 255), {
      raw: { channels: 4, height: 1, width: 2 },
    }).png().toBuffer()

    const image = await decodeRgbImage(png)
    expect(image.data).toEqual(Uint8Array.of(255, 255, 255, 0, 0, 0))
  })

  it('wraps malformed encoded image errors with an input hint', async () => {
    await expect(decodeRgbImage(Uint8Array.of(1, 2, 3))).rejects.toThrow(/Unable to decode image input/)
  })
})

describe('resizeRgbImage', () => {
  it('uses half-pixel bilinear sampling with replicated image edges', async () => {
    const image = { data: Uint8Array.of(0, 0, 0, 255, 255, 255), height: 1, width: 2 }
    const resized = await resizeRgbImage(image, 4, 1)

    expect(Array.from(resized)).toEqual(Array.from(Uint8Array.of(
      0,
      0,
      0,
      64,
      64,
      64,
      191,
      191,
      191,
      255,
      255,
      255,
    )))
  })

  it('rejects invalid output dimensions', async () => {
    await expect(resizeRgbImage({ data: Uint8Array.of(1, 2, 3), height: 1, width: 1 }, 0, 2))
      .rejects
      .toThrow(/positive integers/)
  })
})
