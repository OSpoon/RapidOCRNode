import sharp from 'sharp'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { decodeRgbImage, roundToEven } from '../../src/image/image.js'

afterEach(() => {
  vi.unstubAllGlobals()
})
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

  it('converts upstream-compatible grayscale and BGR pixel buffers to RGB', async () => {
    await expect(decodeRgbImage({ data: Uint8Array.of(42), format: 'GRAY8', height: 1, width: 1 }))
      .resolves
      .toMatchObject({ data: Uint8Array.of(42, 42, 42) })
    await expect(decodeRgbImage({ data: Uint8Array.of(30, 20, 10), format: 'BGR24', height: 1, width: 1 }))
      .resolves
      .toMatchObject({ data: Uint8Array.of(10, 20, 30) })
  })

  it('supports padded rows and bottom-up raw image memory layouts', async () => {
    const image = await decodeRgbImage({
      bottomUp: true,
      data: Uint8Array.of(1, 2, 3, 99, 4, 5, 6, 88),
      format: 'RGB',
      height: 2,
      stride: 4,
      width: 1,
    })
    expect(image.data).toEqual(Uint8Array.of(4, 5, 6, 1, 2, 3))
  })

  it('composites raw RGBA and gray-alpha pixels using contrasting backgrounds', async () => {
    const rgba = await decodeRgbImage({
      data: Uint8Array.of(255, 0, 0, 0, 0, 0, 0, 255),
      format: 'RGBA',
      height: 1,
      width: 2,
    })
    expect(rgba.data).toEqual(Uint8Array.of(255, 255, 255, 0, 0, 0))

    const grayAlpha = await decodeRgbImage({ data: Uint8Array.of(20, 128), format: 'GRAYA', height: 1, width: 1 })
    expect(grayAlpha.data).toEqual(Uint8Array.of(147, 147, 147))
  })

  it('downloads and decodes HTTP image URLs', async () => {
    const png = await sharp(Uint8Array.of(10, 20, 30), {
      raw: { channels: 3, height: 1, width: 1 },
    }).png().toBuffer()
    const fetchMock = vi.fn(async () => new Response(png, {
      headers: { 'content-type': 'image/png' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const image = await decodeRgbImage('https://images.example/test.png')

    expect(fetchMock).toHaveBeenCalledWith('https://images.example/test.png', expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(image).toEqual({ data: Uint8Array.of(10, 20, 30), height: 1, width: 1 })
  })

  it('decodes JPEGs with EXIF orientation through Sharp', async () => {
    const encoded = await sharp(Uint8Array.of(
      255,
      0,
      0,
      0,
      255,
      0,
      0,
      0,
      255,
      255,
      255,
      0,
      255,
      0,
      255,
      0,
      255,
      255,
    ), { raw: { channels: 3, height: 3, width: 2 } })
      .jpeg({ chromaSubsampling: '4:4:4', quality: 100 })
      .withMetadata({ orientation: 6 })
      .toBuffer()
    const decoded = await sharp(encoded, { ignoreIcc: true })
      .toColourspace('srgb')
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true })
    const image = await decodeRgbImage(encoded)
    const expected = new Uint8Array(decoded.info.width * decoded.info.height * 3)
    for (let y = 0; y < decoded.info.height; y++) {
      for (let x = 0; x < decoded.info.width; x++) {
        const targetX = decoded.info.height - 1 - y
        const targetY = x
        const sourceOffset = (y * decoded.info.width + x) * 3
        const targetOffset = (targetY * decoded.info.height + targetX) * 3
        expected[targetOffset] = decoded.data[sourceOffset]!
        expected[targetOffset + 1] = decoded.data[sourceOffset + 1]!
        expected[targetOffset + 2] = decoded.data[sourceOffset + 2]!
      }
    }

    expect(image).toEqual({ data: expected, height: decoded.info.width, width: decoded.info.height })
  })

  it('reports failed remote image responses', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 404 })))
    await expect(decodeRgbImage('https://images.example/missing.png'))
      .rejects
      .toThrow('Remote image request failed: HTTP 404')
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
