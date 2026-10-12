import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import sharp from 'sharp'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { decodeRgbImage, LoadImageError, roundToEven } from '../../src/image/image.js'

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

  it('matches upstream path, URL, and encoded-byte input cases', async () => {
    const encoded = await sharp(Uint8Array.of(10, 20, 30, 40, 50, 60), {
      raw: { channels: 3, height: 1, width: 2 },
    }).png().toBuffer()
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-input-'))
    const imagePath = join(directory, 'sample.png')
    const remoteUrl = 'https://images.example/sample.png'
    const fetchMock = vi.fn(async () => new Response(encoded))
    vi.stubGlobal('fetch', fetchMock)

    try {
      await writeFile(imagePath, encoded)
      const expected = await decodeRgbImage(Uint8Array.from(encoded))
      const inputs = [
        imagePath,
        pathToFileURL(imagePath),
        Uint8Array.from(encoded),
        Uint8Array.from(encoded).buffer,
        new Blob([Uint8Array.from(encoded)]),
        remoteUrl,
        new URL(remoteUrl),
      ] as const

      for (const input of inputs) {
        await expect(decodeRgbImage(input)).resolves.toEqual(expected)
      }
      expect(fetchMock).toHaveBeenCalledTimes(2)
    }
    finally {
      await rm(directory, { recursive: true, force: true })
    }
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

  it('drops raw RGBA alpha and composites gray-alpha pixels using a contrasting background', async () => {
    const rgba = await decodeRgbImage({
      data: Uint8Array.of(255, 0, 0, 0, 0, 0, 0, 255),
      format: 'RGBA',
      height: 1,
      width: 2,
    })
    expect(rgba.data).toEqual(Uint8Array.of(255, 0, 0, 0, 0, 0))

    const grayAlpha = await decodeRgbImage({ data: Uint8Array.of(20, 128), format: 'GRAYA', height: 1, width: 1 })
    expect(grayAlpha.data).toEqual(Uint8Array.of(147, 147, 147))
  })

  it('decodes the monochrome PNG case used by upstream input tests', async () => {
    const encoded = await sharp(Uint8Array.of(0, 255), {
      raw: { channels: 1, height: 1, width: 2 },
    }).png({ palette: true, colours: 2 }).toBuffer()

    await expect(decodeRgbImage(encoded)).resolves.toMatchObject({
      data: Uint8Array.of(0, 0, 0, 255, 255, 255),
      height: 1,
      width: 2,
    })
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

  it('matches upstream EXIF behavior for encoded bytes and file paths', async () => {
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
    const byteImage = await decodeRgbImage(encoded)
    expect(byteImage).toEqual({ data: Uint8Array.from(decoded.data), height: decoded.info.height, width: decoded.info.width })

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

    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-exif-'))
    const imagePath = join(directory, 'orientation.jpg')
    try {
      await writeFile(imagePath, encoded)
      const pathImage = await decodeRgbImage(imagePath)
      expect(pathImage).toEqual({ data: expected, height: decoded.info.width, width: decoded.info.height })
    }
    finally {
      await rm(directory, { recursive: true, force: true })
    }
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

  it.each([
    {
      name: 'black text on transparent pixels',
      input: Uint8Array.of(0, 0, 0, 0, 0, 0, 0, 255),
      expected: Uint8Array.of(255, 255, 255, 0, 0, 0),
    },
    {
      name: 'white text on transparent pixels',
      input: Uint8Array.of(255, 255, 255, 0, 255, 255, 255, 255),
      expected: Uint8Array.of(0, 0, 0, 255, 255, 255),
    },
  ])('matches upstream transparent-image contrast handling for $name', async ({ input, expected }) => {
    const png = await sharp(input, { raw: { channels: 4, height: 1, width: 2 } }).png().toBuffer()
    const image = await decodeRgbImage(png)
    expect(image.data).toEqual(expected)
  })

  it('raises the exported upstream-compatible error for a missing image input', async () => {
    await expect(decodeRgbImage(null as never)).rejects.toBeInstanceOf(LoadImageError)
  })

  it('wraps malformed encoded image errors with an input hint', async () => {
    await expect(decodeRgbImage(Uint8Array.of(1, 2, 3))).rejects.toThrow(/Unable to decode image input/)
  })
})
