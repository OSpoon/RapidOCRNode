import { describe, expect, it } from 'vitest'
import { resizeForDetection } from '../../src/rapidocr.js'

describe('resizeForDetection', () => {
  it('preserves images already inside the upstream side-length bounds', async () => {
    const image = { data: new Uint8Array(323 * 430 * 3), height: 430, width: 323 }
    const resized = await resizeForDetection(image)

    expect(resized.image).toBe(image)
    expect(resized.scaleX).toBe(1)
    expect(resized.scaleY).toBe(1)
  })

  it('scales tiny images to at least the minimum side length using multiples of 32', async () => {
    const image = { data: new Uint8Array(20 * 10 * 3), height: 10, width: 20 }
    const resized = await resizeForDetection(image)

    expect(resized.image.width).toBe(64)
    expect(resized.image.height).toBe(32)
    expect(resized.scaleX).toBe(20 / 64)
    expect(resized.scaleY).toBe(10 / 32)
  })
})
