import { describe, expect, it } from 'vitest'
import { cropPerspective, rotate180 } from '../../src/image/crop.js'

describe('cropPerspective', () => {
  it('preserves pixels for an axis-aligned box', async () => {
    const image = {
      data: Uint8Array.from({ length: 18 }, (_, index) => index),
      height: 2,
      width: 3,
    }
    const crop = await cropPerspective(image, [
      { x: 0, y: 0 },
      { x: 3, y: 0 },
      { x: 3, y: 2 },
      { x: 0, y: 2 },
    ])

    expect(crop).toEqual(image)
  })

  it('rotates tall crops counterclockwise', async () => {
    const image = {
      data: Uint8Array.from([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]),
      height: 4,
      width: 2,
    }
    const crop = await cropPerspective(image, [
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 2, y: 4 },
      { x: 0, y: 4 },
    ])

    expect(crop).toEqual({ data: Uint8Array.from([1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6]), height: 2, width: 4 })
  })

  it('uses the maximum opposing edge lengths for integer-rounded detector boxes', async () => {
    const image = { data: new Uint8Array(500 * 120 * 3), height: 120, width: 500 }
    const crop = await cropPerspective(image, [
      { x: 10, y: 10 },
      { x: 389, y: 12 },
      { x: 388, y: 102 },
      { x: 10, y: 100 },
    ])

    expect(crop.width).toBe(379)
    expect(crop.height).toBe(90)
  })
})

describe('rotate180', () => {
  it('reverses pixel order without changing dimensions', () => {
    const image = { data: Uint8Array.from([1, 2, 3, 4, 5, 6]), height: 1, width: 2 }
    expect(rotate180(image)).toEqual({ data: Uint8Array.from([4, 5, 6, 1, 2, 3]), height: 1, width: 2 })
  })
})
