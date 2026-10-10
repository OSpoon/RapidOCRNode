import type { DetectorBox } from '../../src/detector.js'
import { describe, expect, it } from 'vitest'
import { extractBoxes, sortBoxesInReadingOrder } from '../../src/detector.js'

function box(textX: number, topY: number): DetectorBox {
  return {
    box: [
      { x: textX, y: topY },
      { x: textX + 10, y: topY },
      { x: textX + 10, y: topY + 12 },
      { x: textX, y: topY + 12 },
    ],
    score: 0.9,
  }
}

describe('extractBoxes', () => {
  it('extracts and expands a confident text region from a probability map', async () => {
    const map = new Float32Array(32 * 32)
    for (let y = 12; y <= 18; y++) {
      for (let x = 10; x <= 22; x++) {
        map[y * 32 + x] = 0.9
      }
    }

    const boxes = await extractBoxes(map, 32, 32, 32, 32)
    expect(boxes).toHaveLength(1)
    expect(boxes[0]!.score).toBeGreaterThan(0.7)
    expect(boxes[0]!.box.every(point => point.x >= 0 && point.x < 32 && point.y >= 0 && point.y < 32)).toBe(true)
  })

  it('returns no boxes for empty or below-threshold probability maps', async () => {
    expect(await extractBoxes(new Float32Array(32 * 32), 32, 32, 32, 32)).toEqual([])
    expect(await extractBoxes(new Float32Array(32 * 32).fill(0.4), 32, 32, 32, 32)).toEqual([])
  })

  it('matches the upstream 2-by-2 dilation anchor on regular and clipped boxes', async () => {
    const rectangle = new Float32Array(32 * 32)
    for (let y = 10; y < 18; y++) {
      for (let x = 8; x < 24; x++) rectangle[y * 32 + x] = 0.9
    }
    const regular = await extractBoxes(rectangle, 32, 32, 32, 32)
    expect(regular[0]!.box).toEqual([
      { x: 4, y: 6 },
      { x: 28, y: 6 },
      { x: 28, y: 22 },
      { x: 4, y: 22 },
    ])
    expect(regular[0]!.score).toBeCloseTo(0.75294, 4)

    const edge = new Float32Array(32 * 32)
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 12; x++) edge[y * 32 + x] = 0.9
    }
    const clipped = await extractBoxes(edge, 32, 32, 32, 32)
    expect(clipped[0]!.box).toEqual([
      { x: 0, y: 0 },
      { x: 16, y: 0 },
      { x: 16, y: 12 },
      { x: 0, y: 12 },
    ])
    expect(clipped[0]!.score).toBeCloseTo(0.73846, 4)
  })
})

describe('sortBoxesInReadingOrder', () => {
  it('orders left to right within rows and tolerates a one-pixel contour rounding shift', () => {
    const boxes = [box(135, 241), box(254, 232), box(136, 219), box(33, 225)]

    expect(sortBoxesInReadingOrder(boxes, 9).map(item => item.box[0]))
      .toEqual([{ x: 33, y: 225 }, { x: 136, y: 219 }, { x: 254, y: 232 }, { x: 135, y: 241 }])
  })

  it('groups low-resolution boxes at the upstream 10px row boundary', () => {
    const boxes = [box(54, 169), box(19, 178)]

    expect(sortBoxesInReadingOrder(boxes, 10).map(item => item.box[0]))
      .toEqual([{ x: 19, y: 178 }, { x: 54, y: 169 }])
  })
})
