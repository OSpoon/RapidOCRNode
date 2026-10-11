import { describe, expect, it } from 'vitest'
import { mapWordPositionsToBox } from '../../src/pipeline/recognition/word-boxes.js'

describe('word box mapping', () => {
  it('uses a perspective transform for word boxes inside skewed detector boxes', async () => {
    const words = await mapWordPositionsToBox(
      [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 80, y: 20 },
        { x: 10, y: 20 },
      ],
      100,
      22,
      'AB',
      [{
        columns: [2, 6],
        confidence: 0.9,
        end: 0.7,
        lineWidth: 10,
        start: 0.2,
        text: 'AB',
      }],
    )

    expect(words).toHaveLength(1)
    expect(words[0]!.text).toBe('AB')
    const [topLeft, topRight, bottomRight, bottomLeft] = words[0]!.box
    expect(topRight.x - topLeft.x).toBeGreaterThan(bottomRight.x - bottomLeft.x)
    expect(topLeft.y).toBe(0)
    expect(bottomLeft.y).toBeGreaterThanOrEqual(19)
  })

  it('rotates positions back into a vertical detector quadrilateral', async () => {
    const words = await mapWordPositionsToBox(
      [
        { x: 10, y: 0 },
        { x: 30, y: 0 },
        { x: 30, y: 100 },
        { x: 10, y: 100 },
      ],
      100,
      20,
      '中文',
      [
        { columns: [1], confidence: 0.9, end: 0.2, lineWidth: 10, start: 0.1, text: '中' },
        { columns: [5], confidence: 0.9, end: 0.6, lineWidth: 10, start: 0.5, text: '文' },
      ],
    )

    expect(words).toHaveLength(2)
    expect(Math.abs(words[0]!.box[0].y - words[0]!.box[1].y)).toBeGreaterThan(1)
    expect(words.every(word => word.box.every(point => point.x >= 10 && point.x <= 30))).toBe(true)
  })

  it('estimates Han character widths from neighboring CTC columns', async () => {
    const words = await mapWordPositionsToBox(
      [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 20 },
        { x: 0, y: 20 },
      ],
      100,
      20,
      '中文',
      [
        { columns: [2], confidence: 0.9, end: 0.3, lineWidth: 10, start: 0.2, text: '中' },
        { columns: [4], confidence: 0.9, end: 0.5, lineWidth: 10, start: 0.4, text: '文' },
      ],
    )

    expect(words.map(word => word.box[1].x - word.box[0].x)).toEqual([20, 20])
  })
})
