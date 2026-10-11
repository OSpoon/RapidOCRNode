import type { DetectorBox } from '../../src/pipeline/detection/detector.js'
import { describe, expect, it } from 'vitest'
import { extractBoxes, getDetectorResizeRatio, sortBoxesInReadingOrder } from '../../src/pipeline/detection/detector.js'
import { mapDetectorCoordinate } from '../../src/pipeline/detection/postprocess.js'

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

  it('accepts the upstream DB thresholds and expansion controls', async () => {
    const map = new Float32Array(32 * 32)
    for (let y = 10; y < 18; y++) {
      for (let x = 8; x < 24; x++) map[y * 32 + x] = 0.9
    }

    const defaults = await extractBoxes(map, 32, 32, 32, 32)
    const withoutExpansion = await extractBoxes(map, 32, 32, 32, 32, { unclipRatio: 0 })
    const closeRegions = new Float32Array(32 * 32)
    for (let y = 10; y < 18; y++) {
      for (let x = 5; x < 10; x++) closeRegions[y * 32 + x] = 0.9
      for (let x = 11; x < 16; x++) closeRegions[y * 32 + x] = 0.9
    }
    const dilated = await extractBoxes(closeRegions, 32, 32, 32, 32)
    const withoutDilation = await extractBoxes(closeRegions, 32, 32, 32, 32, { useDilation: false })

    expect(await extractBoxes(map, 32, 32, 32, 32, { threshold: 0.95 })).toEqual([])
    expect(await extractBoxes(map, 32, 32, 32, 32, { boxThreshold: 0.95 })).toEqual([])
    expect(withoutExpansion[0]!.box[0]!.x).toBeGreaterThan(defaults[0]!.box[0]!.x)
    expect(dilated).toHaveLength(1)
    expect(withoutDilation).toHaveLength(2)
  })

  it('limits the number of contours to postprocess', async () => {
    const map = new Float32Array(32 * 32)
    for (let y = 8; y < 15; y++) {
      for (let x = 3; x < 11; x++) map[y * 32 + x] = 0.9
      for (let x = 20; x < 28; x++) map[y * 32 + x] = 0.9
    }

    expect(await extractBoxes(map, 32, 32, 32, 32)).toHaveLength(2)
    expect(await extractBoxes(map, 32, 32, 32, 32, { maxCandidates: 1 })).toHaveLength(1)
  })

  it('rejects invalid postprocess settings', async () => {
    const map = new Float32Array(32 * 32)

    await expect(extractBoxes(map, 32, 32, 32, 32, { threshold: 1.1 })).rejects.toThrow(/threshold/)
    await expect(extractBoxes(map, 32, 32, 32, 32, { unclipRatio: -1 })).rejects.toThrow(/unclipRatio/)
    await expect(extractBoxes(map, 32, 32, 32, 32, { maxCandidates: 0 })).rejects.toThrow(/maxCandidates/)
    await expect(extractBoxes(map, 32, 32, 32, 32, { scoreMode: 'unknown' as 'fast' }))
      .rejects
      .toThrow(/scoreMode/)
  })

  it('supports contour-based slow scoring alongside upstream fast rectangle scoring', async () => {
    const map = new Float32Array(16 * 16)
    for (let y = 4; y < 8; y++) {
      for (let x = 4; x < 12; x++) map[y * 16 + x] = 0.9
    }
    for (let y = 8; y < 12; y++) {
      for (let x = 4; x < 8; x++) map[y * 16 + x] = 0.9
    }

    const options = { boxThreshold: 0, threshold: 0.5, unclipRatio: 0, useDilation: false }
    const fast = await extractBoxes(map, 16, 16, 16, 16, { ...options, scoreMode: 'fast' })
    const slow = await extractBoxes(map, 16, 16, 16, 16, { ...options, scoreMode: 'slow' })

    expect(fast).toHaveLength(1)
    expect(slow).toHaveLength(1)
    expect(slow[0]!.score).toBeGreaterThan(fast[0]!.score)
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

describe('getDetectorResizeRatio', () => {
  it('matches min-side and max-side detector scaling', () => {
    expect(getDetectorResizeRatio(400, 200, 736, 'min')).toBe(3.68)
    expect(getDetectorResizeRatio(1600, 800, 736, 'max')).toBe(0.46)
    expect(getDetectorResizeRatio(400, 200, 736, 'max')).toBe(1)
  })
})

describe('mapDetectorCoordinate', () => {
  it('matches upstream NumPy float32 scaling before ties-to-even rounding', () => {
    expect(mapDetectorCoordinate(247, 736, 368)).toBe(123)
    expect(mapDetectorCoordinate(248, 736, 368)).toBe(124)
  })
})
