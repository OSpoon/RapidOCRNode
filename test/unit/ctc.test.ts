import { describe, expect, it } from 'vitest'
import { decodeCtcOutput, readDictionary } from '../../src/pipeline/recognition/ctc.js'

describe('readDictionary', () => {
  it('adds the CTC blank and space entries and accepts a UTF-8 BOM', () => {
    expect(readDictionary('\uFEFF你\r\n好\r\n')).toEqual(['blank', '你', '好', ' '])
  })

  it('rejects empty dictionaries and empty entries', () => {
    expect(() => readDictionary('')).toThrow(/one non-empty character per line/)
    expect(() => readDictionary('你\n\n好')).toThrow(/one non-empty character per line/)
  })
})

describe('decodeCtcOutput', () => {
  it('collapses repeated characters separated by blanks and averages emitted scores', () => {
    const output = new Float32Array([
      0.9,
      0.1,
      0.0,
      0.1,
      0.8,
      0.1,
      0.2,
      0.7,
      0.1,
      0.8,
      0.1,
      0.1,
      0.0,
      0.05,
      0.95,
    ])

    expect(decodeCtcOutput(output, [1, 5, 3], 0, ['blank', 'A', 'B']))
      .toEqual({ confidence: 0.875, text: 'AB' })
  })

  it('returns an empty result for blank-only output', () => {
    expect(decodeCtcOutput(new Float32Array([1, 0, 1, 0]), [1, 2, 2], 0, ['blank', 'A']))
      .toEqual({ confidence: 0, text: '' })
  })

  it('matches the upstream decoder when an unused model class has no dictionary entry', () => {
    expect(decodeCtcOutput(new Float32Array([0, 0.9, 0.1]), [1, 1, 3], 0, ['blank', 'A']))
      .toEqual({ confidence: 0.9, text: 'A' })
    expect(() => decodeCtcOutput(new Float32Array([0, 0.1, 0.9]), [1, 1, 3], 0, ['blank', 'A']))
      .toThrow(/emitted character index 2, missing from its dictionary/)
  })

  it('returns normalized word positions only when requested', () => {
    const output = new Float32Array([
      0.9,
      0.1,
      0.0,
      0.1,
      0.8,
      0.1,
      0.2,
      0.7,
      0.1,
      0.8,
      0.1,
      0.1,
      0.0,
      0.05,
      0.95,
    ])

    const wordResult = decodeCtcOutput(output, [1, 5, 3], 0, ['blank', 'A', 'B'], {
      returnWordPositions: true,
    })
    expect(wordResult.text).toBe('AB')
    expect(wordResult.wordPositions).toHaveLength(1)
    expect(wordResult.wordPositions?.[0]).toMatchObject({ text: 'AB', end: 1, confidence: 0.875 })
    expect(wordResult.wordPositions?.[0]?.start).toBeCloseTo(0.05)

    const characterResult = decodeCtcOutput(output, [1, 5, 3], 0, ['blank', 'A', 'B'], {
      returnSingleCharBoxes: true,
      returnWordPositions: true,
    })
    expect(characterResult.wordPositions?.map(position => position.text)).toEqual(['A', 'B'])
    expect(decodeCtcOutput(output, [1, 5, 3], 0, ['blank', 'A', 'B']).wordPositions).toBeUndefined()
  })

  it('validates tensor dimensions and batch index', () => {
    expect(() => decodeCtcOutput(new Float32Array(3), [1, 1], 0, ['blank', 'A']))
      .toThrow(/rank-three/)
    expect(() => decodeCtcOutput(new Float32Array(2), [1, 1, 3], 0, ['blank', 'A']))
      .toThrow(/dimensions do not match/)
    expect(() => decodeCtcOutput(new Float32Array(2), [1, 1, 2], 1, ['blank', 'A']))
      .toThrow(/no batch item/)
  })
})
