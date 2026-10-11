import type { RecognizedLine } from '../../src/shared/types.js'
import { describe, expect, it } from 'vitest'
import { addEmptyExportMethods, addOcrExportMethods, linesToMarkdown } from '../../src/pipeline/output/export.js'

function line(text: string, x: number, y: number, width = 10): RecognizedLine {
  return {
    box: [
      { x, y },
      { x: x + width, y },
      { x: x + width, y: y + 10 },
      { x, y: y + 10 },
    ],
    confidence: 0.95,
    text,
  }
}

describe('result export', () => {
  it('matches RapidOCR JSON line fields and keeps methods out of enumerable result data', () => {
    const result = addOcrExportMethods({
      type: 'ocr' as const,
      lines: [line('Hello', 1, 2)],
      img: { data: new Uint8Array(3), height: 1, width: 1 },
      elapse: 6,
      elapseList: [2, 1, 3],
      boxes: [line('Hello', 1, 2).box],
      txts: ['Hello'],
      scores: [0.95],
      wordResults: [[]],
      timings: { classification: 1, detection: 2, recognition: 3 },
    })

    expect(result.toJSON()).toEqual([{
      box: [[1, 2], [11, 2], [11, 12], [1, 12]],
      score: 0.95,
      txt: 'Hello',
    }])
    expect(JSON.stringify(result)).toBe(JSON.stringify(result.toJSON()))
    expect(Object.keys(result)).toEqual([
      'type',
      'lines',
      'img',
      'elapse',
      'elapseList',
      'boxes',
      'txts',
      'scores',
      'wordResults',
      'timings',
    ])
  })

  it('reconstructs ordered Markdown lines and paragraph breaks from boxes', () => {
    expect(linesToMarkdown([
      line('world', 12, 0),
      line('next', 0, 30),
      line('hello', 0, 1),
    ])).toBe('hello world\n\nnext')
  })

  it('returns upstream empty-result exports for an empty result', () => {
    const result = addEmptyExportMethods({
      type: 'empty' as const,
      lines: [] as [],
      img: { data: new Uint8Array(3), height: 1, width: 1 },
      elapse: 0,
      elapseList: [0, 0, 0],
      boxes: null,
      txts: null,
      scores: null,
      wordResults: [] as [],
      timings: { classification: 0, detection: 0, recognition: 0 },
    })

    expect(result.toJSON()).toBeNull()
    expect(result.toMarkdown()).toBe('没有检测到任何文本。')
  })
})
