import type { RapidOCRBox, RecognizedLine } from '../../src/shared/types.js'
import { readFile } from 'node:fs/promises'
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
      img: { data: new Uint8Array(3), format: 'BGR' as const, height: 1, width: 1 },
      elapse: 6,
      elapseList: [2, 1, 3],
      elapse_list: [2, 1, 3],
      boxes: [[[1, 2], [11, 2], [11, 12], [1, 12]]],
      boxPoints: [line('Hello', 1, 2).box],
      txts: ['Hello'],
      scores: [0.95],
      wordResults: [[]],
      word_results: [null],
      lang_rec: 'ch_small',
      timings: { classification: 1, detection: 2, recognition: 3 },
    })

    expect(result.toJSON()).toEqual([{
      box: [[1, 2], [11, 2], [11, 12], [1, 12]],
      score: 0.95,
      txt: 'Hello',
    }])
    expect(result.to_json()).toEqual(result.toJSON())
    expect(JSON.stringify(result)).toBe(JSON.stringify(result.toJSON()))
    expect(Object.keys(result)).toEqual([
      'type',
      'lines',
      'img',
      'elapse',
      'elapseList',
      'elapse_list',
      'boxes',
      'boxPoints',
      'txts',
      'scores',
      'wordResults',
      'word_results',
      'lang_rec',
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

  it('reproduces the upstream sample JSON and Markdown assertions from RapidOCR v3.10.0', async () => {
    const reference = JSON.parse(await readFile(new URL('../fixtures/ch_en_num.expected.json', import.meta.url), 'utf8')) as {
      lines: Array<{ box: number[][], confidence: number, text: string }>
    }
    const lines = reference.lines.map(({ box, confidence, text }) => ({
      box: [
        { x: box[0]![0]!, y: box[0]![1]! },
        { x: box[1]![0]!, y: box[1]![1]! },
        { x: box[2]![0]!, y: box[2]![1]! },
        { x: box[3]![0]!, y: box[3]![1]! },
      ],
      confidence,
      text,
    })) as RecognizedLine[]
    const result = addOcrExportMethods({
      type: 'ocr' as const,
      lines,
      img: { data: new Uint8Array(3), format: 'BGR' as const, height: 1, width: 1 },
      elapse: 0,
      elapseList: [0, 0, 0] as [number, number, number],
      elapse_list: [0, 0, 0],
      boxes: lines.map((line) => {
        const [topLeft, topRight, bottomRight, bottomLeft] = line.box
        return [
          [topLeft.x, topLeft.y],
          [topRight.x, topRight.y],
          [bottomRight.x, bottomRight.y],
          [bottomLeft.x, bottomLeft.y],
        ] as RapidOCRBox
      }),
      boxPoints: lines.map(line => line.box),
      txts: lines.map(line => line.text),
      scores: lines.map(line => line.confidence),
      wordResults: lines.map(() => []),
      word_results: lines.map(() => null),
      lang_rec: 'ch_small',
      timings: { classification: 0, detection: 0, recognition: 0 },
    })

    expect(result.to_json()?.[0]?.txt).toBe('正品促销')
    expect(result.to_markdown().split('\n')).toHaveLength(11)
  })

  it('returns upstream empty-result exports for an empty result', () => {
    const result = addEmptyExportMethods({
      type: 'empty' as const,
      lines: [] as [],
      img: null,
      elapse: 0,
      elapseList: [0, 0, 0],
      elapse_list: [],
      boxes: null,
      boxPoints: null,
      txts: null,
      scores: null,
      wordResults: [] as [],
      word_results: [] as [],
      timings: { classification: 0, detection: 0, recognition: 0 },
    })

    expect(result.toJSON()).toBeNull()
    expect(result.toMarkdown()).toBe('没有检测到任何文本。')
  })
})
