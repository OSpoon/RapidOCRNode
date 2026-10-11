import type { RecognizedLine } from '../../src/shared/types.js'
import { expect } from 'vitest'

export interface WordBoxReference {
  box: number[][]
  confidence: number
  text: string
}

export interface WordBoxLineReference {
  text: string
  words?: WordBoxReference[]
}

export function expectWordBoxesMatchReference(
  actualLines: readonly RecognizedLine[],
  expectedLines: readonly WordBoxLineReference[],
): void {
  expect(actualLines.map(line => line.text)).toEqual(expectedLines.map(line => line.text))
  expect(actualLines).toHaveLength(expectedLines.length)

  for (const [lineIndex, actualLine] of actualLines.entries()) {
    const expectedLine = expectedLines[lineIndex]!
    expect(expectedLine.words?.length, `reference line ${lineIndex} word boxes`).toBeGreaterThan(0)
    expect(actualLine.words, `line ${lineIndex} word boxes`).toHaveLength(expectedLine.words!.length)
    for (const [wordIndex, actualWord] of actualLine.words!.entries()) {
      const expectedWord = expectedLine.words![wordIndex]!
      expect(actualWord.text, `line ${lineIndex} word ${wordIndex}`).toBe(expectedWord.text)
      expect(Math.abs(actualWord.confidence - expectedWord.confidence), `line ${lineIndex} word ${wordIndex} confidence`)
        .toBeLessThanOrEqual(0.04)
      expect(actualWord.box).toHaveLength(expectedWord.box.length)
      for (const [pointIndex, point] of actualWord.box.entries()) {
        expect(Math.abs(point.x - expectedWord.box[pointIndex]![0]!), `line ${lineIndex} word ${wordIndex} point ${pointIndex} x: ${point.x} vs ${expectedWord.box[pointIndex]![0]}`)
          .toBeLessThanOrEqual(2)
        expect(Math.abs(point.y - expectedWord.box[pointIndex]![1]!), `line ${lineIndex} word ${wordIndex} point ${pointIndex} y: ${point.y} vs ${expectedWord.box[pointIndex]![1]}`)
          .toBeLessThanOrEqual(2)
      }
    }
  }
}
