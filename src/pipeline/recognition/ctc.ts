/*
 * TypeScript adaptation of RapidOCR v3.10.0 CTC decoding behavior.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { RecognitionResult, TextRecognitionCallOptions, WordPosition } from '../../shared/types.js'

const tokenConfidencesKey: unique symbol = Symbol('ctcTokenConfidences')
type WordPositionList = WordPosition[] & { [tokenConfidencesKey]?: readonly number[] }

/** Returns Python RapidOCR's per-decoded-token confidence sequence for word-box parity. */
export function getCtcTokenConfidences(positions: readonly WordPosition[]): readonly number[] | undefined {
  return (positions as WordPositionList)[tokenConfidencesKey]
}

interface CtcDecodeOptions extends TextRecognitionCallOptions {
  lineWidth?: number
}

export function readDictionary(contents: string): string[] {
  const lines = contents.replace(/^\uFEFF/, '').split(/\r?\n/)
  if (lines.at(-1) === '') {
    lines.pop()
  }

  if (lines.length === 0 || lines.some(character => character.length === 0)) {
    throw new Error('The recognition dictionary must contain one non-empty character per line.')
  }

  return ['blank', ...lines, ' ']
}

export function decodeCtcOutput(
  data: Float32Array,
  dimensions: readonly number[],
  batchIndex: number,
  characters: readonly string[],
  options: CtcDecodeOptions = {},
): RecognitionResult {
  const [batchCount, timeSteps, characterCount] = dimensions
  if (dimensions.length !== 3 || batchCount === undefined || timeSteps === undefined || characterCount === undefined) {
    throw new Error('Recognition model output must be a rank-three [batch, time, classes] tensor.')
  }

  if (data.length !== batchCount * timeSteps * characterCount) {
    throw new Error('Recognition model output dimensions do not match its float32 data buffer.')
  }

  if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= batchCount) {
    throw new RangeError(`Recognition output has no batch item at index ${batchIndex}.`)
  }

  let text = ''
  let confidenceSum = 0
  let characterSteps = 0
  let previousIndex = -1
  const rowOffset = batchIndex * timeSteps * characterCount
  const decodedTokens: Array<{ column: number, confidence: number, text: string }> = []

  for (let timeIndex = 0; timeIndex < timeSteps; timeIndex++) {
    const stepOffset = rowOffset + timeIndex * characterCount
    const { index: bestIndex, score: bestScore } = findBestClass(data, stepOffset, characterCount)

    if (bestIndex !== 0 && bestIndex !== previousIndex) {
      const character = characters[bestIndex]
      if (character === undefined) {
        throw new Error(`Recognition model emitted character index ${bestIndex}, missing from its dictionary.`)
      }

      text += character
      confidenceSum += bestScore
      characterSteps++
      decodedTokens.push({ column: timeIndex, confidence: bestScore, text: character })
    }

    previousIndex = bestIndex
  }

  const confidence = characterSteps === 0
    ? 0
    : Math.round((confidenceSum / characterSteps) * 100_000) / 100_000

  const result: RecognitionResult = { confidence, text }
  if (options.returnWordPositions) {
    result.wordPositions = getWordPositions(
      decodedTokens,
      options.lineWidth ?? timeSteps,
      options.returnSingleCharBoxes ?? false,
    )
  }
  return result
}

function getWordPositions(
  tokens: Array<{ column: number, confidence: number, text: string }>,
  lineWidth: number,
  returnSingleCharBoxes: boolean,
): WordPosition[] {
  if (tokens.length === 0)
    return []

  const allLatinOrNumeric = tokens.every(token => !isChinese(token.text))
  const groups: typeof tokens[] = []
  let group: typeof tokens = []
  for (const token of tokens) {
    if (/^\s+$/u.test(token.text)) {
      if (group.length > 0)
        groups.push(group)
      group = []
      continue
    }

    const previous = group.at(-1)
    if (previous && (isChinese(previous.text) !== isChinese(token.text) || token.column - previous.column > 5)) {
      groups.push(group)
      group = []
    }
    group.push(token)
  }
  if (group.length > 0)
    groups.push(group)

  const textLength = Math.max(1, tokens.filter(token => !/^\s+$/u.test(token.text)).length)
  const characterWidth = 1 / textLength
  const positions: WordPositionList = []
  for (const word of groups) {
    const segments = allLatinOrNumeric && !returnSingleCharBoxes
      ? [word]
      : word.map(token => [token])
    for (const segment of segments) {
      const first = segment[0]!
      const last = segment.at(-1)!
      const start = Math.max(0, (first.column + 0.5) / lineWidth - characterWidth / 2)
      const end = Math.min(1, (last.column + 0.5) / lineWidth + characterWidth / 2)
      const confidence = roundToFive(segment.reduce((sum, token) => sum + token.confidence, 0) / segment.length)
      positions.push({
        columns: segment.map(token => token.column),
        confidence,
        end,
        lineWidth,
        start,
        text: segment.map(token => token.text).join(''),
      })
    }
  }
  Object.defineProperty(positions, tokenConfidencesKey, {
    configurable: false,
    enumerable: false,
    value: tokens.map(token => roundToFive(token.confidence)),
    writable: false,
  })
  return positions
}

function isChinese(text: string): boolean {
  return /\p{Script=Han}/u.test(text)
}

function roundToFive(value: number): number {
  return Math.round(value * 100_000) / 100_000
}

function findBestClass(
  data: Float32Array,
  offset: number,
  characterCount: number,
): { index: number, score: number } {
  let index = 0
  let score = Number.NEGATIVE_INFINITY

  for (let characterIndex = 0; characterIndex < characterCount; characterIndex++) {
    const candidate = data[offset + characterIndex]
    if (candidate !== undefined && candidate > score) {
      index = characterIndex
      score = candidate
    }
  }

  return { index, score }
}
