/*
 * TypeScript adaptation of RapidOCR v3.10.0 CTC decoding behavior.
 * Copyright (c) RapidOCR Authors; Node.js changes copyright (c) 2026 OSpoon.
 * Licensed under Apache-2.0. See LICENSE.md and NOTICE.md.
 */
import type { RecognitionResult } from './types.js'

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
): RecognitionResult {
  const [batchCount, timeSteps, characterCount] = dimensions
  if (dimensions.length !== 3 || batchCount === undefined || timeSteps === undefined || characterCount === undefined) {
    throw new Error('Recognition model output must be a rank-three [batch, time, classes] tensor.')
  }

  if (data.length !== batchCount * timeSteps * characterCount) {
    throw new Error('Recognition model output dimensions do not match its float32 data buffer.')
  }

  if (characterCount !== characters.length) {
    throw new Error(
      `Recognition model has ${characterCount} classes, but its dictionary defines ${characters.length} classes.`,
    )
  }

  if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= batchCount) {
    throw new RangeError(`Recognition output has no batch item at index ${batchIndex}.`)
  }

  let text = ''
  let confidenceSum = 0
  let characterSteps = 0
  let previousIndex = -1
  const rowOffset = batchIndex * timeSteps * characterCount

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
    }

    previousIndex = bestIndex
  }

  const confidence = characterSteps === 0
    ? 0
    : Math.round((confidenceSum / characterSteps) * 100_000) / 100_000

  return { confidence, text }
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
