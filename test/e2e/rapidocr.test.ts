import type { RapidOCRResult } from '../../src/types.js'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import sharp from 'sharp'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { TEST_FIXTURE_ASSETS } from '../../model-assets.mjs'
import { downloadVerifiedAsset } from '../../src/models.js'

type PackageApi = typeof import('../../src/index.js')
type RapidOCRInstance = Awaited<ReturnType<PackageApi['RapidOCR']['create']>>
type TextRecognizerInstance = Awaited<ReturnType<PackageApi['TextRecognizer']['create']>>

interface ExpectedLine {
  box: number[][]
  confidence: number
  text: string
}

interface ExpectedOutput {
  lines: ExpectedLine[]
  upstream: string
}

const modelsDirectory = resolve('models')
const packageName: string = 'rapidocr-node'
const samplePath = resolve(modelsDirectory, 'ch_en_num.jpg')
const expected = JSON.parse(await readFile(new URL('../fixtures/ch_en_num.expected.json', import.meta.url), 'utf8')) as ExpectedOutput
const lowResolutionExpected = JSON.parse(await readFile(new URL('../fixtures/ch_en_num_162x215.expected.json', import.meta.url), 'utf8')) as { lines: ExpectedLine[] }
const documentExpected = JSON.parse(await readFile(new URL('../fixtures/ch_doc_server.expected.json', import.meta.url), 'utf8')) as ExpectedOutput
let ocr: RapidOCRInstance | undefined
let recognizer: TextRecognizerInstance | undefined

beforeAll(async () => {
  const api = await import(packageName) as PackageApi
  const modelPaths = await api.downloadModels({ modelDirectory: modelsDirectory })
  await Promise.all(TEST_FIXTURE_ASSETS.map(asset => downloadVerifiedAsset(asset, modelsDirectory)))
  ocr = await api.RapidOCR.create({
    ...modelPaths,
    intraOpNumThreads: 1,
  })
  recognizer = await api.TextRecognizer.create({
    modelPath: modelPaths.recognitionModelPath,
    dictionaryPath: modelPaths.dictionaryPath,
    intraOpNumThreads: 1,
  })
})

afterAll(async () => {
  await ocr?.close()
  await recognizer?.close()
})

describe('rapidocr with pinned upstream models', () => {
  it('matches the upstream sample text, reading order, boxes, and confidence within tolerance', async () => {
    const result = await ocr!.recognize(samplePath)
    expectLinesMatchReference(result.lines, expected.lines)
    expectTimings(result)
  })

  it('matches a second upstream document page, including a box at the image edge', async () => {
    const result = await ocr!.recognize(resolve(modelsDirectory, 'ch_doc_server.png'))

    expectLinesMatchReference(result.lines, documentExpected.lines)
  })

  it('matches the upstream low-resolution regression after ties-to-even detector resizing', async () => {
    const lowResolution = await sharp(samplePath)
      .resize(162, 215, { fit: 'fill' })
      .jpeg({ quality: 80 })
      .toBuffer()
    const result = await ocr!.recognize(lowResolution)

    expectLinesMatchReference(result.lines, lowResolutionExpected.lines, 0.04)
  })

  it('accepts encoded image bytes and reuses the loaded model sessions', async () => {
    const bytes = await readFile(samplePath)
    const result = await ocr!.recognize(bytes)

    expectLinesMatchReference(result.lines, expected.lines)
    expectTimings(result)
  })

  it('classifies an upside-down text line before recognition when detection is disabled', async () => {
    const result = await ocr!.recognize(resolve(modelsDirectory, 'text_cls.jpg'), { useDetection: false })

    expect(result.lines).toHaveLength(1)
    expect(result.lines[0]!.text).toBe('怪我咯')
    expect(result.timings.detection).toBe(0)
    expect(result.timings.classification).toBeGreaterThan(0)
  })

  it('recognizes an already-cropped line with the real PP-OCRv6 model', async () => {
    const result = await recognizer!.recognizeLine(resolve(modelsDirectory, 'text_rec.jpg'))

    expect(result.text).toBe('韩国小馆')
    expect(result.confidence).toBeGreaterThan(0.95)
  })

  it('maps boxes back to source coordinates after maximum-side image resizing', async () => {
    const enlarged = await sharp(samplePath).resize(1615, 2150).jpeg().toBuffer()
    const result = await ocr!.recognize(enlarged)

    expect(result.lines).toHaveLength(expected.lines.length)
    expect(result.lines[0]!.text).toBe('正品促销')
    expect(result.lines.at(-1)!.text).toBe('假一赔十')
    expect(result.lines[1]!.box[0].x).toBeGreaterThan(300)
    expect(result.lines[1]!.box[0].x).toBeLessThan(360)
    expect(result.lines[1]!.box[0].y).toBeGreaterThan(430)
    expect(result.lines[1]!.box[0].y).toBeLessThan(500)
  })

  it('returns no text lines for a blank raw RGB image', async () => {
    const result = await ocr!.recognize({
      data: new Uint8Array(320 * 320 * 3).fill(255),
      height: 320,
      width: 320,
    })

    expect(result.lines).toEqual([])
    expect(result.timings.classification).toBe(0)
    expect(result.timings.recognition).toBe(0)
  })

  it('waits for active inference before closing and rejects new work afterward', async () => {
    const pending = ocr!.recognize(samplePath)
    const closing = ocr!.close()

    await expect(pending).resolves.toHaveProperty('lines.length', expected.lines.length)
    await expect(closing).resolves.toBeUndefined()
    await expect(ocr!.close()).resolves.toBeUndefined()
    await expect(ocr!.recognize(samplePath)).rejects.toThrow(/closed/)
  })
})

function expectTimings(result: RapidOCRResult): void {
  expect(result.timings.detection).toBeGreaterThanOrEqual(0)
  expect(result.timings.classification).toBeGreaterThanOrEqual(0)
  expect(result.timings.recognition).toBeGreaterThanOrEqual(0)
  expect(Object.values(result.timings).every(Number.isFinite)).toBe(true)
}

function expectLinesMatchReference(
  actualLines: RapidOCRResult['lines'],
  referenceLines: ExpectedLine[],
  confidenceTolerance = 0.03,
): void {
  expect(actualLines.map(line => normalizeCurrencyGlyph(line.text)))
    .toEqual(referenceLines.map(line => normalizeCurrencyGlyph(line.text)))
  expect(actualLines).toHaveLength(referenceLines.length)

  for (const [lineIndex, actual] of actualLines.entries()) {
    const reference = referenceLines[lineIndex]!
    expect(Math.abs(actual.confidence - reference.confidence)).toBeLessThanOrEqual(confidenceTolerance)
    for (const [pointIndex, point] of actual.box.entries()) {
      const expectedPoint = reference.box[pointIndex]!
      expect(Math.abs(point.x - expectedPoint[0]!)).toBeLessThanOrEqual(2)
      expect(Math.abs(point.y - expectedPoint[1]!)).toBeLessThanOrEqual(2)
    }
  }
}

function normalizeCurrencyGlyph(text: string): string {
  return text.replaceAll('￥', '¥')
}
