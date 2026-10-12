import type { RapidOCRBox, RapidOCRImage, RapidOCRWordInfo, RapidOCRWordResult, RawRgbImage, RecognizedLine, TextBox, WordPosition } from '../../shared/types.js'
import type { Buffer } from 'node:buffer'
import sharp from 'sharp'
import { getCtcTokenConfidences } from '../recognition/ctc.js'

/** Converts the internal RGB image to RapidOCR's OpenCV-compatible BGR result format. */
export function toRapidOCRImage(image: RawRgbImage): RapidOCRImage {
  const data = new Uint8Array(image.data.length)
  for (let source = 0; source < image.data.length; source += 3) {
    data[source] = image.data[source + 2] ?? 0
    data[source + 1] = image.data[source + 1] ?? 0
    data[source + 2] = image.data[source] ?? 0
  }
  return { data, format: 'BGR', height: image.height, width: image.width }
}

function toRapidOCRBox(box: TextBox): RapidOCRBox {
  return box.map(point => [point.x, point.y] as [number, number]) as unknown as RapidOCRBox
}

export function toRapidOCRBoxes(boxes: readonly TextBox[]): RapidOCRBox[] {
  return boxes.map(toRapidOCRBox)
}

export function toRapidOCRWordResults(lines: readonly RecognizedLine[]): Array<RapidOCRWordResult[] | null> {
  return lines.map((line) => {
    if (line.words === undefined)
      return null
    return line.words.map(word => [word.text, word.confidence, toRapidOCRBox(word.box)] as const)
  })
}

export function toRapidOCRWordInfo(positions: readonly WordPosition[] | undefined): RapidOCRWordInfo | null {
  if (positions === undefined)
    return null

  const words: string[][] = []
  const wordCols: number[][] = []
  const wordTypes: RapidOCRWordInfo['word_types'] = []
  let previousColumn = -Infinity
  for (const position of positions) {
    const type = /\p{Script=Han}/u.test(position.text) ? 'cn' : 'en&num'
    const firstColumn = position.columns[0] ?? Number.POSITIVE_INFINITY
    const previousType = wordTypes.at(-1)
    if (previousType !== type || firstColumn - previousColumn > 5) {
      words.push([])
      wordCols.push([])
      wordTypes.push(type)
    }
    words.at(-1)!.push(...[...position.text])
    wordCols.at(-1)!.push(...position.columns)
    previousColumn = position.columns.at(-1) ?? previousColumn
  }

  return {
    words,
    word_cols: wordCols,
    word_types: wordTypes,
    line_txt_len: positions[0]?.lineWidth ?? 0,
    confs: [...(getCtcTokenConfidences(positions) ?? positions.flatMap(position => Array.from(position.text, () => position.confidence)))],
  }
}

export async function decodeVisualization(buffer: Buffer): Promise<RapidOCRImage> {
  const { data, info } = await sharp(buffer)
    .removeAlpha()
    .toColourspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true })
  if (info.channels !== 3)
    throw new TypeError(`Visualization image must have three channels; received ${info.channels}.`)
  return toRapidOCRImage({ data, height: info.height, width: info.width })
}
