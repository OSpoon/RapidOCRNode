import type { DecodedRgbImage } from '../../image/image.js'
import type { RapidOCROptions, RecognizedLine, RecognizedWord, TextBox } from '../../shared/types.js'
import { Buffer } from 'node:buffer'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import sharp from 'sharp'
import { loadVisualizationFont } from './fonts.js'

export type VisualizationSettings = Pick<RapidOCROptions, 'autoDownloadFonts' | 'fontPath' | 'modelDirectory'> & { langType?: string }

export async function renderDetectionStage(
  image: DecodedRgbImage,
  boxes: readonly TextBox[],
  scores: readonly number[],
  savePath?: string,
): Promise<Buffer | undefined> {
  if (boxes.length === 0)
    return undefined

  const annotations = boxes.map((box, index) => {
    const points = box.map(point => `${svgNumber(point.x)},${svgNumber(point.y)}`).join(' ')
    const color = `hsl(${(index * 137.508) % 360}, 78%, 42%)`
    const first = box[0]
    const score = scores[index]
    return `<polygon points="${points}" fill="none" stroke="${color}" stroke-width="2"/><text x="${svgNumber(first.x)}" y="${svgNumber(Math.max(12, first.y - 2))}" font-size="12" fill="${color}">${index}:${(score ?? 0).toFixed(3)}</text>`
  }).join('')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${image.width}" height="${image.height}">${annotations}</svg>`
  const rendered = await sharp(Buffer.from(image.data), {
    raw: { width: image.width, height: image.height, channels: 3 },
  })
    .composite([{ input: Buffer.from(svg) }])
    .png()
    .toBuffer()
  return saveVisualization(rendered, savePath)
}

export async function renderLineStage(
  images: readonly DecodedRgbImage[],
  texts: readonly string[],
  scores: readonly number[],
  settings: VisualizationSettings,
  savePath?: string,
): Promise<Buffer | undefined> {
  if (images.length === 0)
    return undefined

  const font = await loadVisualizationFont({
    autoDownload: settings.autoDownloadFonts,
    fontPath: settings.fontPath,
    langType: settings.langType,
    modelDirectory: settings.modelDirectory,
  })
  const rows = await Promise.all(images.map(async (image, index) => {
    const encoded = await sharp(Buffer.from(image.data), {
      raw: { width: image.width, height: image.height, channels: 3 },
    }).png().toBuffer()
    return {
      dataUrl: `data:image/png;base64,${encoded.toString('base64')}`,
      height: Math.max(image.height, 28),
      imageHeight: image.height,
      imageWidth: image.width,
      score: scores[index] ?? 0,
      text: texts[index] ?? '',
    }
  }))
  const width = Math.max(...rows.map(row => row.imageWidth)) * 2
  const height = rows.reduce((total, row) => total + row.height, 0)
  let top = 0
  const body = rows.map((row) => {
    const imageTop = top + Math.trunc((row.height - row.imageHeight) / 2)
    const fontSize = Math.min(32, Math.max(10, Math.trunc(row.height * 0.75)))
    const text = `${row.text} ${row.score.toFixed(2)}`
    const item = `<image x="0" y="${imageTop}" width="${row.imageWidth}" height="${row.imageHeight}" href="${row.dataUrl}"/><text x="${width / 2 + 5}" y="${top + Math.trunc((row.height + fontSize * 0.7) / 2)}" font-size="${fontSize}">${escapeXml(text)}</text>`
    top += row.height
    return item
  }).join('')
  const fontBase64 = Buffer.from(font).toString('base64')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><style>@font-face{font-family:RapidOCR;src:url(data:font/ttf;base64,${fontBase64})}text{font-family:RapidOCR,sans-serif;fill:#111}</style><rect width="100%" height="100%" fill="#fff"/>${body}</svg>`
  const rendered = await sharp({
    create: { width, height, channels: 3, background: '#ffffff' },
  })
    .composite([{ input: Buffer.from(svg) }])
    .png()
    .toBuffer()
  return saveVisualization(rendered, savePath)
}
export async function renderResult(
  lines: readonly RecognizedLine[],
  image: DecodedRgbImage,
  options: VisualizationSettings,
  savePath?: string,
): Promise<Buffer | undefined> {
  if (lines.length === 0)
    return undefined

  const font = await loadVisualizationFont({
    autoDownload: options.autoDownloadFonts,
    fontPath: options.fontPath,
    langType: options.langType,
    modelDirectory: options.modelDirectory,
  })
  const svg = buildVisualizationSvg(lines, image.width, image.height, font)
  const rendered = await sharp(Buffer.from(image.data), {
    raw: { width: image.width, height: image.height, channels: 3 },
  })
    .extend({ right: image.width, background: '#ffffff' })
    .composite([{ input: Buffer.from(svg) }])
    .png()
    .toBuffer()
  return saveVisualization(rendered, savePath)
}

async function saveVisualization(rendered: Buffer, savePath?: string): Promise<Buffer> {
  if (savePath) {
    const outputPath = resolve(savePath)
    await mkdir(dirname(outputPath), { recursive: true })
    await writeFile(outputPath, rendered)
  }
  return rendered
}

function buildVisualizationSvg(
  lines: readonly RecognizedLine[],
  width: number,
  height: number,
  font: Uint8Array,
): string {
  const displayItems: Array<RecognizedLine | RecognizedWord> = lines.every(line => line.words?.length)
    ? lines.flatMap(line => line.words ?? [])
    : [...lines]
  const fontBase64 = Buffer.from(font).toString('base64')
  const polygons = displayItems.map((item, index) => {
    const points = item.box.map(point => `${svgNumber(point.x)},${svgNumber(point.y)}`).join(' ')
    const color = `hsl(${(index * 137.508) % 360}, 78%, 52%)`
    const [topLeft, topRight, , bottomLeft] = item.box
    const boxWidth = Math.hypot(topRight!.x - topLeft!.x, topRight!.y - topLeft!.y)
    const boxHeight = Math.hypot(bottomLeft!.x - topLeft!.x, bottomLeft!.y - topLeft!.y)
    const fontSize = Math.max(10, Math.trunc(boxHeight * 0.8))
    const text = boxHeight > boxWidth * 2
      ? [...item.text].map((character, characterIndex) => `<text x="${svgNumber(width + topLeft!.x + 3)}" y="${svgNumber(topLeft!.y + fontSize * (characterIndex + 1))}" font-size="${fontSize}">${escapeXml(character)}</text>`).join('')
      : `<text x="${svgNumber(width + topLeft!.x)}" y="${svgNumber(topLeft!.y + fontSize)}" font-size="${fontSize}">${escapeXml(item.text)}</text>`
    return `<polygon points="${points}" fill="${color}" fill-opacity="0.5"/><polygon points="${points.split(' ').map((point) => {
      const [x, y] = point!.split(',')
      return `${svgNumber(Number(x) + width)},${y}`
    }).join(' ')}" fill="none" stroke="${color}" stroke-width="1"/>${text}`
  }).join('')

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width * 2}" height="${height}" viewBox="0 0 ${width * 2} ${height}"><style>@font-face{font-family:RapidOCR;src:url(data:font/ttf;base64,${fontBase64})}text{font-family:RapidOCR,sans-serif;fill:#111}</style>${polygons}</svg>`
}

function svgNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

function escapeXml(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll('\'', '&apos;')
}
