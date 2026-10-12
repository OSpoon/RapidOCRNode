import type { DecodedRgbImage } from '../../image/image.js'
import type { RapidOCREmptyResult, RapidOCRExportMethods, RapidOCRFullResult, RapidOCRJsonLine, RecognizedLine, TextBox } from '../../shared/types.js'
import type { VisualizationSettings } from './visualization.js'
import { decodeVisualization } from './compat.js'
import { renderResult } from './visualization.js'

const NO_TEXT_MARKDOWN = '没有检测到任何文本。'

export function addOcrExportMethods<T extends Omit<RapidOCRFullResult, keyof RapidOCRExportMethods>>(
  result: T,
  image?: DecodedRgbImage,
  visualizationSettings: VisualizationSettings = {},
): T & RapidOCRExportMethods {
  Object.defineProperties(result, {
    toJSON: { value: () => linesToJson(result.lines), enumerable: false },
    to_json: { value: () => linesToJson(result.lines), enumerable: false },
    toMarkdown: { value: () => linesToMarkdown(result.lines), enumerable: false },
    to_markdown: { value: () => linesToMarkdown(result.lines), enumerable: false },
    vis: {
      value: async (savePath?: string) => {
        if (!image)
          return undefined
        const rendered = await renderResult(result.lines, image, visualizationSettings, savePath)
        return rendered ? decodeVisualization(rendered) : undefined
      },
      enumerable: false,
    },
    visBuffer: {
      value: (savePath?: string) => image ? renderResult(result.lines, image, visualizationSettings, savePath) : Promise.resolve(undefined),
      enumerable: false,
    },
  } satisfies PropertyDescriptorMap)
  return result as T & RapidOCRExportMethods
}

export function addEmptyExportMethods<T extends Omit<RapidOCREmptyResult, keyof RapidOCRExportMethods>>(
  result: T,
): T & RapidOCRExportMethods {
  Object.defineProperties(result, {
    toJSON: { value: () => null, enumerable: false },
    to_json: { value: () => null, enumerable: false },
    toMarkdown: { value: () => NO_TEXT_MARKDOWN, enumerable: false },
    to_markdown: { value: () => NO_TEXT_MARKDOWN, enumerable: false },
    vis: { value: async () => undefined, enumerable: false },
    visBuffer: { value: async () => undefined, enumerable: false },
  } satisfies PropertyDescriptorMap)
  return result as T & RapidOCRExportMethods
}

function linesToJson(lines: readonly RecognizedLine[]): RapidOCRJsonLine[] {
  return lines.map(line => ({
    box: line.box.map(point => [point.x, point.y]),
    txt: line.text,
    score: line.confidence,
  }))
}
export function linesToMarkdown(lines: readonly RecognizedLine[]): string {
  if (lines.length === 0)
    return ''

  const items = lines.map(line => ({ text: line.text, props: getBoxProperties(line.box) }))
  items.sort((a, b) => a.props.centerY - b.props.centerY || a.props.left - b.props.left)

  const groupedLines: Array<{ items: typeof items, props: BoxProperties }> = []
  for (const item of items) {
    const matchedLine = groupedLines.find(line => isSameLine(item.props, line.props))
    if (!matchedLine) {
      groupedLines.push({ items: [item], props: { ...item.props } })
      continue
    }

    matchedLine.items.push(item)
    matchedLine.props = mergeProperties(matchedLine.props, item.props)
  }

  groupedLines.sort((a, b) => a.props.top - b.props.top || a.props.left - b.props.left)
  const outputLines: string[] = []
  let previousLine: BoxProperties | undefined
  for (const line of groupedLines) {
    line.items.sort((a, b) => a.props.left - b.props.left)
    let text = line.items[0]!.text
    let previousItem = line.items[0]!.props
    for (const item of line.items.slice(1)) {
      text += getGapText(item.props.left - previousItem.right, previousItem, item.props)
      text += item.text
      previousItem = item.props
    }

    if (previousLine && line.props.top - previousLine.bottom > Math.max(previousLine.height, line.props.height) * 0.8)
      outputLines.push('')
    outputLines.push(text)
    previousLine = line.props
  }

  return outputLines.join('\n')
}

interface BoxProperties {
  bottom: number
  centerY: number
  height: number
  left: number
  right: number
  top: number
  width: number
}

function getBoxProperties(box: TextBox): BoxProperties {
  const xs = box.map(point => point.x)
  const ys = box.map(point => point.y)
  const top = Math.min(...ys)
  const bottom = Math.max(...ys)
  const left = Math.min(...xs)
  const right = Math.max(...xs)
  return { top, bottom, left, right, height: bottom - top, width: right - left, centerY: top + (bottom - top) / 2 }
}

function isSameLine(current: BoxProperties, line: BoxProperties): boolean {
  const overlap = Math.max(0, Math.min(line.bottom, current.bottom) - Math.max(line.top, current.top))
  const minimumHeight = Math.max(1, Math.min(current.height, line.height))
  const overlapRatio = overlap / minimumHeight
  const centerDifference = Math.abs(current.centerY - line.centerY)
  return overlapRatio > 0.5 || centerDifference < minimumHeight * 0.35
}

function mergeProperties(a: BoxProperties, b: BoxProperties): BoxProperties {
  const top = Math.min(a.top, b.top)
  const bottom = Math.max(a.bottom, b.bottom)
  const left = Math.min(a.left, b.left)
  const right = Math.max(a.right, b.right)
  return { top, bottom, left, right, height: bottom - top, width: right - left, centerY: top + (bottom - top) / 2 }
}

function getGapText(gap: number, previous: BoxProperties, current: BoxProperties): string {
  if (gap <= 1)
    return ''
  const referenceWidth = Math.max(1, Math.min(previous.width, current.width))
  const spaces = Math.max(1, Math.round(gap / Math.max(1, referenceWidth * 0.5)))
  return ' '.repeat(Math.min(spaces, 12))
}
