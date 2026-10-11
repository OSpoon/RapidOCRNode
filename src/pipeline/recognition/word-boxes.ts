import type { RecognizedWord, TextBox, WordPosition } from '../../shared/types.js'
import { getCtcTokenConfidences } from './ctc.js'

/** Maps CTC time-step positions back through the detector's perspective crop. */
export function mapWordPositionsToBox(
  detectionBox: TextBox,
  cropWidth: number,
  cropHeight: number,
  text: string,
  positions: readonly WordPosition[],
  rotated180 = false,
): RecognizedWord[] {
  if (positions.length === 0 || cropWidth < 1 || cropHeight < 1)
    return []

  const lineWidth = positions[0]!.lineWidth
  if (lineWidth <= 0)
    return []

  const tokenConfidences = getCtcTokenConfidences(positions)
  const vertical = isVerticalBox(detectionBox)
  const rectWidth = Math.max(1, Math.trunc(distance(detectionBox[0], detectionBox[1])))
  const rectHeight = Math.max(1, Math.trunc(distance(detectionBox[0], detectionBox[3])))
  const averageColumnWidth = cropWidth / lineWidth
  const positionGroups: WordPosition[][] = []
  for (const position of positions) {
    const currentGroup = positionGroups.at(-1)
    const previousPosition = currentGroup?.at(-1)
    const sameScript = previousPosition && containsHan(previousPosition.text) === containsHan(position.text)
    const adjacent = previousPosition && position.columns[0]! - previousPosition.columns.at(-1)! <= 5
    if (currentGroup && sameScript && adjacent) {
      currentGroup.push(position)
    }
    else {
      positionGroups.push([position])
    }
  }
  const widths = positionGroups.flatMap((group) => {
    const columns = group.flatMap(position => position.columns)
    return columns.length > 1
      ? [(columns.at(-1)! - columns[0]!) * averageColumnWidth / (columns.length - 1)]
      : []
  })
  const averageCharacterWidth = widths.length > 0
    ? widths.reduce((sum, width) => sum + width, 0) / widths.length
    : cropWidth / Math.max(1, [...text].length)
  const allLatinOrNumeric = positions.every(position => !containsHan(position.text))

  const cells = positions.flatMap((position) => {
    const characterCells = position.columns.map((column) => {
      const center = (column + 0.5) * averageColumnWidth
      return {
        left: Math.max(Math.trunc(center - averageCharacterWidth / 2), 0),
        right: Math.min(Math.trunc(center + averageCharacterWidth / 2), cropWidth),
      }
    })

    if (allLatinOrNumeric && characterCells.length > 1) {
      return [{
        confidence: position.confidence,
        left: Math.min(...characterCells.map(cell => cell.left)),
        right: Math.max(...characterCells.map(cell => cell.right)),
        text: position.text,
      }]
    }

    return characterCells.map((cell, index) => ({
      confidence: position.confidence,
      ...cell,
      text: allLatinOrNumeric ? [...position.text][index] ?? position.text : position.text,
    }))
  }).sort((left, right) => left.left - right.left)

  for (let index = 0; index < cells.length - 1; index++) {
    const current = cells[index]!
    const next = cells[index + 1]!
    if (current.right > next.left) {
      const overlap = current.right - next.left
      current.right -= overlap / 2
      next.left += overlap - overlap / 2
    }
  }

  const transform = getRectToQuadTransform(rectWidth, rectHeight, detectionBox)
  return cells.map((cell, index) => {
    const localBox = [
      { x: cell.left, y: 0 },
      { x: cell.right, y: 0 },
      { x: cell.right, y: cropHeight },
      { x: cell.left, y: cropHeight },
    ]
    const box = localBox.map((point) => {
      let x = point.x
      let y = point.y
      if (rotated180) {
        x = cropWidth - x
        y = cropHeight - y
      }
      if (vertical) {
        const rotatedX = rectWidth - y
        y = x
        x = rotatedX
      }
      const mapped = projectPoint(x, y, transform)
      return { x: Math.trunc(mapped.x), y: Math.trunc(mapped.y) }
    }) as unknown as TextBox

    return {
      box,
      confidence: tokenConfidences?.[index] ?? cell.confidence,
      text: cell.text,
    }
  })
}

function isVerticalBox(box: TextBox): boolean {
  const width = Math.max(distance(box[0], box[1]), distance(box[2], box[3]))
  const height = Math.max(distance(box[0], box[3]), distance(box[1], box[2]))
  if (width < 1e-6)
    return true
  return Math.round((height / width) * 100) / 100 >= 1.5
}

function containsHan(text: string): boolean {
  return /\p{Script=Han}/u.test(text)
}

type Homography = readonly [number, number, number, number, number, number, number, number, number]

function getRectToQuadTransform(width: number, height: number, box: TextBox): Homography {
  const source = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ]
  const system: number[][] = []
  for (let index = 0; index < 4; index++) {
    const from = source[index]!
    const to = box[index]!
    system.push([from.x, from.y, 1, 0, 0, 0, -to.x * from.x, -to.x * from.y, to.x])
    system.push([0, 0, 0, from.x, from.y, 1, -to.y * from.x, -to.y * from.y, to.y])
  }

  const values = solveLinearSystem(system)
  return [values[0]!, values[1]!, values[2]!, values[3]!, values[4]!, values[5]!, values[6]!, values[7]!, 1]
}

function solveLinearSystem(matrix: number[][]): number[] {
  for (let column = 0; column < matrix.length; column++) {
    let pivot = column
    for (let row = column + 1; row < matrix.length; row++) {
      if (Math.abs(matrix[row]![column]!) > Math.abs(matrix[pivot]![column]!))
        pivot = row
    }
    const pivotValue = matrix[pivot]![column]!
    if (Math.abs(pivotValue) < 1e-12) {
      throw new RangeError('Cannot map word boxes through a degenerate detection quadrilateral.')
    }

    ;[matrix[column], matrix[pivot]] = [matrix[pivot]!, matrix[column]!]
    for (let entry = column; entry <= matrix.length; entry++)
      matrix[column]![entry] = matrix[column]![entry]! / pivotValue

    for (let row = 0; row < matrix.length; row++) {
      if (row === column) {
        continue
      }
      const factor = matrix[row]![column]!
      for (let entry = column; entry <= matrix.length; entry++)
        matrix[row]![entry] = matrix[row]![entry]! - factor * matrix[column]![entry]!
    }
  }
  return matrix.map(row => row.at(-1)!)
}

function projectPoint(x: number, y: number, transform: Homography): { x: number, y: number } {
  const denominator = transform[6] * x + transform[7] * y + transform[8]
  return {
    x: (transform[0] * x + transform[1] * y + transform[2]) / denominator,
    y: (transform[3] * x + transform[4] * y + transform[5]) / denominator,
  }
}

function distance(start: TextBox[number], end: TextBox[number]): number {
  return Math.hypot(end.x - start.x, end.y - start.y)
}
