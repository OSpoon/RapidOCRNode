import type { Buffer } from 'node:buffer'
import type { RapidOCRImage } from '../../shared/types.js'
import { decodeVisualization } from './compat.js'

export function addStageVisualization<T extends object>(
  result: T,
  render: (savePath?: string) => Promise<Buffer | undefined>,
): T & { vis: (savePath?: string) => Promise<RapidOCRImage | undefined>, visBuffer: (savePath?: string) => Promise<Buffer | undefined> } {
  Object.defineProperties(result, {
    vis: {
      value: async (savePath?: string) => {
        const buffer = await render(savePath)
        return buffer === undefined ? undefined : decodeVisualization(buffer)
      },
      enumerable: false,
    },
    visBuffer: { value: render, enumerable: false },
  })
  return result as T & { vis: (savePath?: string) => Promise<RapidOCRImage | undefined>, visBuffer: (savePath?: string) => Promise<Buffer | undefined> }
}

export function addArrayMetadata<T extends object, M extends object>(result: T, metadata: M): T & M {
  Object.defineProperties(result, Object.fromEntries(
    Object.entries(metadata).map(([key, value]) => [key, { value, enumerable: false }]),
  ))
  return result as T & M
}
