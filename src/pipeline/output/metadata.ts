import type { Buffer } from 'node:buffer'

export function addStageVisualization<T extends object>(
  result: T,
  render: (savePath?: string) => Promise<Buffer | undefined>,
): T & { vis: (savePath?: string) => Promise<Buffer | undefined> } {
  Object.defineProperty(result, 'vis', { value: render, enumerable: false })
  return result as T & { vis: (savePath?: string) => Promise<Buffer | undefined> }
}

export function addArrayMetadata<T extends object, M extends object>(result: T, metadata: M): T & M {
  Object.defineProperties(result, Object.fromEntries(
    Object.entries(metadata).map(([key, value]) => [key, { value, enumerable: false }]),
  ))
  return result as T & M
}
