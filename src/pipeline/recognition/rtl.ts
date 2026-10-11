import bidiFactory from 'bidi-js'

const bidi = bidiFactory()

/** Reorders RTL text into the visual string form returned by upstream RapidOCR. */
export function reorderTextForDisplay(text: string): string {
  const levels = bidi.getEmbeddingLevels(text)
  return bidi.getReorderedString(text, levels)
}
