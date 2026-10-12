import { arch } from 'node:process'

export interface KnownRuntimeTextDifference {
  confidenceTolerance: number
  fixture?: string
  index: number
  nodeText: string
  upstreamText: string
}

const x64RecognitionFullPageTextDifferences: Record<string, readonly KnownRuntimeTextDifference[]> = {
  'chinese_cht_PP-OCRv3_rec_mobile': [
    {
      confidenceTolerance: 0.06,
      fixture: 'ch_en_num-sharp.png',
      index: 4,
      nodeText: '1o08',
      upstreamText: '1o108',
    },
  ],
}

/** Returns only known differences scoped to the route, fixture, and runtime architecture. */
export function getKnownRecognitionFullPageTextDifferences(
  modelKey: string,
  fixture: string,
  runtimeArch = arch,
): readonly KnownRuntimeTextDifference[] {
  if (runtimeArch === 'arm64')
    return []
  return (x64RecognitionFullPageTextDifferences[modelKey] ?? []).filter(difference => difference.fixture === fixture)
}
