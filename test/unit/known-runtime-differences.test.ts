import { describe, expect, it } from 'vitest'
import { getKnownRecognitionFullPageTextDifferences } from '../helpers/known-runtime-differences.js'

describe('known runtime text differences', () => {
  it('allows the pinned x64 traditional-Chinese route output difference', () => {
    expect(getKnownRecognitionFullPageTextDifferences(
      'chinese_cht_PP-OCRv3_rec_mobile',
      'ch_en_num-sharp.png',
      'x64',
    )).toEqual([
      {
        confidenceTolerance: 0.06,
        fixture: 'ch_en_num-sharp.png',
        index: 4,
        nodeText: '1o08',
        upstreamText: '1o108',
      },
    ])
  })

  it('does not apply the difference to other routes, fixtures, or arm64', () => {
    expect(getKnownRecognitionFullPageTextDifferences(
      'chinese_cht_PP-OCRv3_rec_mobile',
      'another-fixture.png',
      'x64',
    )).toEqual([])
    expect(getKnownRecognitionFullPageTextDifferences(
      'another-model',
      'ch_en_num-sharp.png',
      'x64',
    )).toEqual([])
    expect(getKnownRecognitionFullPageTextDifferences(
      'chinese_cht_PP-OCRv3_rec_mobile',
      'ch_en_num-sharp.png',
      'arm64',
    )).toEqual([])
  })
})
