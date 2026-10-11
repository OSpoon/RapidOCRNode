import { describe, expect, it } from 'vitest'
import { reorderTextForDisplay } from '../../src/pipeline/recognition/rtl.js'

describe('rTL display ordering', () => {
  it('reorders Arabic text into the display order used by upstream RapidOCR', () => {
    expect(reorderTextForDisplay('مرحبا بالعالم')).toBe('ملاعلاب ابحرم')
  })

  it('preserves LTR runs and mirrors punctuation in mixed-direction text', () => {
    expect(reorderTextForDisplay('مرحبا 123 ABC')).toBe('ABC 123 ابحرم')
    expect(reorderTextForDisplay('A (مرحبا)')).toBe('A (ابحرم)')
  })
})
