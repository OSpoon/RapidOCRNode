import { Writable } from 'node:stream'
import { describe, expect, it } from 'vitest'
import { createRapidOCRLogger } from '../../src/shared/logger.js'

describe('rapidocr diagnostic logging', () => {
  it('filters below the configured level and writes enabled messages to its output stream', () => {
    const messages: string[] = []
    const output = new Writable({
      write(chunk, _encoding, callback) {
        messages.push(chunk.toString())
        callback()
      },
    })
    const logger = createRapidOCRLogger('warning', output)

    logger('debug', 'hidden debug')
    logger('info', 'hidden info')
    logger('warning', 'visible warning')
    logger('critical', 'visible critical')

    expect(messages.join('')).toContain('[RapidOCR] visible warning')
    expect(messages.join('')).toContain('[RapidOCR] visible critical')
    expect(messages.join('')).not.toContain('hidden')
  })
})
