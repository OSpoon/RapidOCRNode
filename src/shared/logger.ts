import type { Writable } from 'node:stream'
import type { RapidOCRLogLevel } from './types.js'
import { Console } from 'node:console'
import { stderr } from 'node:process'

const LOG_LEVEL_PRIORITY: Record<RapidOCRLogLevel, number> = {
  debug: 10,
  info: 20,
  warning: 30,
  error: 40,
  critical: 50,
}

const CONSOLE_METHOD: Record<RapidOCRLogLevel, 'debug' | 'info' | 'warn' | 'error'> = {
  debug: 'debug',
  info: 'info',
  warning: 'warn',
  error: 'error',
  critical: 'error',
}

/** Creates a leveled logger that keeps diagnostics off stdout and CLI data output. */
export function createRapidOCRLogger(level: RapidOCRLogLevel, output: Writable = stderr): (severity: RapidOCRLogLevel, message: string) => void {
  const logger = new Console({ stdout: output, stderr: output })
  const threshold = LOG_LEVEL_PRIORITY[level]
  return (severity, message) => {
    if (LOG_LEVEL_PRIORITY[severity] >= threshold)
      logger[CONSOLE_METHOD[severity]](`[RapidOCR] ${message}`)
  }
}
