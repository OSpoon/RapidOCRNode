import { resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { RapidOCR } from 'rapidocr-node'

/* eslint unused-imports/no-unused-vars: off -- The plugin misclassifies runtime locals in this standalone JS demo; the base no-unused-vars rule remains active. */

async function main() {
  const samplePath = fileURLToPath(new URL('./sample.svg', import.meta.url))
  const imagePath = process.argv[2]
    ? resolve(process.cwd(), process.argv[2])
    : samplePath

  const ocr = await RapidOCR.create()

  try {
    const result = await ocr.recognize(imagePath)

    process.stdout.write(`Image: ${imagePath}\n`)
    if (result.lines.length === 0) {
      process.stdout.write('No text found.\n')
    }
    else {
      for (const [index, line] of result.lines.entries()) {
        process.stdout.write(`${index + 1}. ${line.text} (confidence: ${line.confidence.toFixed(3)})\n`)
      }
    }

    process.stdout.write(
      `Timings: detection ${result.timings.detection.toFixed(1)} ms, `
      + `classification ${result.timings.classification.toFixed(1)} ms, `
      + `recognition ${result.timings.recognition.toFixed(1)} ms\n`,
    )
  }
  finally {
    await ocr.close()
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`)
  process.exitCode = 1
})
