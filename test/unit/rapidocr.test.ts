import { describe, expect, it } from 'vitest'
import { addVerticalPadding } from '../../src/image/image.js'
import { RapidOCR, resizeForDetection } from '../../src/pipeline/rapidocr.js'

describe('rapidocr options', () => {
  it('validates detector thresholds before resolving or downloading models', async () => {
    const creation = RapidOCR.create({ detectionThreshold: 1.1, autoDownloadModels: false })
    await expect(creation).rejects.toThrow(/threshold must be between 0 and 1/)
  })

  it('validates inter-op thread and provider configuration before resolving models', async () => {
    await expect(RapidOCR.create({ interOpNumThreads: 0, autoDownloadModels: false }))
      .rejects
      .toThrow('interOpNumThreads must be a positive integer.')
    await expect(RapidOCR.create({ executionProviders: [], autoDownloadModels: false }))
      .rejects
      .toThrow('fallback executionProviders must contain at least one provider.')
  })

  it('rejects removed inference-engine options at runtime', async () => {
    await expect(RapidOCR.create({ inferenceEngine: 'torch', autoDownloadModels: false } as never))
      .rejects
      .toThrow('inferenceEngine is no longer supported; RapidOCR Node uses ONNX Runtime.')
  })

  it('validates preprocessing dimensions and vertical-padding configuration', async () => {
    await expect(RapidOCR.create({ maxSideLength: 20, minSideLength: 30, autoDownloadModels: false }))
      .rejects
      .toThrow('maxSideLength must be finite and at least minSideLength.')
    await expect(RapidOCR.create({ widthHeightRatio: 0, autoDownloadModels: false }))
      .rejects
      .toThrow('widthHeightRatio must be positive or -1.')
  })

  it('validates the orientation-classifier rotation threshold before resolving models', async () => {
    await expect(RapidOCR.create({ classifierThreshold: 1.1, autoDownloadModels: false }))
      .rejects
      .toThrow('classifierThreshold must be between 0 and 1.')
  })

  it('validates persistent runtime updates before changing instance defaults', async () => {
    const ocr = await RapidOCR.create({ autoDownloadModels: false })
    try {
      expect(() => ocr.updateParams({ boxThreshold: 1.1 })).toThrow('boxThreshold must be between 0 and 1.')
      expect(() => ocr.updateParams({ useDetection: 'false' } as never)).toThrow('useDetection must be a boolean.')
      expect(() => ocr.updateParams({ unsupported: true } as never)).toThrow('Unknown RapidOCR update parameter: unsupported.')
    }
    finally {
      await ocr.close()
    }
  })

  it('validates the dynamic detector resize settings before resolving models', async () => {
    await expect(RapidOCR.create({ limitSideLength: 0, autoDownloadModels: false }))
      .rejects
      .toThrow('limitSideLength must be a positive finite number.')
    await expect(RapidOCR.create({ limitType: 'invalid' as 'min', autoDownloadModels: false }))
      .rejects
      .toThrow('limitType must be "min" or "max".')
  })

  it('creates without downloading models and only resolves a stage on first use', async () => {
    const ocr = await RapidOCR.create({
      autoDownloadModels: false,
      useClassifier: false,
      useRecognition: false,
    })
    try {
      const input = { data: new Uint8Array(3), height: 1, width: 1 }
      await expect(ocr.recognize(input)).rejects.toThrow(/Missing model paths: detectionModelPath/)
    }
    finally {
      await ocr.close()
    }
  })

  it('does not load any model when every OCR stage is disabled', async () => {
    const ocr = await RapidOCR.create({
      autoDownloadModels: false,
      useClassifier: false,
      useDetection: false,
      useRecognition: false,
    })
    try {
      const result = await ocr.recognize({ data: new Uint8Array(3), height: 1, width: 1 })
      expect(result.type).toBe('empty')
    }
    finally {
      await ocr.close()
    }
  })
})

describe('resizeForDetection', () => {
  it('preserves images already inside the upstream side-length bounds', async () => {
    const image = { data: new Uint8Array(323 * 430 * 3), height: 430, width: 323 }
    const resized = await resizeForDetection(image)

    expect(resized.image).toBe(image)
    expect(resized.scaleX).toBe(1)
    expect(resized.scaleY).toBe(1)
  })

  it('scales tiny images to at least the minimum side length using multiples of 32', async () => {
    const image = { data: new Uint8Array(20 * 10 * 3), height: 10, width: 20 }
    const resized = await resizeForDetection(image)

    expect(resized.image.width).toBe(64)
    expect(resized.image.height).toBe(32)
    expect(resized.scaleX).toBe(20 / 64)
    expect(resized.scaleY).toBe(10 / 32)
  })

  it('can disable image resizing and uses configured side-length bounds', async () => {
    const image = { data: new Uint8Array(80 * 40 * 3), height: 40, width: 80 }
    const unchanged = await resizeForDetection(image, {
      maxSideLength: 200,
      minSideLength: 30,
      usePreprocessImage: false,
    })
    expect(unchanged).toEqual({ image, scaleX: 1, scaleY: 1 })

    const resized = await resizeForDetection(image, {
      maxSideLength: 200,
      minSideLength: 60,
      usePreprocessImage: true,
    })
    expect(resized.image.height).toBeGreaterThanOrEqual(60)
    expect(resized.scaleY).toBe(40 / resized.image.height)
  })
})

describe('upstream vertical padding', () => {
  it('adds centered black padding for wide short images and preserves the source rows', () => {
    const image = { data: new Uint8Array(400 * 40 * 3).fill(255), height: 40, width: 400 }
    const padded = addVerticalPadding(image)

    expect(padded.top).toBe(30)
    expect(padded.image.height).toBe(100)
    expect(padded.image.data.slice(0, 400 * 3).every(value => value === 0)).toBe(true)
    expect(padded.image.data.slice(padded.top * 400 * 3, (padded.top + 40) * 400 * 3)).toEqual(image.data)
  })

  it('leaves normal page proportions unchanged', () => {
    const image = { data: new Uint8Array(300 * 400 * 3), height: 400, width: 300 }
    expect(addVerticalPadding(image)).toEqual({ image, top: 0 })
  })
})
