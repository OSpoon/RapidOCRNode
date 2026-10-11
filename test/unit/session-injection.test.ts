import type * as ort from 'onnxruntime-node'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import * as runtime from 'onnxruntime-node'
import { describe, expect, it, vi } from 'vitest'
import { TextClassifier } from '../../src/pipeline/classification/classifier.js'
import { TextDetector } from '../../src/pipeline/detection/detector.js'
import { RapidOCR } from '../../src/pipeline/rapidocr.js'
import { TextRecognizer } from '../../src/pipeline/recognition/recognizer.js'

function makeSession(
  inputShape: Array<number | string>,
  outputShape: Array<number | string>,
  run?: ort.InferenceSession['run'],
): ort.InferenceSession {
  return {
    inputNames: ['input'],
    outputNames: ['output'],
    inputMetadata: [{ isTensor: true, shape: inputShape, type: 'float32' }],
    outputMetadata: [{ isTensor: true, shape: outputShape, type: 'float32' }],
    run: run ?? vi.fn(),
    release: vi.fn(async () => {}),
  } as unknown as ort.InferenceSession
}

describe('caller-created inference sessions', () => {
  it('wraps detector and classifier sessions without releasing caller resources', async () => {
    const detectionSession = makeSession([1, 3, 'height', 'width'], [1, 1, 'height', 'width'])
    const classificationSession = makeSession(['batch', 3, 48, 192], ['batch', 2])
    const detector = await TextDetector.createWithSession(detectionSession)
    const classifier = await TextClassifier.createWithSession(classificationSession)

    await detector.close()
    await classifier.close()

    expect(detectionSession.release).not.toHaveBeenCalled()
    expect(classificationSession.release).not.toHaveBeenCalled()
  })

  it('uses custom upstream classifier labels and rotates when the predicted label contains 180', async () => {
    const classificationSession = makeSession(
      ['batch', 3, 48, 192],
      ['batch', 2],
      vi.fn(async () => ({
        output: new runtime.Tensor('float32', new Float32Array([0.95, 0.05]), [1, 2]),
      })),
    )
    const ocr = await RapidOCR.create({
      autoDownloadModels: false,
      classifierSession: classificationSession,
      params: { 'Cls.label_list': ['rotated 180', 'upright'] },
      useDetection: false,
      usePreprocessImage: false,
      useRecognition: false,
    })

    try {
      const result = await ocr.recognize({
        data: new Uint8Array([255, 0, 0, 0, 0, 255]),
        height: 1,
        width: 2,
      })
      expect(result.type).toBe('classification')
      if (result.type !== 'classification')
        throw new TypeError('Expected a classification-only result.')
      expect(result.clsRes[0]?.[0]).toBe('rotated 180')
      expect(result.clsRes[0]?.[1]).toBeCloseTo(0.95)
      expect(result.results).toEqual([expect.objectContaining({ angle: 180 })])
      expect(Array.from(result.imgList[0]!.data)).toEqual([0, 0, 255, 255, 0, 0])
    }
    finally {
      await ocr.close()
    }

    expect(classificationSession.release).not.toHaveBeenCalled()
  })

  it('uses the PP-OCRv5 classifier shape when upstream YAML cls_image_shape is customized', async () => {
    let inputShape: readonly number[] | undefined
    const classificationSession = makeSession(
      ['batch', 3, 'height', 'width'],
      ['batch', 2],
      vi.fn(async (feeds) => {
        const input = feeds.input
        if (!(input instanceof runtime.Tensor))
          throw new TypeError('Expected the classifier input tensor.')
        inputShape = input.dims
        return { output: new runtime.Tensor('float32', new Float32Array([0.95, 0.05]), [1, 2]) }
      }),
    )
    const ocr = await RapidOCR.create({
      autoDownloadModels: false,
      classifierSession: classificationSession,
      params: {
        'Cls.ocr_version': 'PP-OCRv5',
        'Cls.cls_image_shape': [3, 64, 256],
      },
      useDetection: false,
      usePreprocessImage: false,
      useRecognition: false,
    })

    try {
      const result = await ocr.recognize({ data: new Uint8Array([255, 0, 0]), height: 1, width: 1 })
      expect(result.type).toBe('classification')
      expect(inputShape).toEqual([1, 3, 80, 160])
    }
    finally {
      await ocr.close()
    }
  })

  it('keeps a supplied recognizer session and loads only its dictionary', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-session-'))
    const dictionaryPath = join(directory, 'dictionary.txt')
    const session = makeSession(['batch', 3, 48, 'width'], ['batch', 'time', 3])
    await writeFile(dictionaryPath, 'a\nb', 'utf8')

    try {
      const recognizer = await TextRecognizer.create({ session, dictionaryPath, autoDownloadModels: false })
      await recognizer.close()

      expect(session.release).not.toHaveBeenCalled()
    }
    finally {
      await rm(directory, { recursive: true, force: true })
    }
  })

  it('uses an injected detector without resolving or downloading the detector model', async () => {
    const detectionSession = makeSession(
      [1, 3, 'height', 'width'],
      [1, 1, 'height', 'width'],
      vi.fn(async () => ({
        output: new runtime.Tensor('float32', new Float32Array(32 * 32), [1, 1, 32, 32]),
      })),
    )
    const ocr = await RapidOCR.create({
      detectionSession,
      useClassifier: false,
      useRecognition: false,
      autoDownloadModels: false,
    })

    try {
      const result = await ocr.recognize({ data: new Uint8Array([255, 255, 255]), height: 1, width: 1 })
      expect(result.type).toBe('empty')
    }
    finally {
      await ocr.close()
    }

    expect(detectionSession.release).not.toHaveBeenCalled()
  })

  it('does not retry borderline classified crops when detection filtering is disabled', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-session-'))
    const dictionaryPath = join(directory, 'dictionary.txt')
    await writeFile(dictionaryPath, 'a\nb', 'utf8')

    const classifierSession = makeSession(
      ['batch', 3, 48, 192],
      ['batch', 2],
      vi.fn(async () => ({
        output: new runtime.Tensor('float32', new Float32Array([0.0966, 0.9034]), [1, 2]),
      })),
    )
    const tokenScores = [0.01, 0.96, 0.02, 0.01]
    const recognitionRun = vi.fn(async () => ({
      output: new runtime.Tensor('float32', new Float32Array([
        ...tokenScores,
        ...tokenScores,
        ...tokenScores,
      ]), [1, 3, 4]),
    }))
    const recognitionSession = makeSession(['batch', 3, 48, 'width'], ['batch', 3, 4], recognitionRun)
    const ocr = await RapidOCR.create({
      classifierSession,
      recognitionSession,
      dictionaryPath,
      useDetection: false,
      textScoreThreshold: 0.99,
      autoDownloadModels: false,
    })

    try {
      const result = await ocr.recognize({ data: new Uint8Array(4 * 4 * 3).fill(255), height: 4, width: 4 })
      expect(result.type).toBe('recognition')
      expect(recognitionRun).toHaveBeenCalledTimes(1)
    }
    finally {
      await ocr.close()
      await rm(directory, { recursive: true, force: true })
    }
  })

  it('does not release a caller session when its model metadata is incompatible', async () => {
    const invalidSession = makeSession([1, 1, 16, 16], [1, 1, 16, 16])

    await expect(TextDetector.createWithSession(invalidSession)).rejects.toThrow(/Detection model/)
    expect(invalidSession.release).not.toHaveBeenCalled()
  })
})
