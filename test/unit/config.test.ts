import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { resolveRapidOCROptions, writeRapidOCRConfig } from '../../src/config/rapidocr.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map(directory => rm(directory, { force: true, recursive: true })))
})

describe('onnxruntime RapidOCR configuration', () => {
  it('loads the pinned upstream YAML structure and chooses ONNX Runtime models', async () => {
    const path = fileURLToPath(new URL('../fixtures/rapidocr-v3.10.0-config.yaml', import.meta.url))
    await expect(resolveRapidOCROptions({ configPath: path })).resolves.toMatchObject({
      enableCpuMemArena: false,
      recognitionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
      executionProviders: 'auto',
    })
  })

  it('maps OCR stages, preprocessing, model routes, and ONNX execution providers', async () => {
    const path = await createConfig(`Global:\n  use_det: false\n  text_score: 0.4\n  model_root_dir: ./cache\nEngineConfig:\n  onnxruntime:\n    intra_op_num_threads: 2\n    enable_cpu_mem_arena: true\n    use_cuda: true\n    cuda_ep_cfg:\n      device_id: 3\nDet:\n  lang_type: ch\n  model_type: mobile\n  ocr_version: PP-OCRv5\n  mean: [0.4, 0.5, 0.6]\n  std: [0.2, 0.3, 0.4]\n  score_mode: slow\nCls:\n  lang_type: ch\n  model_type: mobile\n  ocr_version: PP-OCRv5\n  cls_thresh: 0.8\n  label_list: [upright, rotated]\nRec:\n  lang_type: en\n  model_type: mobile\n  ocr_version: PP-OCRv5\n  rec_img_shape: [3, 64, 384]\n  rec_batch_num: 4\n`)
    const options = await resolveRapidOCROptions({ configPath: path })
    expect(options).toMatchObject({
      useDetection: false,
      textScoreThreshold: 0.4,
      modelDirectory: './cache',
      intraOpNumThreads: 2,
      enableCpuMemArena: true,
      executionProviders: [{ name: 'cuda', deviceId: 3 }, 'cpu'],
      detectionModel: { langType: 'ch', modelType: 'mobile', ocrVersion: 'PP-OCRv5' },
      recognitionModel: { langType: 'en', modelType: 'mobile', ocrVersion: 'PP-OCRv5' },
      detectionMean: [0.4, 0.5, 0.6],
      detectionStd: [0.2, 0.3, 0.4],
      scoreMode: 'slow',
      classifierThreshold: 0.8,
      classifierLabels: ['upright', 'rotated'],
      classifierImageShape: [3, 80, 160],
      recognitionImageShape: [3, 64, 384],
      batchSize: 4,
    })
  })

  it('preserves direct ONNX Runtime CoreML and DML settings', async () => {
    const path = await createConfig(`EngineConfig:\n  onnxruntime:\n    use_coreml: true\n    coreml_ep_cfg:\n      ModelFormat: NeuralNetwork\n    use_dml: true\n    dml_ep_cfg:\n      device_id: 2\n`)
    await expect(resolveRapidOCROptions({ configPath: path })).resolves.toMatchObject({
      executionProviders: [
        { name: 'coreml', coreMlFlags: 0 },
        { name: 'dml', deviceId: 2 },
        'cpu',
      ],
    })
  })

  it('rejects unsupported backends and the unavailable CANN provider', async () => {
    await expect(resolveRapidOCROptions({ inferenceEngine: 'torch' } as never))
      .rejects
      .toThrow('inferenceEngine is no longer supported; RapidOCR Node uses ONNX Runtime.')
    const enginePath = await createConfig('Det:\n  engine_type: openvino\n')
    await expect(resolveRapidOCROptions({ configPath: enginePath }))
      .rejects
      .toThrow('Det.engine_type=openvino is not supported; RapidOCR Node uses ONNX Runtime.')
    const configPath = await createConfig('EngineConfig:\n  openvino: {}\n')
    await expect(resolveRapidOCROptions({ configPath }))
      .rejects
      .toThrow('EngineConfig.openvino is not supported by RapidOCR Node.')
    const cannPath = await createConfig('EngineConfig:\n  onnxruntime:\n    use_cann: true\n')
    await expect(resolveRapidOCROptions({ configPath: cannPath }))
      .rejects
      .toThrow('EngineConfig.onnxruntime.use_cann is not supported by RapidOCR Node')
  })

  it('applies dot-path overrides and direct options in order', async () => {
    const path = await createConfig('Global:\n  text_score: 0.6\n  use_det: true\n')
    const options = await resolveRapidOCROptions({ configPath: path, params: { 'Global.text_score': 0.3 }, useDetection: false })
    expect(options.textScoreThreshold).toBe(0.3)
    expect(options.useDetection).toBe(false)
  })

  it('writes a valid ONNX Runtime YAML config that can be loaded again', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-config-'))
    temporaryDirectories.push(directory)
    const path = await writeRapidOCRConfig(join(directory, 'rapidocr.yaml'))
    const contents = await readFile(path, 'utf8')
    expect(contents).toContain('ocr_version: PP-OCRv6')
    expect(contents).toContain('onnxruntime:')
    expect(contents).not.toContain('openvino:')
    await expect(resolveRapidOCROptions({ configPath: path })).resolves.toMatchObject({
      recognitionModel: { ocrVersion: 'PP-OCRv6', modelType: 'small' },
      useClassifier: true,
    })
  })

  it('rejects invalid options, log levels, and parameter paths', async () => {
    await expect(resolveRapidOCROptions({ logLevel: 'verbose' as never })).rejects.toThrow('Global.log_level must be')
    await expect(resolveRapidOCROptions({ params: { 'Missing.value': true } })).rejects.toThrow('Missing.value is not a valid RapidOCR config key.')
    const path = await createConfig('Global:\n  log_level: verbose\n')
    await expect(resolveRapidOCROptions({ configPath: path })).rejects.toThrow('Global.log_level must be')
  })
})

async function createConfig(contents: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-config-'))
  temporaryDirectories.push(directory)
  const path = join(directory, 'config.yaml')
  await writeFile(path, contents)
  return path
}
