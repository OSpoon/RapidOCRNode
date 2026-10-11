import { Buffer } from 'node:buffer'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { runCli } from '../../src/cli.js'
import { RapidOCR } from '../../src/pipeline/rapidocr.js'

describe('rapidocr node cli', () => {
  afterEach(() => vi.restoreAllMocks())

  it('generates a supported upstream-style configuration file', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-cli-'))
    const configPath = join(directory, 'config', 'rapidocr.yaml')
    const upstreamAliasPath = join(directory, 'config', 'rapidocr-upstream.yaml')
    try {
      await runCli(['config', '--save-config', configPath])
      await runCli(['config', '--save_cfg_file', upstreamAliasPath])
      const config = await readFile(configPath, 'utf8')
      expect(await readFile(upstreamAliasPath, 'utf8')).toBe(config)
      expect(config).toContain('use_det: true')
      expect(config).toContain('ocr_version: PP-OCRv6')
    }
    finally {
      await rm(directory, { force: true, recursive: true })
    }
  })

  it('shows help without loading models', async () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    await expect(runCli(['--help'])).resolves.toBeUndefined()
    const help = write.mock.calls.map(([chunk]) => String(chunk)).join('')
    expect(help).toContain('Disable text detection (enabled by default)')
    expect(help).not.toMatch(/--no-det.*default: true/u)
  })

  it('rejects unknown command-line options before touching models', async () => {
    await expect(runCli(['--unknown'])).rejects.toThrow('unknown option \'--unknown\'')
  })

  it('rejects missing values and extra positional arguments through Commander', async () => {
    await expect(runCli(['--config'])).rejects.toThrow('option \'--config <path>\' argument missing')
    await expect(runCli(['one.png', 'two.png'])).rejects.toThrow('too many arguments')
  })

  it('validates enum and numeric options before model initialization', async () => {
    await expect(runCli(['--text-score', 'invalid'])).rejects.toThrow('--text-score must be a finite number.')
    await expect(runCli(['--output', 'xml'])).rejects.toThrow('Allowed choices are json, markdown.')
    await expect(runCli(['--engine', 'onnxruntime'])).rejects.toThrow('unknown option \'--engine\'')
  })

  it('lists upstream model languages using Commander-parsed options without loading OCR models', async () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    await runCli(['--list_langs', '--task', 'rec', '--ocr_version', 'PP-OCRv5', '--model_type', 'mobile'])
    expect(write).toHaveBeenCalledWith(expect.stringContaining('arabic'))
    expect(write).toHaveBeenCalledWith(expect.stringContaining('ch'))
  })

  it('matches upstream CLI routing: language overrides recognition while version and model type only list languages', async () => {
    const instance = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => ({ type: 'empty', toJSON: () => null })),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(instance as unknown as Awaited<ReturnType<typeof RapidOCR.create>>)

    await runCli([
      '--img_path',
      'receipt.png',
      '--lang_type',
      'en',
      '--return_word_box',
      '--ocr_version',
      'PP-OCRv4',
      '--model_type',
      'server',
    ])

    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      detectionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
      recognitionModel: { langType: 'en', modelType: 'small', ocrVersion: 'PP-OCRv6' },
      returnWordBoxes: true,
    }))
    expect(instance.recognize).toHaveBeenCalledWith('receipt.png', { returnWordBoxes: true })
    expect(instance.close).toHaveBeenCalledOnce()
  })

  it('accepts upstream multi-character aliases and names word-box visualizations like upstream', async () => {
    const vis = vi.fn(async () => Buffer.from('png'))
    const instance = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => ({ type: 'ocr', lines: [], toJSON: () => [], vis })),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(instance as unknown as Awaited<ReturnType<typeof RapidOCR.create>>)

    await runCli(['-img', 'receipt.png', '-vis', '-word'])

    expect(create).toHaveBeenCalledWith(expect.objectContaining({ returnWordBoxes: true }))
    expect(instance.recognize).toHaveBeenCalledWith('receipt.png', { returnWordBoxes: true })
    expect(vis).toHaveBeenCalledWith(resolve('receipt_vis_single.png'))
    expect(instance.close).toHaveBeenCalledOnce()
  })

  it('rejects an unsupported model-list task before model initialization', async () => {
    await expect(runCli(['--list-langs', '--task', 'detector'])).rejects.toThrow('Allowed choices are det, cls, rec.')
  })
})
