import { Buffer } from 'node:buffer'
import { mkdtemp, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { runCli } from '../../src/cli.js'
import { isMainModule } from '../../src/cli/is-main.js'
import { RapidOCR } from '../../src/pipeline/rapidocr.js'
import { OCRTextStructurer } from '../../src/pipeline/structured-output.js'

describe('rapidocr node cli', () => {
  afterEach(() => vi.restoreAllMocks())

  it('detects the CLI entrypoint through a package-manager symlink', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-cli-entry-'))
    const entryPath = join(directory, 'cli.mjs')
    const symlinkPath = join(directory, 'rapidocr-node')
    try {
      await writeFile(entryPath, '')
      await symlink(entryPath, symlinkPath)
      expect(isMainModule(pathToFileURL(await realpath(entryPath)).href, symlinkPath)).toBe(true)
      expect(isMainModule(pathToFileURL(entryPath).href, undefined)).toBe(false)
    }
    finally {
      await rm(directory, { force: true, recursive: true })
    }
  })

  it('generates a supported upstream-style configuration file', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-cli-'))
    const configPath = join(directory, 'config', 'rapidocr.yaml')
    try {
      await runCli(['config', '--save-config', configPath])
      const config = await readFile(configPath, 'utf8')
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
    expect(help).toMatch(/Disable text detection \(enabled by\s+default\)/u)
    expect(help).toContain('--providers <list>')
    expect(help.match(/--providers <list>/gu)).toHaveLength(1)
    expect(help).toContain('--structured-device <device>')
    expect(help).toContain('--text-score-threshold <value>')
    expect(help).toContain('--return-word-boxes')
    expect(help).not.toContain('--execution-providers')
    expect(help).not.toContain('--detection-providers')
    expect(help).not.toContain('--structured-model-id')
    expect(help).not.toContain('--structured-dtype')
    expect(help).not.toContain('--word-boxes')
    expect(help).not.toContain('--return_word_box')
    expect(help).not.toContain('--img_path')
    expect(help).not.toMatch(/--no-detection.*default: true/u)
  })

  it('documents the HTTP server lifecycle commands and validates server limits', async () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    await runCli(['serve', '--help'])
    const help = write.mock.calls.map(([chunk]) => String(chunk)).join('')
    expect(help).toContain('start')
    expect(help).toContain('status')
    expect(help).toContain('restart')
    expect(help).toContain('stop')
    write.mockClear()
    await runCli(['serve', 'start', '--help'])
    const startHelp = write.mock.calls.map(([chunk]) => String(chunk)).join('')
    expect(startHelp).toContain('--providers <list>')
    expect(startHelp).toContain('--structured-device <device>')
    expect(startHelp.match(/--providers <list>/gu)).toHaveLength(1)
    expect(startHelp).not.toContain('--detection-providers')
    expect(startHelp).not.toContain('--structured-model-id')
    await expect(runCli(['serve', 'run', '--port', '0'])).rejects.toThrow('Value must be a positive integer.')
  })

  it('rejects unknown command-line options before touching models', async () => {
    await expect(runCli(['--unknown'])).rejects.toThrow('unknown option \'--unknown\'')
    await expect(runCli(['--img_path', 'receipt.png'])).rejects.toThrow('unknown option \'--img_path\'')
  })

  it('rejects missing values and extra positional arguments through Commander', async () => {
    await expect(runCli(['--config'])).rejects.toThrow('option \'--config <path>\' argument missing')
    await expect(runCli(['one.png', 'two.png'])).rejects.toThrow('too many arguments')
  })

  it('validates enum and numeric options before model initialization', async () => {
    await expect(runCli(['--text-score-threshold', 'invalid'])).rejects.toThrow('Value must be a finite number.')
    await expect(runCli(['--output', 'xml'])).rejects.toThrow('Allowed choices are json, markdown.')
    await expect(runCli(['--engine', 'onnxruntime'])).rejects.toThrow('unknown option \'--engine\'')
  })

  it('validates structured field options before initializing OCR models', async () => {
    const create = vi.spyOn(RapidOCR, 'create')
    await expect(runCli(['receipt.png', '--structured-fields', '{}'])).rejects.toThrow('Structured fields must contain at least one field.')
    expect(create).not.toHaveBeenCalled()
    await expect(runCli(['receipt.png', '--structured-max-new-tokens', '128'])).rejects.toThrow('Structured model options require --structured-fields.')
    expect(create).not.toHaveBeenCalled()
  })

  it('lists upstream model languages using Commander-parsed options without loading OCR models', async () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    await runCli(['--list-langs', '--task', 'rec', '--ocr-version', 'PP-OCRv5', '--model-type', 'mobile'])
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
      '--image',
      'receipt.png',
      '--lang-type',
      'en',
      '--return-word-boxes',
      '--ocr-version',
      'PP-OCRv4',
      '--model-type',
      'server',
    ])

    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      detectionModel: { langType: 'ch', modelType: 'small', ocrVersion: 'PP-OCRv6' },
      recognitionModel: { langType: 'en', modelType: 'small', ocrVersion: 'PP-OCRv6' },
      returnWordBoxes: true,
    }))
    expect(instance.recognize).toHaveBeenCalledWith('receipt.png')
    expect(instance.close).toHaveBeenCalledOnce()
  })

  it('passes explicit execution providers through the image CLI', async () => {
    const instance = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => ({ type: 'empty', toJSON: () => null })),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(instance as unknown as Awaited<ReturnType<typeof RapidOCR.create>>)

    await runCli([
      'receipt.png',
      '--providers',
      'webgpu,cpu',
      '--detection-providers',
      'cpu',
      '--recognition-providers',
      '["webgpu","cpu"]',
    ])

    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      executionProviders: ['webgpu', 'cpu'],
      detectionExecutionProviders: ['cpu'],
      recognitionExecutionProviders: ['webgpu', 'cpu'],
    }))
    expect(instance.close).toHaveBeenCalledOnce()
  })

  it('rejects the removed auto provider from CLI options', async () => {
    const create = vi.spyOn(RapidOCR, 'create')
    await expect(runCli(['receipt.png', '--providers', 'auto']))
      .rejects
      .toThrow('Provider list must contain explicit provider names; auto is not supported.')
    expect(create).not.toHaveBeenCalled()
  })

  it('runs optional structured extraction and returns it alongside the OCR JSON', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'rapidocr-node-cli-structured-'))
    const fieldsPath = join(directory, 'fields.json')
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    const instance = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => ({
        type: 'ocr',
        lines: [{ text: '发票号码：12345678' }],
        toJSON: () => ({ type: 'ocr', txts: ['发票号码：12345678'] }),
      })),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(instance as unknown as Awaited<ReturnType<typeof RapidOCR.create>>)
    const extract = vi.spyOn(OCRTextStructurer.prototype, 'extract').mockImplementation(async () => ({
      values: { invoiceNumber: '12345678' },
      evidence: { invoiceNumber: [0] },
    } as never))
    const closeStructurer = vi.spyOn(OCRTextStructurer.prototype, 'close').mockResolvedValue()
    try {
      await writeFile(fieldsPath, JSON.stringify({ invoiceNumber: '发票号码' }))
      await runCli([
        './invoice.png',
        '--model-dir',
        directory,
        '--structured-fields',
        fieldsPath,
        '--structured-model-id',
        'example/receipt-model',
        '--structured-max-new-tokens',
        '512',
      ])

      const output = JSON.parse(write.mock.calls.map(([chunk]) => String(chunk)).join(''))
      expect(output).toMatchObject({
        ocr: { type: 'ocr', txts: ['发票号码：12345678'] },
        structured: { values: { invoiceNumber: '12345678' }, evidence: { invoiceNumber: [0] } },
      })
      expect(extract).toHaveBeenCalledWith([{ text: '发票号码：12345678' }], { invoiceNumber: '发票号码' })
      expect(closeStructurer).toHaveBeenCalledOnce()
      expect(instance.close).toHaveBeenCalledOnce()
      expect(create).toHaveBeenCalledWith(expect.objectContaining({ modelDirectory: directory }))
    }
    finally {
      await rm(directory, { force: true, recursive: true })
    }
  })

  it('uses the JavaScript API names for visualization and word-box options', async () => {
    const vis = vi.fn(async () => Buffer.from('png'))
    const instance = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(async () => ({ type: 'ocr', lines: [], toJSON: () => [], vis })),
    }
    const create = vi.spyOn(RapidOCR, 'create').mockResolvedValue(instance as unknown as Awaited<ReturnType<typeof RapidOCR.create>>)

    await runCli(['receipt.png', '--vis', './receipt-result.png', '--return-word-boxes'])

    expect(create).toHaveBeenCalledWith(expect.objectContaining({ returnWordBoxes: true }))
    expect(instance.recognize).toHaveBeenCalledWith('receipt.png')
    expect(vis).toHaveBeenCalledWith('./receipt-result.png')
    expect(instance.close).toHaveBeenCalledOnce()
  })

  it('rejects an unsupported model-list task before model initialization', async () => {
    await expect(runCli(['--list-langs', '--task', 'detector'])).rejects.toThrow('Allowed choices are det, cls, rec.')
  })
})
