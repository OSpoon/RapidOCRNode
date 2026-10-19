#!/usr/bin/env node
import type { ModelStage, OCRVersion } from './models/registry.js'
import type { OCRStructuredFields, OCRTextStructurerOptions } from './pipeline/structured-output.js'
import type { RapidOCROptions, RapidOCRResult } from './shared/types.js'
import { readFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import process from 'node:process'
import { Command, CommanderError, InvalidArgumentError, Option } from 'commander'
import { isMainModule } from './cli/is-main.js'
import { resolveRapidOCROptions, writeRapidOCRConfig } from './config/rapidocr.js'
import { downloadModels } from './models/download.js'
import { listSupportedLanguages } from './models/registry.js'
import { RapidOCR } from './pipeline/rapidocr.js'
import { OCRTextStructurer } from './pipeline/structured-output.js'
import {
  defaultRuntimeDirectory,
  queryRuntimeStatus,
  restartServer,
  runServer,
  startServerProcess,
  stopServer,
} from './server/runtime.js'

const SAMPLE_IMAGE_URL = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/resources/test_files/ch_en_num.jpg'

type CliOptions = Record<string, unknown>

/** Runs the RapidOCR Node CLI with arguments excluding the executable and script path. */
export async function runCli(args: readonly string[]): Promise<void> {
  const cli = createCli()
  try {
    await cli.parseAsync(normalizeUpstreamAliases(args), { from: 'user' })
  }
  catch (error) {
    if (error instanceof CommanderError && error.exitCode === 0)
      return
    if (error instanceof CommanderError)
      throw new TypeError(error.message.replace(/^error: /u, ''))
    throw error
  }
}

type CliCommand = 'check' | 'config' | 'download-models' | 'image'

async function runCommand(command: CliCommand, positionalImage: string | undefined, options: CliOptions): Promise<void> {
  if (command === 'config') {
    const path = stringOption(options.saveConfig) ?? 'default_rapidocr.yaml'
    process.stdout.write(`Wrote ${await writeRapidOCRConfig(path)}\n`)
    return
  }

  if (options.listLangs === true) {
    const stage = (stringOption(options.task) ?? 'rec') as ModelStage
    const version = (stringOption(options.ocrVersion) ?? 'PP-OCRv5') as OCRVersion
    const modelType = stringOption(options.modelType) ?? 'mobile'
    process.stdout.write(`${listSupportedLanguages(stage, version, modelType).join('\n')}\n`)
    return
  }

  const rapidOptions = await resolveRapidOCROptions(toRapidOptions(options))
  if (command === 'download-models') {
    const paths = await downloadModels({
      ...(rapidOptions.modelDirectory === undefined ? {} : { modelDirectory: rapidOptions.modelDirectory }),
      ...(rapidOptions.detectionModel === undefined ? {} : { detectionModel: rapidOptions.detectionModel }),
      ...(rapidOptions.recognitionModel === undefined ? {} : { recognitionModel: rapidOptions.recognitionModel }),
      ...(rapidOptions.classifierModel === undefined ? {} : { classifierModel: rapidOptions.classifierModel }),
    })
    process.stdout.write(`${JSON.stringify(paths, null, 2)}\n`)
    return
  }

  if (command === 'check') {
    const ocr = await RapidOCR.create(rapidOptions)
    try {
      const result = await ocr.recognize(SAMPLE_IMAGE_URL)
      if (!hasOutput(result))
        throw new Error(`OCR check returned no output (result type: ${result.type}).`)
      process.stdout.write(`Success: RapidOCR Node recognized ${firstText(result) || 'text'} from the sample image.\n`)
    }
    finally {
      await ocr.close()
    }
    return
  }

  const optionImage = stringOption(options.img) ?? stringOption(options.imgPath)
  if (positionalImage && optionImage)
    throw new TypeError('Provide the image as a positional argument or with --img, not both.')
  const imagePath = positionalImage ?? optionImage
  if (!imagePath)
    throw new TypeError('Provide an image path or URL.')

  const structuredFields = await readStructuredFields(stringOption(options.structuredFields))
  if (structuredFields === undefined && (options.structuredModel !== undefined || options.structuredMaxNewTokens !== undefined || options.structuredDevice !== undefined))
    throw new TypeError('--structured-model, --structured-device, and --structured-max-new-tokens require --structured-fields.')
  if (structuredFields !== undefined && options.output === 'markdown')
    throw new TypeError('Structured extraction requires --output json.')

  const ocr = await RapidOCR.create(rapidOptions)
  try {
    const result = await ocr.recognize(imagePath, {
      ...(options.wordBoxes === undefined && options.returnWordBox === undefined
        ? {}
        : { returnWordBoxes: options.wordBoxes === true || options.returnWordBox === true }),
    })
    const visDirectory = stringOption(options.visSaveDir) ?? '.'
    const upstreamVis = options.visRes === true
    const visPath = options.vis
    if (visPath) {
      if (result.type !== 'ocr' && result.type !== 'empty')
        throw new TypeError(`Visualization requires a full OCR result; received ${result.type}.`)
      const savePath = visPath === true
        ? `${basename(imagePath).replace(/\.[^.]+$/u, '') || 'rapidocr'}-vis.png`
        : String(visPath)
      const rendered = await result.vis(savePath)
      if (rendered)
        process.stdout.write(`Saved visualization to ${resolve(savePath)}\n`)
    }
    else if (upstreamVis) {
      if (result.type !== 'ocr' && result.type !== 'empty')
        throw new TypeError(`Visualization requires a full OCR result; received ${result.type}.`)
      const suffix = options.wordBoxes === true || options.returnWordBox === true ? '_vis_single.png' : '_vis.png'
      const savePath = resolve(visDirectory, `${basename(imagePath).replace(/\.[^.]+$/u, '') || 'rapidocr'}${suffix}`)
      const rendered = await result.vis(savePath)
      if (rendered)
        process.stdout.write(`The vis result has saved in ${savePath}\n`)
    }

    if (options.output === 'markdown') {
      if (result.type !== 'ocr' && result.type !== 'empty')
        throw new TypeError(`Markdown output requires a full OCR result; received ${result.type}.`)
      process.stdout.write(`${result.toMarkdown()}\n`)
      return
    }

    const json = result.type === 'ocr' || result.type === 'empty'
      ? result.toJSON()
      : result
    if (structuredFields === undefined) {
      process.stdout.write(`${JSON.stringify(json, null, 2)}\n`)
      return
    }
    if (result.type !== 'ocr' && result.type !== 'empty')
      throw new TypeError(`Structured extraction requires a full OCR result; received ${result.type}.`)

    const structurer = new OCRTextStructurer(structuredOutputOptions(options, rapidOptions))
    try {
      const structured = await structurer.extract(result.lines, structuredFields)
      process.stdout.write(`${JSON.stringify({ ocr: json, structured }, null, 2)}\n`)
    }
    finally {
      await structurer.close()
    }
  }
  finally {
    await ocr.close()
  }
}

/** Commander only permits one-character short options; translate upstream's legacy multi-character aliases. */
function normalizeUpstreamAliases(args: readonly string[]): string[] {
  const aliases: Record<string, string> = {
    '-img': '--img_path',
    '-vis': '--vis_res',
    '-word': '--return_word_box',
  }
  return args.map((argument) => {
    for (const [alias, option] of Object.entries(aliases)) {
      if (argument === alias)
        return option
      if (argument.startsWith(`${alias}=`))
        return `${option}${argument.slice(alias.length)}`
    }
    return argument
  })
}

function createCli(): Command {
  const cli = new Command()
    .name('rapidocr-node')
    .description('Node.js OCR powered by RapidOCR models')
    .configureOutput({ writeErr: () => {} })
    .exitOverride()
    .option('--config <path>', 'Read an upstream-style YAML/JSON config')
    .option('--model-dir <path>', 'Model and font cache directory')
    .option('--structured-fields <json-or-path>', 'Extract fields using a JSON object or a JSON file (invoice and receipt use)')
    .option('--structured-model <id-or-path>', 'Text model for structured extraction')
    .addOption(new Option('--structured-device <device>', 'Execution device for structured extraction (default: cpu)').choices(['cpu', 'coreml', 'webgpu', 'cuda', 'dml']))
    .option('--structured-max-new-tokens <count>', 'Optional generation token limit for structured extraction', parsePositiveInteger)
    .option('--no-download', 'Disable automatic model downloads (enabled by default)')
    .option('--list_langs, --list-langs', 'List languages supported by a task model')
    .addOption(new Option('--task <task>', 'Task for --list-langs: det, cls, or rec').choices(['det', 'cls', 'rec']))
    .addOption(new Option('--ocr_version, --ocr-version <version>', 'PP-OCR version for --list-langs').choices(['PP-OCRv4', 'PP-OCRv5', 'PP-OCRv6']))
    .option('--model_type, --model-type <type>', 'Model type for --list-langs')
    .option('-i, --img <path>', 'Image path or URL (optional when using a positional image)')
    .addOption(new Option('--img_path, --img-path <path>', 'Image path or URL'))
    .option('--lang <language>', 'Detection and recognition language route')
    .addOption(new Option('--lang_type, --lang-type <language>', 'Detection and recognition language route'))
    .option('--text_score, --text-score <value>', 'Minimum recognition confidence', parseFiniteNumber)
    .option('--word-boxes', 'Return per-character/word boxes')
    .addOption(new Option('--return_word_box, --return-word-box', 'Return per-character/word boxes'))
    .option('--no-det', 'Disable text detection (enabled by default)')
    .option('--no-cls', 'Disable 180-degree classification (enabled by default)')
    .option('--no-rec', 'Disable text recognition (enabled by default)')
    .option('--vis [path]', 'Render and save the side-by-side visualization')
    .option('--vis_res, --vis-res', 'Save visualization to --vis-save-dir using the upstream filename')
    .option('--vis_save_dir, --vis-save-dir <path>', 'Directory for --vis-res output (default: .)')
    .option('--font_path, --font-path <path>', 'Use a local visualization font')
    .addOption(new Option('--output <format>', 'Output format: json or markdown').choices(['json', 'markdown']))
    .argument('[image]', 'Image path or URL')
    .action((image: string | undefined, _options: CliOptions, command: Command) => runCommand('image', image, command.opts()))

  cli.command('config')
    .description('Generate a RapidOCR YAML config')
    .option('--save_cfg_file, --save-config <path>', 'Config file path (default: default_rapidocr.yaml)')
    .action((options: CliOptions, command: Command) => runCommand('config', undefined, command.optsWithGlobals()))
  cli.command('download-models')
    .description('Download models selected by the config')
    .alias('download_models')
    .action((_options: CliOptions, command: Command) => runCommand('download-models', undefined, command.optsWithGlobals()))
  cli.command('check')
    .description('Run OCR on a sample image to check the installation')
    .action((_options: CliOptions, command: Command) => runCommand('check', undefined, command.optsWithGlobals()))

  const serve = cli.command('serve').description('Run the RapidOCR HTTP API service')
  const serverOptions = (command: Command): Command => command
    .option('--host <host>', 'HTTP bind address', '127.0.0.1')
    .option('--port <port>', 'HTTP port', parsePort, 8000)
    .option('--runtime-dir <path>', 'Runtime state and log directory', defaultRuntimeDirectory())
    .option('--max-file-size <bytes>', 'Maximum uploaded image size', parsePositiveInteger, 10 * 1024 * 1024)
    .option('--max-image-pixels <pixels>', 'Maximum decoded image pixel count', parsePositiveInteger, 20_000_000)
    .option('--max-concurrency <count>', 'Maximum concurrent OCR requests', parsePositiveInteger, 1)
    .option('--structured-model <id-or-path>', 'Text model for optional invoice/receipt field extraction')
    .addOption(new Option('--structured-device <device>', 'Execution device for structured extraction (default: cpu)').choices(['cpu', 'coreml', 'webgpu', 'cuda', 'dml']))
    .option('--structured-max-new-tokens <count>', 'Optional generation token limit for structured extraction', parsePositiveInteger)

  serverOptions(serve.command('run'))
    .description('Run the HTTP API in the foreground')
    .action(async (_options: CliOptions, command: Command) => {
      const options = command.optsWithGlobals() as CliOptions
      const restartArgs = makeServeArgs('start', options)
      const envArgs = process.env.RAPIDOCR_SERVE_RESTART_ARGS
      const parsedRestartArgs = envArgs ? parseRestartArgs(envArgs) : []
      const savedRestartArgs = parsedRestartArgs.length > 0 ? parsedRestartArgs : restartArgs
      const rapidOptions = await resolveRapidOCROptions(toRapidOptions(options))
      await runServer({
        host: stringOption(options.host) ?? '127.0.0.1',
        port: numberOption(options.port, 8000),
        maxFileSize: numberOption(options.maxFileSize, 10 * 1024 * 1024),
        maxConcurrency: numberOption(options.maxConcurrency, 1),
        maxImagePixels: numberOption(options.maxImagePixels, 20_000_000),
        runtimeDir: stringOption(options.runtimeDir) ?? defaultRuntimeDirectory(),
        rapidOptions,
        structuredOutput: structuredOutputOptions(options, rapidOptions),
        restartArgs: savedRestartArgs,
      })
    })

  serverOptions(serve.command('start'))
    .description('Start the HTTP API in the background')
    .action(async (_options: CliOptions, command: Command) => {
      const options = command.optsWithGlobals() as CliOptions
      const entryPath = process.argv[1]
      if (!entryPath)
        throw new Error('Cannot start the OCR server because the CLI entrypoint is unavailable.')
      const restartArgs = makeServeArgs('start', options)
      const status = await startServerProcess(
        entryPath,
        makeServeArgs('run', options),
        restartArgs,
        stringOption(options.runtimeDir) ?? defaultRuntimeDirectory(),
      )
      process.stdout.write(`${JSON.stringify(status, null, 2)}\n`)
    })

  serve.command('status')
    .description('Show the HTTP API service status')
    .option('--runtime-dir <path>', 'Runtime state and log directory', defaultRuntimeDirectory())
    .action(async (_options: CliOptions, command: Command) => {
      const options = command.optsWithGlobals() as CliOptions
      const status = await queryRuntimeStatus(stringOption(options.runtimeDir) ?? defaultRuntimeDirectory())
      process.stdout.write(`${JSON.stringify(status, null, 2)}\n`)
    })

  serve.command('stop')
    .description('Gracefully stop the HTTP API service')
    .option('--runtime-dir <path>', 'Runtime state and log directory', defaultRuntimeDirectory())
    .action(async (_options: CliOptions, command: Command) => {
      const options = command.optsWithGlobals() as CliOptions
      await stopServer(stringOption(options.runtimeDir) ?? defaultRuntimeDirectory())
      process.stdout.write('RapidOCR API stopped.\n')
    })

  serve.command('restart')
    .description('Restart the HTTP API with its previous configuration')
    .option('--runtime-dir <path>', 'Runtime state and log directory', defaultRuntimeDirectory())
    .action(async (_options: CliOptions, command: Command) => {
      const options = command.optsWithGlobals() as CliOptions
      const status = await restartServer(stringOption(options.runtimeDir) ?? defaultRuntimeDirectory())
      process.stdout.write(`${JSON.stringify(status, null, 2)}\n`)
    })

  return cli
}

function makeServeArgs(action: 'run' | 'start', options: CliOptions): string[] {
  const args: string[] = []
  const configPath = stringOption(options.config)
  const modelDirectory = stringOption(options.modelDir)
  if (configPath)
    args.push('--config', configPath)
  if (modelDirectory)
    args.push('--model-dir', modelDirectory)
  if (options.download === false)
    args.push('--no-download')
  const language = stringOption(options.lang) ?? stringOption(options.langType)
  if (language)
    args.push('--lang', language)
  if (typeof options.textScore === 'number')
    args.push('--text-score', String(options.textScore))
  if (options.wordBoxes === true || options.returnWordBox === true)
    args.push('--word-boxes')
  const fontPath = stringOption(options.fontPath)
  if (fontPath)
    args.push('--font-path', fontPath)
  args.push('serve', action)
  args.push('--host', stringOption(options.host) ?? '127.0.0.1')
  args.push('--port', String(numberOption(options.port, 8000)))
  args.push('--runtime-dir', stringOption(options.runtimeDir) ?? defaultRuntimeDirectory())
  args.push('--max-file-size', String(numberOption(options.maxFileSize, 10 * 1024 * 1024)))
  args.push('--max-image-pixels', String(numberOption(options.maxImagePixels, 20_000_000)))
  args.push('--max-concurrency', String(numberOption(options.maxConcurrency, 1)))
  const structuredModel = stringOption(options.structuredModel)
  if (structuredModel)
    args.push('--structured-model', structuredModel)
  const structuredDevice = stringOption(options.structuredDevice)
  if (structuredDevice)
    args.push('--structured-device', structuredDevice)
  const structuredMaxNewTokens = numberOption(options.structuredMaxNewTokens, 0)
  if (structuredMaxNewTokens > 0)
    args.push('--structured-max-new-tokens', String(structuredMaxNewTokens))
  return args
}

async function readStructuredFields(value: string | undefined): Promise<OCRStructuredFields | undefined> {
  if (value === undefined)
    return undefined
  const source = value.trim()
  if (source.length === 0)
    throw new TypeError('--structured-fields must be a JSON object or a path to a JSON file.')
  let json: string
  if (source.startsWith('{')) {
    json = source
  }
  else {
    json = await readFile(source, 'utf8')
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  }
  catch (error) {
    throw new TypeError(`Could not parse structured field definitions as JSON: ${error instanceof Error ? error.message : String(error)}`, { cause: error })
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed))
    throw new TypeError('Structured fields must be a JSON object mapping field names to descriptions.')
  const fields = parsed as Record<string, unknown>
  if (Object.keys(fields).length === 0)
    throw new TypeError('Structured fields must contain at least one field.')
  for (const [key, description] of Object.entries(fields)) {
    if (key.trim().length === 0 || typeof description !== 'string' || description.trim().length === 0)
      throw new TypeError('Structured field names and descriptions must be non-empty strings.')
  }
  return fields as OCRStructuredFields
}

function structuredOutputOptions(options: CliOptions, rapidOptions: RapidOCROptions): OCRTextStructurerOptions {
  const modelId = stringOption(options.structuredModel)
  const device = stringOption(options.structuredDevice) as OCRTextStructurerOptions['device'] | undefined
  const maxNewTokens = typeof options.structuredMaxNewTokens === 'number' ? options.structuredMaxNewTokens : undefined
  return {
    ...(modelId === undefined ? {} : { modelId }),
    ...(device === undefined ? {} : { device }),
    ...(maxNewTokens === undefined ? {} : { maxNewTokens }),
    ...(rapidOptions.modelDirectory === undefined ? {} : { modelDirectory: rapidOptions.modelDirectory }),
  }
}

function parseRestartArgs(value: string): string[] {
  try {
    const parsed: unknown = JSON.parse(value)
    if (Array.isArray(parsed) && parsed.every(argument => typeof argument === 'string'))
      return parsed
  }
  catch {
    // A malformed inherited value should not prevent foreground startup.
  }
  return []
}

function parsePositiveInteger(value: string): number {
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < 1)
    throw new InvalidArgumentError('Value must be a positive integer.')
  return parsed
}

function parsePort(value: string): number {
  const parsed = parsePositiveInteger(value)
  if (parsed > 65_535)
    throw new InvalidArgumentError('Port must be between 1 and 65535.')
  return parsed
}

function numberOption(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function toRapidOptions(options: CliOptions): RapidOCROptions {
  const langType = stringOption(options.lang) ?? stringOption(options.langType)
  const textScoreThreshold = typeof options.textScore === 'number' ? options.textScore : undefined

  return {
    ...(options.download === false ? { autoDownloadModels: false } : {}),
    ...(stringOption(options.config) === undefined ? {} : { configPath: stringOption(options.config) }),
    ...(stringOption(options.fontPath) === undefined ? {} : { fontPath: stringOption(options.fontPath) }),
    ...(stringOption(options.modelDir) === undefined ? {} : { modelDirectory: stringOption(options.modelDir) }),
    ...(options.cls === false ? { useClassifier: false } : {}),
    ...(options.det === false ? { useDetection: false } : {}),
    ...(options.rec === false ? { useRecognition: false } : {}),
    ...(options.wordBoxes === true || options.returnWordBox === true ? { returnWordBoxes: true } : {}),
    ...(textScoreThreshold === undefined ? {} : { textScoreThreshold }),
    ...(langType === undefined ? {} : { params: { 'Rec.lang_type': langType } }),
  }
}

function parseFiniteNumber(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed))
    throw new InvalidArgumentError('--text-score must be a finite number.')
  return parsed
}

function stringOption(value: unknown): string | undefined {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : undefined
}

function hasOutput(result: RapidOCRResult): boolean {
  switch (result.type) {
    case 'classification': return result.results.length > 0
    case 'detection': return result.boxes.length > 0
    case 'empty': return false
    case 'ocr': return result.lines.length > 0
    case 'recognition': return result.results.length > 0
  }
}

function firstText(result: RapidOCRResult): string {
  switch (result.type) {
    case 'ocr': return result.lines[0]?.text ?? ''
    case 'recognition': return result.results[0]?.text ?? ''
    default: return ''
  }
}

if (isMainModule(import.meta.url, process.argv[1])) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(`rapidocr-node: ${message}\n`)
    process.exitCode = 1
  })
}
