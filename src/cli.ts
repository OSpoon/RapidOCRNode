#!/usr/bin/env node
import type { ModelStage, OCRVersion } from './models/registry.js'
import type { OCRStructuredFields, OCRTextStructurerOptions } from './pipeline/structured-output.js'
import type { ExecutionProviderSelection, RapidOCROptions, RapidOCRResult } from './shared/types.js'
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
    await cli.parseAsync([...args], { from: 'user' })
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

  const optionImage = stringOption(options.image)
  if (positionalImage && optionImage)
    throw new TypeError('Provide the image as a positional argument or with --image, not both.')
  const imagePath = positionalImage ?? optionImage
  if (!imagePath)
    throw new TypeError('Provide an image path or URL.')

  const structuredFields = await readStructuredFields(stringOption(options.structuredFields))
  if (structuredFields === undefined && [
    options.structuredModelId,
    options.structuredDevice,
    options.structuredDtype,
    options.structuredCacheDirectory,
    options.structuredRevision,
    options.structuredMaxNewTokens,
  ].some(value => value !== undefined)) {
    throw new TypeError('Structured model options require --structured-fields.')
  }
  if (structuredFields !== undefined && options.output === 'markdown') {
    throw new TypeError('Structured extraction requires --output json.')
  }

  const ocr = await RapidOCR.create(rapidOptions)
  try {
    const result = await ocr.recognize(imagePath)
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

function createCli(): Command {
  const cli = new Command()
    .name('rapidocr-node')
    .description('Node.js OCR powered by RapidOCR models')
    .configureOutput({ writeErr: () => {} })
    .exitOverride()
    .option('--structured-fields <json-or-path>', 'Extract fields using a JSON object or a JSON file (invoice and receipt use)')
    .option('--list-langs', 'List languages supported by a task model')
    .addOption(new Option('--task <task>', 'Task for --list-langs: det, cls, or rec').choices(['det', 'cls', 'rec']))
    .addOption(new Option('--ocr-version <version>', 'PP-OCR version for --list-langs').choices(['PP-OCRv4', 'PP-OCRv5', 'PP-OCRv6']))
    .option('--model-type <type>', 'Model type for --list-langs')
    .option('-i, --image <path>', 'Image path or URL (optional when using a positional image)')
    .option('--vis [path]', 'Render and save the side-by-side visualization')
    .addOption(new Option('--output <format>', 'Output format: json or markdown').choices(['json', 'markdown']))
    .argument('[image]', 'Image path or URL')
    .action((image: string | undefined, _options: CliOptions, command: Command) => runCommand('image', image, command.opts()))

  addRapidOCROptions(cli)
  addStructuredModelOptions(cli)

  cli.command('config')
    .description('Generate a RapidOCR YAML config')
    .option('--save-config <path>', 'Config file path (default: default_rapidocr.yaml)')
    .action((options: CliOptions, command: Command) => runCommand('config', undefined, command.optsWithGlobals()))
  cli.command('download-models')
    .description('Download models selected by the config')
    .action((_options: CliOptions, command: Command) => runCommand('download-models', undefined, command.optsWithGlobals()))
  cli.command('check')
    .description('Run OCR on a sample image to check the installation')
    .action((_options: CliOptions, command: Command) => runCommand('check', undefined, command.optsWithGlobals()))

  const serve = cli.command('serve').description('Run the RapidOCR HTTP API service')
  const serverOptions = (command: Command): Command => {
    const configured = command
      .option('--host <host>', 'HTTP bind address', '127.0.0.1')
      .option('--port <port>', 'HTTP port', parsePort, 8000)
      .option('--runtime-dir <path>', 'Runtime state and log directory', defaultRuntimeDirectory())
      .option('--max-file-size <bytes>', 'Maximum uploaded image size', parsePositiveInteger, 10 * 1024 * 1024)
      .option('--max-image-pixels <pixels>', 'Maximum decoded image pixel count', parsePositiveInteger, 20_000_000)
      .option('--max-concurrency <count>', 'Maximum concurrent OCR requests', parsePositiveInteger, 1)

    return addStructuredModelOptions(addRapidOCROptions(configured))
  }

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
  const language = stringOption(options.langType)
  if (language)
    args.push('--lang-type', language)
  if (options.detection === false)
    args.push('--no-detection')
  if (options.classifier === false)
    args.push('--no-classifier')
  if (options.recognition === false)
    args.push('--no-recognition')
  if (typeof options.textScoreThreshold === 'number')
    args.push('--text-score-threshold', String(options.textScoreThreshold))
  if (typeof options.boxThreshold === 'number')
    args.push('--box-threshold', String(options.boxThreshold))
  if (typeof options.unclipRatio === 'number')
    args.push('--unclip-ratio', String(options.unclipRatio))
  if (options.returnWordBoxes === true)
    args.push('--return-word-boxes')
  if (options.returnSingleCharBoxes === true)
    args.push('--return-single-char-boxes')
  const fontPath = stringOption(options.fontPath)
  if (fontPath)
    args.push('--font-path', fontPath)
  appendExecutionProviderArgs(args, options)
  args.push('serve', action)
  args.push('--host', stringOption(options.host) ?? '127.0.0.1')
  args.push('--port', String(numberOption(options.port, 8000)))
  args.push('--runtime-dir', stringOption(options.runtimeDir) ?? defaultRuntimeDirectory())
  args.push('--max-file-size', String(numberOption(options.maxFileSize, 10 * 1024 * 1024)))
  args.push('--max-image-pixels', String(numberOption(options.maxImagePixels, 20_000_000)))
  args.push('--max-concurrency', String(numberOption(options.maxConcurrency, 1)))
  const structuredModel = stringOption(options.structuredModelId)
  if (structuredModel)
    args.push('--structured-model-id', structuredModel)
  const structuredDevice = stringOption(options.structuredDevice)
  if (structuredDevice)
    args.push('--structured-device', structuredDevice)
  const structuredDtype = stringOption(options.structuredDtype)
  if (structuredDtype)
    args.push('--structured-dtype', structuredDtype)
  const structuredCacheDirectory = stringOption(options.structuredCacheDirectory)
  if (structuredCacheDirectory)
    args.push('--structured-cache-directory', structuredCacheDirectory)
  const structuredRevision = stringOption(options.structuredRevision)
  if (structuredRevision)
    args.push('--structured-revision', structuredRevision)
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
  const modelId = stringOption(options.structuredModelId)
  const device = stringOption(options.structuredDevice) as OCRTextStructurerOptions['device'] | undefined
  const dtype = stringOption(options.structuredDtype) as OCRTextStructurerOptions['dtype'] | undefined
  const cacheDirectory = stringOption(options.structuredCacheDirectory)
  const revision = stringOption(options.structuredRevision)
  const maxNewTokens = typeof options.structuredMaxNewTokens === 'number' ? options.structuredMaxNewTokens : undefined
  return {
    ...(modelId === undefined ? {} : { modelId }),
    ...(device === undefined ? {} : { device }),
    ...(dtype === undefined ? {} : { dtype }),
    ...(cacheDirectory === undefined ? {} : { cacheDirectory }),
    ...(revision === undefined ? {} : { revision }),
    ...(maxNewTokens === undefined ? {} : { maxNewTokens }),
    ...(rapidOptions.modelDirectory === undefined ? {} : { modelDirectory: rapidOptions.modelDirectory }),
  }
}

function addExecutionProviderOptions(command: Command): Command {
  return command
    .addOption(new Option('--providers <list>', 'OCR ONNX Runtime providers (priority list; default: cpu)').argParser(parseExecutionProviders))
    .addOption(new Option('--detection-providers <list>', 'Detection ONNX Runtime providers (priority list)').argParser(parseExecutionProviders).hideHelp())
    .addOption(new Option('--classifier-providers <list>', 'Classification ONNX Runtime providers (priority list)').argParser(parseExecutionProviders).hideHelp())
    .addOption(new Option('--recognition-providers <list>', 'Recognition ONNX Runtime providers (priority list)').argParser(parseExecutionProviders).hideHelp())
}

function addRapidOCROptions(command: Command): Command {
  return addExecutionProviderOptions(command
    .option('--config <path>', 'Read an upstream-style YAML/JSON config')
    .option('--model-dir <path>', 'Model and font cache directory')
    .option('--no-download', 'Disable automatic model downloads (enabled by default)')
    .option('--lang-type <language>', 'Recognition language route')
    .option('--box-threshold <value>', 'Minimum text probability inside each detected box', parseFiniteNumber)
    .option('--unclip-ratio <value>', 'DB polygon expansion ratio', parseFiniteNumber)
    .option('--text-score-threshold <value>', 'Minimum recognition confidence', parseFiniteNumber)
    .option('--return-word-boxes', 'Return Chinese character boxes and Latin/numeric word boxes')
    .option('--return-single-char-boxes', 'Split Latin/numeric word boxes into individual character boxes')
    .option('--no-detection', 'Disable text detection (enabled by default)')
    .option('--no-classifier', 'Disable 180-degree classification (enabled by default)')
    .option('--no-recognition', 'Disable text recognition (enabled by default)')
    .option('--font-path <path>', 'Use a local visualization font'))
}

function addStructuredModelOptions(command: Command): Command {
  return command
    .addOption(new Option('--structured-model-id <id-or-path>', 'Text model for structured extraction').hideHelp())
    .addOption(new Option('--structured-device <device>', 'Structured extraction device (single value; default: cpu)').choices(['cpu', 'coreml', 'webgpu', 'cuda', 'dml']))
    .addOption(new Option('--structured-dtype <dtype>', 'Transformers.js model quantization').hideHelp())
    .addOption(new Option('--structured-cache-directory <path>', 'Cache directory for the structured extraction model').hideHelp())
    .addOption(new Option('--structured-revision <revision>', 'Optional immutable model revision for structured extraction').hideHelp())
    .addOption(new Option('--structured-max-new-tokens <count>', 'Optional generation token limit for structured extraction').argParser(parsePositiveInteger).hideHelp())
}

function appendExecutionProviderArgs(args: string[], options: CliOptions): void {
  const providerOptions = [
    ['--providers', options.providers],
    ['--detection-providers', options.detectionProviders],
    ['--classifier-providers', options.classifierProviders],
    ['--recognition-providers', options.recognitionProviders],
  ] as const
  for (const [flag, value] of providerOptions) {
    if (Array.isArray(value) && value.length > 0)
      args.push(flag, JSON.stringify(value))
  }
}

function parseExecutionProviders(value: string): ExecutionProviderSelection {
  const source = value.trim()
  if (source.length === 0)
    throw new InvalidArgumentError('Provider list must not be empty.')

  let providers: unknown[]
  if (source.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(source)
      if (!Array.isArray(parsed))
        throw new Error('Expected an array.')
      providers = parsed
    }
    catch {
      throw new InvalidArgumentError('Provider list must be comma-separated names or a JSON array.')
    }
  }
  else {
    providers = source.split(',').map(provider => provider.trim())
  }

  if (providers.length === 0 || providers.some((provider) => {
    if (typeof provider === 'string')
      return provider.trim().length === 0 || provider.toLowerCase() === 'auto'
    if (typeof provider !== 'object' || provider === null || Array.isArray(provider))
      return true
    const name = (provider as { name?: unknown }).name
    return typeof name !== 'string' || name.trim().length === 0 || name.toLowerCase() === 'auto'
  })) {
    throw new InvalidArgumentError('Provider list must contain explicit provider names; auto is not supported.')
  }

  return providers as ExecutionProviderSelection
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
  const langType = stringOption(options.langType)
  const boxThreshold = typeof options.boxThreshold === 'number' ? options.boxThreshold : undefined
  const unclipRatio = typeof options.unclipRatio === 'number' ? options.unclipRatio : undefined
  const textScoreThreshold = typeof options.textScoreThreshold === 'number' ? options.textScoreThreshold : undefined
  const executionProviders = Array.isArray(options.providers) ? options.providers as ExecutionProviderSelection : undefined
  const detectionExecutionProviders = Array.isArray(options.detectionProviders) ? options.detectionProviders as ExecutionProviderSelection : undefined
  const classifierExecutionProviders = Array.isArray(options.classifierProviders) ? options.classifierProviders as ExecutionProviderSelection : undefined
  const recognitionExecutionProviders = Array.isArray(options.recognitionProviders) ? options.recognitionProviders as ExecutionProviderSelection : undefined

  return {
    ...(options.download === false ? { autoDownloadModels: false } : {}),
    ...(stringOption(options.config) === undefined ? {} : { configPath: stringOption(options.config) }),
    ...(stringOption(options.fontPath) === undefined ? {} : { fontPath: stringOption(options.fontPath) }),
    ...(stringOption(options.modelDir) === undefined ? {} : { modelDirectory: stringOption(options.modelDir) }),
    ...(options.classifier === false ? { useClassifier: false } : {}),
    ...(options.detection === false ? { useDetection: false } : {}),
    ...(options.recognition === false ? { useRecognition: false } : {}),
    ...(options.returnWordBoxes === true ? { returnWordBoxes: true } : {}),
    ...(options.returnSingleCharBoxes === true ? { returnSingleCharBoxes: true } : {}),
    ...(boxThreshold === undefined ? {} : { boxThreshold }),
    ...(unclipRatio === undefined ? {} : { unclipRatio }),
    ...(textScoreThreshold === undefined ? {} : { textScoreThreshold }),
    ...(executionProviders === undefined ? {} : { executionProviders }),
    ...(detectionExecutionProviders === undefined ? {} : { detectionExecutionProviders }),
    ...(classifierExecutionProviders === undefined ? {} : { classifierExecutionProviders }),
    ...(recognitionExecutionProviders === undefined ? {} : { recognitionExecutionProviders }),
    ...(langType === undefined ? {} : { params: { 'Rec.lang_type': langType } }),
  }
}

function parseFiniteNumber(value: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed))
    throw new InvalidArgumentError('Value must be a finite number.')
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
