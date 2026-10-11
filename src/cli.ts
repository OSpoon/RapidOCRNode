#!/usr/bin/env node
import type { ModelStage, OCRVersion } from './models/registry.js'
import type { RapidOCROptions, RapidOCRResult } from './shared/types.js'
import { basename, resolve } from 'node:path'
import process from 'node:process'
import { pathToFileURL } from 'node:url'
import { Command, CommanderError, InvalidArgumentError, Option } from 'commander'
import { resolveRapidOCROptions, writeRapidOCRConfig } from './config/rapidocr.js'
import { downloadModels } from './models/download.js'
import { listSupportedLanguages } from './models/registry.js'
import { RapidOCR } from './pipeline/rapidocr.js'

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

  const ocr = await RapidOCR.create(rapidOptions)
  try {
    if (command === 'check') {
      const result = await ocr.recognize(SAMPLE_IMAGE_URL)
      if (!hasOutput(result))
        throw new Error(`OCR check returned no output (result type: ${result.type}).`)
      process.stdout.write(`Success: RapidOCR Node recognized ${firstText(result) || 'text'} from the sample image.\n`)
      return
    }

    const optionImage = stringOption(options.img) ?? stringOption(options.imgPath)
    if (positionalImage && optionImage)
      throw new TypeError('Provide the image as a positional argument or with --img, not both.')
    const imagePath = positionalImage ?? optionImage
    if (!imagePath)
      throw new TypeError('Provide an image path or URL.')

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
    process.stdout.write(`${JSON.stringify(json, null, 2)}\n`)
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

  return cli
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

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(`rapidocr-node: ${message}\n`)
    process.exitCode = 1
  })
}
