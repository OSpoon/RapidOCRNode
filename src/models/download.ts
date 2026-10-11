/*
 * Default model download and cache support for RapidOCR Node.
 * Model files are published by RapidOCR; see MODEL-LICENSES.md.
 */
import type { ModelAsset } from '../../model-assets.mjs'
import type { ModelSelection } from './registry.js'
import { Buffer } from 'node:buffer'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, open, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { homedir, platform } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { env, pid } from 'node:process'
import pRetry from 'p-retry'
import { MODEL_VERSION } from '../../model-assets.mjs'
import { resolveModelRoute } from './registry.js'

export interface DownloadModelsOptions {
  /** Directory for downloaded model files. Defaults to the operating system cache directory. */
  modelDirectory?: string
  /** Detection model family, language, and size. Defaults to PP-OCRv6 small. */
  detectionModel?: ModelSelection
  /** Recognition model family, language, and size. Defaults to PP-OCRv6 small. */
  recognitionModel?: ModelSelection
  /** Orientation classifier route. Defaults to PP-OCRv4 mobile. */
  classifierModel?: ModelSelection
}

export interface DownloadedModelPaths {
  classifierModelPath: string
  detectionModelPath: string
  dictionaryPath: string
  recognitionModelPath: string
}

export interface ResolveModelPathsOptions<K extends keyof DownloadedModelPaths> extends DownloadModelsOptions {
  /** Defaults to true when a required model path is missing. */
  autoDownloadModels?: boolean
  paths: Partial<DownloadedModelPaths>
  /** Model assets selected for this pipeline instance. */
  assets?: Partial<Record<keyof DownloadedModelPaths, ModelAsset>>
  required: readonly K[]
}

const MODEL_KEYS = [
  'detectionModelPath',
  'recognitionModelPath',
  'classifierModelPath',
  'dictionaryPath',
] as const satisfies readonly (keyof DownloadedModelPaths)[]

const DOWNLOAD_RANGE_SIZE = 8 * 1024 * 1024
const downloadsInProgress = new Map<string, Promise<string>>()

function isRetryableHttpStatus(status: number): boolean {
  return status === 404 || status === 408 || status === 425 || status === 429 || status >= 500
}

class RetryableDownloadResponseError extends Error {
  constructor(file: string, status: number, statusText: string) {
    super(`Download failed for ${file}: ${status} ${statusText}`)
    this.name = 'RetryableDownloadResponseError'
  }
}

/** Downloads the selected runtime models and recognition dictionary. */
export async function downloadModels(options: DownloadModelsOptions = {}): Promise<DownloadedModelPaths> {
  const modelDirectory = getModelDirectory(options.modelDirectory)
  const assets = selectModelAssets(options)
  const paths = {} as DownloadedModelPaths
  await Promise.all(MODEL_KEYS.map(async (key) => {
    const asset = assets[key]
    if (!asset)
      throw new Error(`Unable to resolve selected model asset: ${key}`)
    paths[key] = await downloadVerifiedAsset(asset, modelDirectory)
  }))
  return paths
}

/** Resolves supplied paths and downloads only required defaults that are missing. */
export async function resolveModelPaths<K extends keyof DownloadedModelPaths>(
  options: ResolveModelPathsOptions<K>,
): Promise<Pick<DownloadedModelPaths, K>> {
  const modelDirectory = getModelDirectory(options.modelDirectory)
  const missing = options.required.filter(key => !options.paths[key])
  if (missing.length > 0 && options.autoDownloadModels === false) {
    throw new TypeError(`Missing model paths: ${missing.join(', ')}. Provide local paths or enable autoDownloadModels.`)
  }

  const downloaded = new Map<keyof DownloadedModelPaths, string>()
  const assets = options.assets ?? selectModelAssets(options)
  await Promise.all(missing.map(async (key) => {
    const asset = assets[key]
    if (!asset)
      throw new Error(`Unable to resolve selected model asset: ${key}`)
    downloaded.set(key, await downloadVerifiedAsset(asset, modelDirectory))
  }))

  const paths = {} as Pick<DownloadedModelPaths, K>
  for (const key of options.required) {
    const path = options.paths[key] || downloaded.get(key)
    if (!path) {
      throw new Error(`Unable to resolve model path: ${key}`)
    }
    paths[key] = path as Pick<DownloadedModelPaths, K>[K]
  }
  return paths
}

/** Selects upstream-compatible model and dictionary assets for each stage. */
export function selectModelAssets(
  options: DownloadModelsOptions = {},
  includeClassifier = true,
): Partial<Record<keyof DownloadedModelPaths, ModelAsset>> {
  const detection = resolveModelRoute('det', options.detectionModel)
  const recognition = resolveModelRoute('rec', options.recognitionModel)
  const classifier = includeClassifier
    ? resolveModelRoute('cls', options.classifierModel ?? { ocrVersion: 'PP-OCRv4' })
    : undefined
  if (!recognition.dictionary) {
    throw new TypeError(`RapidOCR upstream has no dictionary route for ${recognition.modelKey}.`)
  }

  return {
    detectionModelPath: detection.asset,
    recognitionModelPath: recognition.asset,
    ...(classifier === undefined ? {} : { classifierModelPath: classifier.asset }),
    dictionaryPath: recognition.dictionary,
  }
}

/** Downloads a pinned asset once, verifies its checksum, and installs it atomically. */
export async function downloadVerifiedAsset(asset: ModelAsset, modelDirectory: string): Promise<string> {
  const path = join(resolve(modelDirectory), asset.file)
  const pending = downloadsInProgress.get(path)
  if (pending)
    return pending

  const task = (async () => {
    await downloadAndVerify(asset, path)
    await Promise.all((asset.companions ?? []).map(companion => downloadVerifiedAsset(companion, modelDirectory)))
    return path
  })()
  downloadsInProgress.set(path, task)
  try {
    return await task
  }
  finally {
    if (downloadsInProgress.get(path) === task)
      downloadsInProgress.delete(path)
  }
}

export function getModelDirectory(modelDirectory?: string): string {
  if (modelDirectory)
    return resolve(modelDirectory)

  const configuredDirectory = env.RAPIDOCR_MODEL_DIR
  if (configuredDirectory)
    return resolve(configuredDirectory)

  return getRapidOCRCacheDirectory('models', MODEL_VERSION)
}

/** Returns the operating system cache path for a RapidOCR asset category. */
function getRapidOCRCacheDirectory(category: string, version: string): string {
  const system = platform()
  const cacheRoot = system === 'darwin'
    ? join(homedir(), 'Library', 'Caches')
    : system === 'win32'
      ? env.LOCALAPPDATA || join(homedir(), 'AppData', 'Local')
      : env.XDG_CACHE_HOME || join(homedir(), '.cache')

  return join(cacheRoot, 'rapidocr-node', category, version)
}

async function downloadAndVerify(asset: ModelAsset, path: string): Promise<string> {
  try {
    const local = await readFile(path)
    if (asset.sha256 === undefined ? local.byteLength > 0 : sha256(local) === asset.sha256)
      return path
  }
  catch (error) {
    if (!isMissingFile(error))
      throw error
  }

  await mkdir(dirname(path), { recursive: true })
  let temporaryPath: string | undefined
  try {
    let response: Response
    try {
      response = await pRetry(async () => {
        const attempt = await fetch(asset.url, { signal: AbortSignal.timeout(120_000) })
        if (!attempt.ok && isRetryableHttpStatus(attempt.status)) {
          throw new RetryableDownloadResponseError(asset.file, attempt.status, attempt.statusText)
        }
        return attempt
      }, {
        maxTimeout: 2_000,
        minTimeout: 500,
        randomize: true,
        retries: 3,
        shouldRetry: ({ error }) => error instanceof RetryableDownloadResponseError,
      })
    }
    catch {
      return await downloadByRanges(asset, path)
    }
    if (!response.ok) {
      throw new Error(`Download failed for ${asset.file}: ${response.status} ${response.statusText}`)
    }

    let contents: Buffer
    try {
      contents = Buffer.from(await response.arrayBuffer())
    }
    catch {
      return await downloadByRanges(asset, path)
    }
    if (contents.byteLength === 0) {
      throw new Error(`Downloaded asset is empty: ${asset.file}`)
    }
    verifyAssetHash(asset, contents)

    temporaryPath = `${path}.${pid}.${randomUUID()}.partial`
    await writeFile(temporaryPath, contents, { flag: 'wx' })
    await rename(temporaryPath, path)
    return path
  }
  finally {
    if (temporaryPath)
      await rm(temporaryPath, { force: true })
  }
}

async function downloadByRanges(asset: ModelAsset, path: string): Promise<string> {
  const firstEnd = DOWNLOAD_RANGE_SIZE - 1
  const first = await fetchRange(asset.url, 0, firstEnd, asset.file)
  if (first.status === 200) {
    if (first.contents.byteLength === 0)
      throw new Error(`Downloaded asset is empty: ${asset.file}`)
    verifyAssetHash(asset, first.contents)
    return await installContents(path, first.contents)
  }

  const firstRange = parseContentRange(first.contentRange, asset.file)
  if (firstRange.start !== 0 || firstRange.end > firstEnd || firstRange.total < 1) {
    throw new Error(`Invalid byte range received for ${asset.file}: ${first.contentRange ?? 'missing Content-Range'}`)
  }

  const temporaryPath = `${path}.${pid}.${randomUUID()}.partial`
  const output = await open(temporaryPath, 'wx')
  const hash = createHash('sha256')
  let position = 0
  let current = first
  let outputClosed = false
  try {
    while (position < firstRange.total) {
      const range = parseContentRange(current.contentRange, asset.file)
      const expectedEnd = Math.min(position + DOWNLOAD_RANGE_SIZE - 1, firstRange.total - 1)
      if (range.start !== position || range.end > expectedEnd || range.total !== firstRange.total) {
        throw new Error(`Invalid byte range received for ${asset.file}: ${current.contentRange ?? 'missing Content-Range'}`)
      }
      if (current.contents.byteLength !== range.end - range.start + 1) {
        throw new Error(`Incomplete byte range received for ${asset.file}: ${current.contents.byteLength} bytes.`)
      }

      await output.writeFile(current.contents)
      hash.update(current.contents)
      position = range.end + 1
      if (position < firstRange.total) {
        current = await fetchRange(
          asset.url,
          position,
          Math.min(position + DOWNLOAD_RANGE_SIZE - 1, firstRange.total - 1),
          asset.file,
        )
        if (current.status !== 206) {
          throw new Error(`Byte-range downloads are not supported for ${asset.file} (HTTP ${current.status}).`)
        }
      }
    }

    const actualHash = hash.digest('hex')
    if (asset.sha256 !== undefined && actualHash !== asset.sha256) {
      throw new Error(`SHA-256 mismatch for ${asset.file}: expected ${asset.sha256}, received ${actualHash}`)
    }
    await output.close()
    outputClosed = true
    await rename(temporaryPath, path)
    return path
  }
  finally {
    if (!outputClosed)
      await output.close().catch(() => {})
    await rm(temporaryPath, { force: true })
  }
}

async function fetchRange(url: string, start: number, end: number, file: string): Promise<{
  contentRange: string | null
  contents: Buffer
  status: number
}> {
  let lastError: unknown
  for (let attempt = 0; attempt < 3; attempt++) {
    let response: Response
    try {
      response = await fetch(url, {
        headers: { Range: `bytes=${start}-${end}` },
        signal: AbortSignal.timeout(120_000),
      })
    }
    catch (error) {
      lastError = error
      await pauseBeforeRetry(attempt)
      continue
    }

    if (!response.ok) {
      throw new Error(`Range download failed for ${file}: ${response.status} ${response.statusText}`)
    }
    if (response.status !== 206 && !(start === 0 && response.status === 200)) {
      throw new Error(`Range download for ${file} returned unexpected HTTP status ${response.status}.`)
    }

    try {
      return {
        contentRange: response.headers.get('content-range'),
        contents: Buffer.from(await response.arrayBuffer()),
        status: response.status,
      }
    }
    catch (error) {
      lastError = error
      await pauseBeforeRetry(attempt)
    }
  }
  throw new Error(`Failed to download byte range for ${file} after three attempts.`, { cause: lastError })
}

async function pauseBeforeRetry(attempt: number): Promise<void> {
  if (attempt < 2)
    await new Promise(resolve => setTimeout(resolve, 200 * (attempt + 1)))
}

function parseContentRange(value: string | null, file: string): { end: number, start: number, total: number } {
  const match = value?.match(/^bytes (\d+)-(\d+)\/(\d+)$/u)
  if (!match) {
    throw new Error(`Invalid Content-Range received for ${file}: ${value ?? 'missing Content-Range'}`)
  }
  return { end: Number(match[2]), start: Number(match[1]), total: Number(match[3]) }
}

async function installContents(path: string, contents: Buffer): Promise<string> {
  const temporaryPath = `${path}.${pid}.${randomUUID()}.partial`
  try {
    await writeFile(temporaryPath, contents, { flag: 'wx' })
    await rename(temporaryPath, path)
    return path
  }
  finally {
    await rm(temporaryPath, { force: true })
  }
}

function verifyAssetHash(asset: ModelAsset, contents: Uint8Array): void {
  if (asset.sha256 === undefined)
    return
  const actualHash = sha256(contents)
  if (actualHash !== asset.sha256) {
    throw new Error(`SHA-256 mismatch for ${asset.file}: expected ${asset.sha256}, received ${actualHash}`)
  }
}

function isMissingFile(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function sha256(contents: Uint8Array): string {
  return createHash('sha256').update(contents).digest('hex')
}
