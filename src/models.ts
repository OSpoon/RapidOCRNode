/*
 * Default model download and cache support for RapidOCR Node.
 * Model files are published by RapidOCR; see MODEL-LICENSES.md.
 */
import type { ModelAsset } from '../model-assets.mjs'
import { Buffer } from 'node:buffer'
import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { homedir, platform } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { env, pid } from 'node:process'
import { MODEL_ASSETS, MODEL_VERSION } from '../model-assets.mjs'

export interface DownloadModelsOptions {
  /** Directory for downloaded model files. Defaults to the operating system cache directory. */
  modelDirectory?: string
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
  required: readonly K[]
}

const MODEL_KEYS = [
  'detectionModelPath',
  'recognitionModelPath',
  'classifierModelPath',
  'dictionaryPath',
] as const satisfies readonly (keyof DownloadedModelPaths)[]

const downloadsInProgress = new Map<string, Promise<string>>()

/** Downloads and verifies the default PP-OCRv6 models and dictionary. */
export async function downloadModels(options: DownloadModelsOptions = {}): Promise<DownloadedModelPaths> {
  const modelDirectory = getModelDirectory(options.modelDirectory)
  await Promise.all(MODEL_KEYS.map(async key => downloadVerifiedAsset(MODEL_ASSETS[key], modelDirectory)))
  return getDownloadedModelPaths(modelDirectory)
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
  await Promise.all(missing.map(async (key) => {
    downloaded.set(key, await downloadVerifiedAsset(MODEL_ASSETS[key], modelDirectory))
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

/** Downloads a pinned asset once, verifies its checksum, and installs it atomically. */
export async function downloadVerifiedAsset(asset: ModelAsset, modelDirectory: string): Promise<string> {
  const path = join(resolve(modelDirectory), asset.file)
  const pending = downloadsInProgress.get(path)
  if (pending)
    return pending

  const task = downloadAndVerify(asset, path)
  downloadsInProgress.set(path, task)
  try {
    return await task
  }
  finally {
    if (downloadsInProgress.get(path) === task)
      downloadsInProgress.delete(path)
  }
}

function getModelDirectory(modelDirectory?: string): string {
  if (modelDirectory)
    return resolve(modelDirectory)

  const configuredDirectory = env.RAPIDOCR_MODEL_DIR
  if (configuredDirectory)
    return resolve(configuredDirectory)

  const system = platform()
  const cacheRoot = system === 'darwin'
    ? join(homedir(), 'Library', 'Caches')
    : system === 'win32'
      ? env.LOCALAPPDATA || join(homedir(), 'AppData', 'Local')
      : env.XDG_CACHE_HOME || join(homedir(), '.cache')

  return join(cacheRoot, 'rapidocr-node', 'models', MODEL_VERSION)
}

function getDownloadedModelPaths(modelDirectory: string): DownloadedModelPaths {
  return {
    detectionModelPath: join(modelDirectory, MODEL_ASSETS.detectionModelPath.file),
    recognitionModelPath: join(modelDirectory, MODEL_ASSETS.recognitionModelPath.file),
    classifierModelPath: join(modelDirectory, MODEL_ASSETS.classifierModelPath.file),
    dictionaryPath: join(modelDirectory, MODEL_ASSETS.dictionaryPath.file),
  }
}

async function downloadAndVerify(asset: ModelAsset, path: string): Promise<string> {
  try {
    const local = await readFile(path)
    if (sha256(local) === asset.sha256)
      return path
  }
  catch (error) {
    if (!isMissingFile(error))
      throw error
  }

  await mkdir(dirname(path), { recursive: true })
  let temporaryPath: string | undefined
  try {
    const response = await fetch(asset.url, { signal: AbortSignal.timeout(120_000) })
    if (!response.ok) {
      throw new Error(`Download failed for ${asset.file}: ${response.status} ${response.statusText}`)
    }

    const contents = Buffer.from(await response.arrayBuffer())
    const actualHash = sha256(contents)
    if (actualHash !== asset.sha256) {
      throw new Error(`SHA-256 mismatch for ${asset.file}: expected ${asset.sha256}, received ${actualHash}`)
    }

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

function isMissingFile(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function sha256(contents: Uint8Array): string {
  return createHash('sha256').update(contents).digest('hex')
}
