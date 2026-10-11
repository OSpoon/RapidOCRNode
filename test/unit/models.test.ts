import { Buffer } from 'node:buffer'
import { createHash } from 'node:crypto'
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { parse } from 'yaml'
import { downloadVerifiedAsset, resolveModelPaths, selectModelAssets } from '../../src/models/download.js'
import { listSupportedLanguages, resolveModelRoute } from '../../src/models/registry.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  vi.unstubAllGlobals()
  await Promise.all(temporaryDirectories.splice(0).map(path => rm(path, { recursive: true, force: true })))
})

describe('verified ONNX model downloads', () => {
  it('downloads once for concurrent callers and reuses the verified model', async () => {
    const directory = await makeTemporaryDirectory()
    const contents = Buffer.from('verified model payload')
    const asset = createAsset(contents)
    const fetchMock = vi.fn(async () => new Response(contents))
    vi.stubGlobal('fetch', fetchMock)
    const [first, second] = await Promise.all([downloadVerifiedAsset(asset, directory), downloadVerifiedAsset(asset, directory)])
    expect(first).toBe(join(directory, asset.file))
    expect(second).toBe(first)
    expect(await readFile(first)).toEqual(contents)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('retries a truncated response with checksum-verified byte ranges', async () => {
    const directory = await makeTemporaryDirectory()
    const contents = Buffer.alloc(8 * 1024 * 1024 + 5, 0x61)
    const asset = createAsset(contents)
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const range = new Headers(init?.headers).get('range')
      if (!range) {
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(Buffer.from('partial'))
            controller.error(new TypeError('terminated'))
          },
        }))
      }
      const [, from, to] = range.match(/^bytes=(\d+)-(\d+)$/u)!
      const start = Number(from)
      const end = Math.min(Number(to), contents.length - 1)
      return new Response(contents.subarray(start, end + 1), { status: 206, headers: { 'content-range': `bytes ${start}-${end}/${contents.length}` } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const path = await downloadVerifiedAsset(asset, directory)
    expect(createHash('sha256').update(await readFile(path)).digest('hex')).toBe(asset.sha256)
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('rejects checksum mismatches and removes partial downloads', async () => {
    const directory = await makeTemporaryDirectory()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('unexpected payload')))
    await expect(downloadVerifiedAsset(createAsset(Buffer.from('expected payload')), directory)).rejects.toThrow('SHA-256 mismatch')
    expect(await readdir(directory)).toEqual([])
  })

  it('uses explicit model paths when automatic download is disabled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const paths = { detectionModelPath: '/local/models/detector.onnx' }
    await expect(resolveModelPaths({ paths, required: ['detectionModelPath'], autoDownloadModels: false })).resolves.toEqual(paths)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('reports missing required paths without starting downloads', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    await expect(resolveModelPaths({ paths: {}, required: ['recognitionModelPath', 'dictionaryPath'], autoDownloadModels: false }))
      .rejects
      .toThrow('Missing model paths: recognitionModelPath, dictionaryPath')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('upstream ONNX model registry', () => {
  it('matches every language route and alias in the pinned upstream route table', async () => {
    const upstream = parse(await readFile(new URL('../fixtures/rapidocr-v3.10.0-default_models.yaml', import.meta.url), 'utf8')) as {
      model_routes: Record<string, Record<string, unknown>>
    }
    let uniqueRoutes = 0
    for (const version of ['PP-OCRv4', 'PP-OCRv5', 'PP-OCRv6'] as const) {
      const versionConfig = upstream.model_routes[version]!
      const versionAliases = (versionConfig.aliases ?? {}) as Record<string, string>
      for (const stage of ['det', 'cls', 'rec'] as const) {
        const task = versionConfig[stage] as Record<string, unknown> | undefined
        if (!task) {
          expect(listSupportedLanguages(stage, version)).toEqual([])
          continue
        }
        const aliases = (task.aliases ?? {}) as Record<string, string>
        for (const [modelType, rawRoutes] of Object.entries(task)) {
          if (!Array.isArray(rawRoutes))
            continue
          const routes = rawRoutes as Array<{ model_key: string, supported_langs: string[] }>
          const languages = [...new Set(routes.flatMap(route => route.supported_langs))].sort()
          expect(listSupportedLanguages(stage, version, modelType)).toEqual(languages)
          for (const route of routes) {
            uniqueRoutes++
            for (const langType of route.supported_langs) {
              const result = resolveModelRoute(stage, { langType, modelType, ocrVersion: version })
              expect(result.modelKey).toBe(route.model_key)
              expect(result.asset.file).toMatch(/\.onnx$/u)
              expect(result.asset.sha256).toMatch(/^[\da-f]{64}$/u)
              if (stage === 'rec')
                expect(result.dictionary?.sha256).toMatch(/^[\da-f]{64}$/u)
            }
            for (const [alias, canonical] of Object.entries({ ...versionAliases, ...aliases })) {
              if (languages.includes(canonical)) {
                expect(resolveModelRoute(stage, { langType: alias, modelType, ocrVersion: version }).modelKey)
                  .toBe(resolveModelRoute(stage, { langType: canonical, modelType, ocrVersion: version }).modelKey)
              }
            }
          }
        }
      }
    }
    expect(uniqueRoutes).toBeGreaterThanOrEqual(42)
  })

  it('selects matching model and dictionary assets for default and explicit routes', () => {
    const defaults = selectModelAssets()
    expect(defaults.detectionModelPath?.file).toMatch(/\.onnx$/u)
    expect(defaults.classifierModelPath?.file).toMatch(/\.onnx$/u)
    expect(defaults.recognitionModelPath?.file).toBe('PP-OCRv6_rec_small.onnx')
    expect(defaults.dictionaryPath?.file).toBe('multi_PP-OCRv6_rec_small-dictionary.txt')
    const english = selectModelAssets({ recognitionModel: { langType: 'en', modelType: 'mobile', ocrVersion: 'PP-OCRv5' } })
    expect(english.recognitionModelPath?.file).toBe('en_PP-OCRv5_rec_mobile.onnx')
    expect(english.dictionaryPath?.file).toBe('en_PP-OCRv5_rec_mobile-dictionary.txt')
  })

  it('normalizes language aliases and rejects unsupported route combinations', () => {
    expect(resolveModelRoute('rec', { langType: 'zh-CN', modelType: 'small', ocrVersion: 'PP-OCRv6' }).modelKey)
      .toBe('multi_PP-OCRv6_rec_small')
    expect(() => resolveModelRoute('rec', { langType: 'japan', modelType: 'tiny', ocrVersion: 'PP-OCRv6' }))
      .toThrow(/Supported languages:.*ch/)
    expect(listSupportedLanguages('rec', 'PP-OCRv5', 'mobile')).toContain('korean')
  })
})

async function makeTemporaryDirectory(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), 'rapidocr-node-models-'))
  temporaryDirectories.push(path)
  return path
}

function createAsset(contents: Uint8Array) {
  return { file: 'test-model.onnx', sha256: createHash('sha256').update(contents).digest('hex'), url: 'https://models.example/test-model.onnx' }
}
