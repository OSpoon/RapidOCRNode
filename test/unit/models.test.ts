import { Buffer } from 'node:buffer'
import { createHash } from 'node:crypto'
import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { downloadVerifiedAsset, resolveModelPaths } from '../../src/models.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  vi.unstubAllGlobals()
  await Promise.all(temporaryDirectories.splice(0).map(path => rm(path, { recursive: true, force: true })))
})

describe('default model download', () => {
  it('downloads once for concurrent callers and reuses the verified file', async () => {
    const directory = await makeTemporaryDirectory()
    const contents = Buffer.from('verified model payload')
    const asset = createAsset(contents)
    const fetchMock = vi.fn(async () => new Response(contents))
    vi.stubGlobal('fetch', fetchMock)

    const [firstPath, secondPath] = await Promise.all([
      downloadVerifiedAsset(asset, directory),
      downloadVerifiedAsset(asset, directory),
    ])

    expect(firstPath).toBe(join(directory, asset.file))
    expect(secondPath).toBe(firstPath)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(await readFile(firstPath)).toEqual(contents)

    await downloadVerifiedAsset(asset, directory)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('rejects a checksum mismatch and leaves no partial file', async () => {
    const directory = await makeTemporaryDirectory()
    const asset = createAsset(Buffer.from('expected payload'))
    vi.stubGlobal('fetch', vi.fn(async () => new Response('unexpected payload')))

    await expect(downloadVerifiedAsset(asset, directory)).rejects.toThrow('SHA-256 mismatch')
    expect(await readdir(directory)).toEqual([])
  })

  it('uses explicit paths without network access when automatic downloads are disabled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const paths = {
      detectionModelPath: '/local/models/detector.onnx',
    }

    await expect(resolveModelPaths({
      paths,
      required: ['detectionModelPath'],
      autoDownloadModels: false,
    })).resolves.toEqual(paths)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('reports required paths when downloads are disabled', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(resolveModelPaths({
      paths: {},
      required: ['recognitionModelPath', 'dictionaryPath'],
      autoDownloadModels: false,
    })).rejects.toThrow('Missing model paths: recognitionModelPath, dictionaryPath')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

async function makeTemporaryDirectory(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), 'rapidocr-node-models-'))
  temporaryDirectories.push(path)
  return path
}

function createAsset(contents: Uint8Array) {
  return {
    file: 'test-model.onnx',
    sha256: createHash('sha256').update(contents).digest('hex'),
    url: 'https://models.example/test-model.onnx',
  }
}
