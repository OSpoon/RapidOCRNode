import type { RapidOCR } from '../../src/pipeline/rapidocr.js'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { RapidOCR as RapidOCRClass } from '../../src/pipeline/rapidocr.js'
import { queryRuntimeStatus, runServer, stopServer } from '../../src/server/runtime.js'

describe('rapidOCR server runtime control', () => {
  afterEach(() => vi.restoreAllMocks())

  it('warms models before reporting ready and stops gracefully over the local control channel', async () => {
    const runtimeDir = await mkdtemp(join(tmpdir(), 'rapidocr-server-runtime-'))
    const fakeOcr = {
      close: vi.fn(async () => {}),
      recognize: vi.fn(),
      warmup: vi.fn(async () => {}),
    }
    const create = vi.spyOn(RapidOCRClass, 'create').mockResolvedValue(fakeOcr as unknown as RapidOCR)
    let serverRun: Promise<void> | undefined
    try {
      serverRun = runServer({
        host: '127.0.0.1',
        port: 0,
        maxFileSize: 1024,
        maxConcurrency: 1,
        maxImagePixels: 20_000_000,
        runtimeDir,
        rapidOptions: {},
        restartArgs: ['serve', 'start', '--runtime-dir', runtimeDir],
      })
      const status = await waitForStatus(runtimeDir)
      expect(status.status).toBe('running')
      expect(status.httpPort).toBeGreaterThan(0)
      expect(create).toHaveBeenCalledOnce()
      expect(create).toHaveBeenCalledWith({ useDetection: true, useClassifier: true, useRecognition: true })
      expect(fakeOcr.warmup).toHaveBeenCalledOnce()
      expect(await fetch(`${status.baseUrl}/health/ready`).then(response => response.json())).toEqual({ status: 'ready' })
      await stopServer(runtimeDir)
      await serverRun
      expect(fakeOcr.close).toHaveBeenCalledOnce()
      await expect(readFile(join(runtimeDir, 'rapidocr-node-server.json'))).rejects.toMatchObject({ code: 'ENOENT' })
    }
    finally {
      if (serverRun)
        await Promise.race([serverRun, new Promise(resolve => setTimeout(resolve, 2_000))])
      await rm(runtimeDir, { force: true, recursive: true })
    }
  })
})

async function waitForStatus(runtimeDir: string): Promise<Awaited<ReturnType<typeof queryRuntimeStatus>>> {
  const deadline = Date.now() + 5_000
  let lastError: unknown
  while (Date.now() < deadline) {
    try {
      return await queryRuntimeStatus(runtimeDir)
    }
    catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, 20))
    }
  }
  throw new Error('Server runtime did not publish ready state.', { cause: lastError })
}
