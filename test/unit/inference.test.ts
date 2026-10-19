import { describe, expect, it } from 'vitest'
import { buildOnnxSessionOptions, createInferenceSession, resolveSupportedExecutionProviders } from '../../src/backends/inference.js'

describe('oNNX Runtime session configuration', () => {
  it('keeps available execution providers in caller priority order', () => {
    const coreml = { name: 'coreml', coreMlFlags: 0x010 } as const
    expect(resolveSupportedExecutionProviders(
      [coreml, { name: 'cuda', deviceId: 2 }, 'cpu'],
      ['cpu', 'coreml', 'cuda'],
    )).toEqual({ providers: [coreml, { name: 'cuda', deviceId: 2 }, 'cpu'], unavailable: [] })
  })

  it('drops unavailable providers and preserves CPU fallback', () => {
    expect(resolveSupportedExecutionProviders(
      [{ name: 'dml', deviceId: 0 }, { name: 'cuda', deviceId: 1 }, 'cpu'],
      ['cpu', 'coreml'],
    )).toEqual({ providers: ['cpu'], unavailable: ['dml', 'cuda'] })
  })

  it('adds CPU when every requested provider is unavailable', () => {
    expect(resolveSupportedExecutionProviders(['tensorrt', 'cuda'], ['webgpu']))
      .toEqual({ providers: ['cpu'], unavailable: ['tensorrt', 'cuda'] })
  })

  it('rejects the removed auto option', async () => {
    await expect(createInferenceSession('unused.onnx', { executionProviders: 'auto' as never }))
      .rejects
      .toThrow('executionProviders must be an explicit provider list; auto is not supported.')
  })

  it('disables memory patterns and parallel execution for DirectML', () => {
    expect(buildOnnxSessionOptions({ intraOpNumThreads: 2 }, [{ name: 'dml', deviceId: 1 }, 'cpu']))
      .toEqual({
        enableMemPattern: false,
        executionMode: 'sequential',
        executionProviders: [{ name: 'dml', deviceId: 1 }, 'cpu'],
        intraOpNumThreads: 2,
      })
    expect(buildOnnxSessionOptions({}, ['cuda', 'cpu']))
      .toEqual({ executionProviders: ['cuda', 'cpu'] })
  })
})
