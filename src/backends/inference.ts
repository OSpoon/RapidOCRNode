import type * as ort from 'onnxruntime-node'
import type { ExecutionProviderSelection } from '../shared/types.js'
import { emitWarning } from 'node:process'
import { listSupportedBackends, InferenceSession as OnnxInferenceSession } from 'onnxruntime-node'

export interface InferenceSessionOptions {
  enableCpuMemArena?: boolean
  executionProviders?: ExecutionProviderSelection
  interOpNumThreads?: number
  intraOpNumThreads?: number
  onExecutionProviderFallback?: (message: string) => void
}

export interface ResolvedExecutionProviders {
  providers: NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>
  unavailable: string[]
}

/** Creates an ONNX Runtime session using the requested Node execution providers. */
export async function createInferenceSession(
  modelPath: string,
  options: InferenceSessionOptions = {},
): Promise<ort.InferenceSession> {
  if (options.executionProviders !== undefined && !Array.isArray(options.executionProviders)) {
    throw new TypeError('executionProviders must be an explicit provider list; auto is not supported.')
  }
  if (options.executionProviders?.length === 0)
    throw new RangeError('executionProviders must contain at least one provider.')
  const availableProviders = listSupportedBackends().map(backend => backend.name)
  const resolvedProviders = resolveSupportedExecutionProviders(
    options.executionProviders ?? ['cpu'],
    availableProviders,
  )
  if (resolvedProviders.unavailable.length > 0) {
    const message = `ONNX Runtime execution provider(s) ${resolvedProviders.unavailable.join(', ')} are unavailable in this build; using ${resolvedProviders.providers.map(providerName).join(', ')}.`
    if (options.onExecutionProviderFallback)
      options.onExecutionProviderFallback(message)
    else
      emitWarning(message, { type: 'RapidOCRExecutionProviderWarning' })
  }

  return await OnnxInferenceSession.create(modelPath, buildOnnxSessionOptions(options, resolvedProviders.providers))
}

/** Keeps available providers in caller priority order and falls back to CPU when needed. */
export function resolveSupportedExecutionProviders(
  requested: NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>,
  supportedNames: readonly string[],
): ResolvedExecutionProviders {
  const supported = new Set(supportedNames.map(name => name.toLowerCase()))
  const unavailable: string[] = []
  const providers = requested.filter((provider) => {
    const name = providerName(provider)
    if (supported.has(name.toLowerCase()))
      return true
    if (!unavailable.includes(name))
      unavailable.push(name)
    return false
  })
  if (unavailable.length > 0 && !providers.some(provider => providerName(provider).toLowerCase() === 'cpu'))
    providers.push('cpu')
  if (providers.length === 0)
    providers.push('cpu')
  return { providers, unavailable }
}

/** Applies ONNX Runtime session constraints required by selected execution providers. */
export function buildOnnxSessionOptions(
  options: Pick<InferenceSessionOptions, 'enableCpuMemArena' | 'interOpNumThreads' | 'intraOpNumThreads'>,
  executionProviders: NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>,
): ort.InferenceSession.SessionOptions {
  const usesDirectML = executionProviders.some(provider => providerName(provider).toLowerCase() === 'dml')
  return {
    executionProviders,
    ...(usesDirectML ? { enableMemPattern: false, executionMode: 'sequential' as const } : {}),
    ...(options.intraOpNumThreads === undefined ? {} : { intraOpNumThreads: options.intraOpNumThreads }),
    ...(options.interOpNumThreads === undefined ? {} : { interOpNumThreads: options.interOpNumThreads }),
    ...(options.enableCpuMemArena === undefined ? {} : { enableCpuMemArena: options.enableCpuMemArena }),
  }
}

function providerName(provider: NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>[number]): string {
  return typeof provider === 'string' ? provider : provider.name
}
