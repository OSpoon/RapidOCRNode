import type * as ort from 'onnxruntime-node'
import type { ExecutionProviderConfig, ExecutionProviderSelection } from '../shared/types.js'
import process, { emitWarning } from 'node:process'
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
  const isAutomatic = options.executionProviders === 'auto'
  const availableProviders = listSupportedBackends().map(backend => backend.name)
  const requestedProviders: readonly ExecutionProviderConfig[] = options.executionProviders === 'auto'
    ? getAutomaticExecutionProviders(availableProviders)
    : options.executionProviders ?? ['cpu']
  const resolvedProviders = resolveSupportedExecutionProviders(
    requestedProviders,
    availableProviders,
  )
  if (resolvedProviders.unavailable.length > 0) {
    const message = `ONNX Runtime execution provider(s) ${resolvedProviders.unavailable.join(', ')} are unavailable in this build; using ${resolvedProviders.providers.map(providerName).join(', ')}.`
    if (options.onExecutionProviderFallback)
      options.onExecutionProviderFallback(message)
    else
      emitWarning(message, { type: 'RapidOCRExecutionProviderWarning' })
  }

  if (!isAutomatic)
    return await OnnxInferenceSession.create(modelPath, buildOnnxSessionOptions(options, resolvedProviders.providers))

  for (const provider of resolvedProviders.providers) {
    if (providerName(provider).toLowerCase() === 'cpu')
      continue
    try {
      return await OnnxInferenceSession.create(modelPath, buildOnnxSessionOptions(options, [provider, 'cpu']))
    }
    catch {
      const message = `Automatic ONNX Runtime provider ${providerName(provider)} failed to initialize; trying the next available provider.`
      if (options.onExecutionProviderFallback)
        options.onExecutionProviderFallback(message)
      else
        emitWarning(message, { type: 'RapidOCRExecutionProviderWarning' })
    }
  }
  return await OnnxInferenceSession.create(modelPath, buildOnnxSessionOptions(options, ['cpu']))
}

/** Selects platform-appropriate accelerators and always leaves CPU as the final fallback. */
function getAutomaticExecutionProviders(
  supportedNames: readonly string[],
  platform: NodeJS.Platform = process.platform,
): ExecutionProviderConfig[] {
  const supported = new Set(supportedNames.map(name => name.toLowerCase()))
  const preferredNames = platform === 'darwin'
    ? ['webgpu']
    : platform === 'win32'
      ? ['cuda', 'dml', 'webgpu']
      : platform === 'linux'
        ? ['cuda', 'webgpu']
        : ['webgpu']
  const providers = preferredNames
    .filter(name => supported.has(name))
    .map(name => name as ExecutionProviderConfig)
  providers.push('cpu')
  return providers
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
