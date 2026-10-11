import { MODEL_REGISTRY } from './registry-data.js'

export type OCRVersion = keyof typeof MODEL_REGISTRY.routes
export type ModelStage = 'det' | 'rec' | 'cls'

/** Selects an upstream model route for one OCR stage. */
export interface ModelSelection {
  /** Upstream model family, such as PP-OCRv6. */
  ocrVersion?: OCRVersion
  /** Language code or alias supported by the selected model. */
  langType?: string
  /** Upstream model size, such as small, mobile, or server. */
  modelType?: string
}

interface ModelRoute {
  lang: string
  model_key: string
  supported_langs: string[]
}

interface ModelAssetRecord {
  file: string
  sha256: string
  url: string
  dictionary?: { file: string, sha256: string, url: string }
}

interface RegistryData {
  upstreamVersion: string
  upstreamSource: string
  routes: Record<string, Record<string, unknown>>
  assets: Record<string, Record<string, Record<string, ModelAssetRecord>>>
}

const registry = MODEL_REGISTRY as unknown as RegistryData

/** Returns the upstream model asset selected by an OCR task and language. */
export function resolveModelRoute(stage: ModelStage, selection: ModelSelection = {}): {
  asset: ModelAssetRecord
  dictionary?: ModelAssetRecord['dictionary']
  modelKey: string
} {
  const version = selection.ocrVersion ?? (stage === 'cls' ? 'PP-OCRv4' : 'PP-OCRv6')
  const task = registry.routes[version]?.[stage] as Record<string, unknown> | undefined
  if (!task) {
    throw new TypeError(`RapidOCR upstream has no ${stage} model routes for ${version}.`)
  }

  const modelType = selection.modelType ?? defaultModelType(stage, version)
  const routes = task[modelType]
  if (!Array.isArray(routes)) {
    const available = Object.keys(task).filter(key => key !== 'aliases').join(', ')
    throw new TypeError(`Unsupported ${stage} model type ${modelType} for ${version}. Available: ${available}.`)
  }

  let langType = (selection.langType ?? 'ch').trim().toLowerCase()
  const versionAliases = registry.routes[version]?.aliases as Record<string, string> | undefined
  const taskAliases = task.aliases as Record<string, string> | undefined
  langType = taskAliases?.[langType] ?? versionAliases?.[langType] ?? langType
  const route = (routes as ModelRoute[]).find(item => item.supported_langs.includes(langType))
  if (!route) {
    const supported = [...new Set((routes as ModelRoute[]).flatMap(item => item.supported_langs))].sort()
    throw new TypeError(`Unsupported ${stage}.langType=${langType} for ${version} ${modelType} model. Supported languages: ${supported.join(', ')}.`)
  }

  const asset = registry.assets[version]?.[stage]?.[route.model_key]
  if (!asset) {
    throw new Error(`RapidOCR model registry is missing asset ${version}/${stage}/${route.model_key}.`)
  }

  return {
    asset,
    ...(asset.dictionary === undefined ? {} : { dictionary: asset.dictionary }),
    modelKey: route.model_key,
  }
}

/** Lists languages that can be routed to an upstream stage model. */
export function listSupportedLanguages(stage: ModelStage, ocrVersion: OCRVersion, modelType?: string): string[] {
  const task = registry.routes[ocrVersion]?.[stage] as Record<string, unknown> | undefined
  if (!task)
    return []
  const types = modelType === undefined ? Object.keys(task).filter(key => key !== 'aliases') : [modelType]
  return [...new Set(types.flatMap((type) => {
    const routes = task[type]
    return Array.isArray(routes) ? (routes as ModelRoute[]).flatMap(route => route.supported_langs) : []
  }))].sort()
}

function defaultModelType(stage: ModelStage, version: string): string {
  if (stage === 'cls')
    return 'mobile'
  if (version === 'PP-OCRv6')
    return 'small'
  return 'mobile'
}
