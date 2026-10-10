export interface ModelAsset {
  readonly file: string
  readonly sha256: string
  readonly url: string
}

export const MODEL_VERSION: string
export const MODEL_ASSETS: {
  readonly detectionModelPath: ModelAsset
  readonly recognitionModelPath: ModelAsset
  readonly classifierModelPath: ModelAsset
  readonly dictionaryPath: ModelAsset
}
export const TEST_FIXTURE_ASSETS: readonly ModelAsset[]
