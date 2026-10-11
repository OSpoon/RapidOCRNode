export interface ModelAsset {
  readonly file: string
  readonly sha256?: string
  readonly url: string
  readonly companions?: readonly ModelAsset[]
}

export const MODEL_VERSION: string
export const TEST_FIXTURE_ASSETS: readonly ModelAsset[]
