import type { ExecutionProviderConfig, ExecutionProviderSelection, RapidOCROptions } from '../shared/types.js'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { parse, stringify } from 'yaml'
import { validateRapidOCRConfigShape } from './schema.js'

const DEFAULT_CONFIG = {
  Global: {
    text_score: 0.5,
    use_det: true,
    use_cls: true,
    use_rec: true,
    use_preprocess_img: true,
    min_side_len: 30,
    max_side_len: 2000,
    use_vertical_padding: true,
    min_height: 30,
    width_height_ratio: 8,
    return_word_box: false,
    return_single_char_box: false,
    font_path: null,
    log_level: 'info',
    model_root_dir: null,
  },
  EngineConfig: {
    onnxruntime: {
      intra_op_num_threads: -1,
      inter_op_num_threads: -1,
      enable_cpu_mem_arena: false,
      cpu_ep_cfg: { arena_extend_strategy: 'kSameAsRequested' },
      use_cuda: false,
      cuda_ep_cfg: {
        device_id: 0,
        arena_extend_strategy: 'kNextPowerOfTwo',
        cudnn_conv_algo_search: 'EXHAUSTIVE',
        do_copy_in_default_stream: true,
      },
      use_dml: false,
      dml_ep_cfg: null,
      use_cann: false,
      cann_ep_cfg: {
        device_id: 0,
        arena_extend_strategy: 'kNextPowerOfTwo',
        npu_mem_limit: 21474836480,
        op_select_impl_mode: 'high_performance',
        optypelist_for_implmode: 'Gelu',
        enable_cann_graph: true,
      },
      use_coreml: false,
      coreml_ep_cfg: {
        ModelFormat: 'MLProgram',
        MLComputeUnits: 'ALL',
        RequireStaticInputShapes: 0,
        EnableOnSubgraphs: 0,
        SpecializationStrategy: 'FastPrediction',
        ProfileComputePlan: 0,
        AllowLowPrecisionAccumulationOnGPU: 0,
        ModelCacheDirectory: '/tmp/RapidOCR',
      },
    },
  },
  Det: {
    engine_type: 'onnxruntime',
    lang_type: 'ch',
    model_type: 'small',
    ocr_version: 'PP-OCRv6',
    task_type: 'det',
    model_path: null,
    model_dir: null,
    limit_side_len: 736,
    limit_type: 'min',
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
    thresh: 0.3,
    box_thresh: 0.5,
    max_candidates: 1000,
    unclip_ratio: 1.6,
    use_dilation: true,
    score_mode: 'fast',
  },
  Cls: {
    engine_type: 'onnxruntime',
    lang_type: 'ch',
    model_type: 'mobile',
    ocr_version: 'PP-OCRv4',
    task_type: 'cls',
    model_path: null,
    model_dir: null,
    cls_image_shape: [3, 48, 192],
    cls_batch_num: 6,
    cls_thresh: 0.9,
    label_list: ['0', '180'],
  },
  Rec: {
    engine_type: 'onnxruntime',
    lang_type: 'ch',
    model_type: 'small',
    ocr_version: 'PP-OCRv6',
    task_type: 'rec',
    model_path: null,
    model_dir: null,
    rec_keys_path: null,
    rec_img_shape: [3, 48, 320],
    rec_batch_num: 6,
  },
} satisfies ConfigRecord

type ConfigRecord = Record<string, unknown>

const INACTIVE_UPSTREAM_ENGINE_DEFAULTS: Record<string, unknown> = {
  openvino: {
    inference_num_threads: -1,
    performance_hint: null,
    performance_num_requests: -1,
    enable_cpu_pinning: null,
    num_streams: -1,
    enable_hyper_threading: null,
    scheduling_core_type: null,
  },
  paddle: {
    cpu_math_library_num_threads: -1,
    use_npu: false,
    npu_ep_cfg: {
      device_id: 0,
      envs: {
        FLAGS_npu_jit_compile: 0,
        FLAGS_use_stride_kernel: 0,
        FLAGS_allocator_strategy: 'auto_growth',
        CUSTOM_DEVICE_BLACK_LIST: 'pad3d,pad3d_grad,set_value,set_value_with_tensor',
        FLAGS_npu_scale_aclnn: 'True',
        FLAGS_npu_split_aclnn: 'True',
      },
    },
    use_cuda: false,
    cuda_ep_cfg: { device_id: 0, gpu_mem: 500 },
  },
  torch: {
    use_cuda: false,
    cuda_ep_cfg: { device_id: 0 },
    use_npu: false,
    npu_ep_cfg: { device_id: 0 },
    use_mps: false,
  },
  tensorrt: {
    device_id: 0,
    use_fp16: true,
    use_int8: false,
    workspace_size: 1073741824,
    cache_dir: null,
    force_rebuild: false,
    det_profile: { min_shape: [1, 3, 32, 32], opt_shape: [1, 3, 736, 736], max_shape: [1, 3, 2048, 2048] },
    rec_profile: { min_shape: [1, 3, 48, 32], opt_shape: [6, 3, 48, 320], max_shape: [6, 3, 48, 2048] },
    cls_profile: { min_shape: [1, 3, 48, 32], opt_shape: [6, 3, 48, 192], max_shape: [6, 3, 48, 192] },
  },
  mnn: {},
}

/** Loads upstream-style sections, applies dot-path params, and overlays direct Node options. */
export async function resolveRapidOCROptions(input: RapidOCROptions = {}): Promise<RapidOCROptions> {
  const removedBackendOptions = [
    'inferenceEngine',
    'detectionInferenceEngine',
    'classifierInferenceEngine',
    'recognitionInferenceEngine',
    'openVINOOptions',
    'paddleNumThreads',
    'paddleUseCuda',
    'paddleDeviceId',
    'paddleGpuMemoryPoolSizeMb',
    'paddleUseNpu',
    'paddleNpuDeviceId',
    'paddleNpuEnvs',
    'torchUseMps',
    'tensorRTOptions',
  ]
  const removedOption = removedBackendOptions.find(key => Object.hasOwn(input, key))
  if (removedOption)
    throw new TypeError(`${removedOption} is no longer supported; RapidOCR Node uses ONNX Runtime.`)

  const { configPath, params, ...directOptions } = input
  let config = structuredClone(DEFAULT_CONFIG) as ConfigRecord
  if (configPath) {
    const contents = await readFile(configPath, 'utf8')
    const parsed = parse(contents) as unknown
    if (!isRecord(parsed))
      throw new TypeError(`RapidOCR configuration must contain a YAML or JSON mapping: ${configPath}`)
    config = mergeRecords(config, parsed)
  }
  if (params) {
    for (const [key, value] of Object.entries(params))
      setDottedValue(config, key, value)
  }

  validateConfig(config)
  const configuredOptions = mapConfigToOptions(config)
  const options = { ...configuredOptions, ...directOptions }
  if (
    directOptions.classifierImageShape === undefined
    && options.classifierModel?.ocrVersion === 'PP-OCRv5'
    && (
      configuredOptions.classifierImageShape === undefined
      || areEqual(configuredOptions.classifierImageShape, DEFAULT_CONFIG.Cls.cls_image_shape)
    )
  ) {
    // RapidOCR v3.10.0 uses a version-specific resize shape for PP-OCRv5 cls.
    options.classifierImageShape = [3, 80, 160]
  }
  if (options.logLevel !== undefined)
    options.logLevel = normalizeLogLevel(options.logLevel)
  return options
}

/** Writes the supported RapidOCR v3.10.0 YAML configuration fields. */
export async function writeRapidOCRConfig(path: string): Promise<string> {
  const outputPath = resolve(path)
  await mkdir(dirname(outputPath), { recursive: true })
  await writeFile(outputPath, stringify(DEFAULT_CONFIG), 'utf8')
  return outputPath
}

function mapConfigToOptions(config: ConfigRecord): RapidOCROptions {
  const global = readSection(config, 'Global')
  const detection = readSection(config, 'Det')
  const classifier = readSection(config, 'Cls')
  const recognition = readSection(config, 'Rec')
  assertEngine(detection, 'Det')
  assertEngine(classifier, 'Cls')
  assertEngine(recognition, 'Rec')
  const engineConfig = readSection(config, 'EngineConfig')
  const onnxRuntimeConfig = readSection(engineConfig, 'onnxruntime')

  const options: RapidOCROptions = {}
  assignBoolean(options, 'useDetection', global, 'use_det')
  assignBoolean(options, 'useClassifier', global, 'use_cls')
  assignBoolean(options, 'useRecognition', global, 'use_rec')
  assignBoolean(options, 'usePreprocessImage', global, 'use_preprocess_img')
  assignBoolean(options, 'useVerticalPadding', global, 'use_vertical_padding')
  assignBoolean(options, 'returnWordBoxes', global, 'return_word_box')
  assignBoolean(options, 'returnSingleCharBoxes', global, 'return_single_char_box')
  assignBoolean(options, 'useDilation', detection, 'use_dilation')
  assignNumber(options, 'textScoreThreshold', global, 'text_score')
  assignNumber(options, 'minSideLength', global, 'min_side_len')
  assignNumber(options, 'maxSideLength', global, 'max_side_len')
  assignNumber(options, 'minHeight', global, 'min_height')
  assignNumber(options, 'widthHeightRatio', global, 'width_height_ratio')
  assignNumber(options, 'limitSideLength', detection, 'limit_side_len')
  assignNumber(options, 'detectionThreshold', detection, 'thresh')
  assignNumber(options, 'boxThreshold', detection, 'box_thresh')
  assignNumber(options, 'maxCandidates', detection, 'max_candidates')
  assignNumber(options, 'unclipRatio', detection, 'unclip_ratio')
  assignNumber(options, 'classifierThreshold', classifier, 'cls_thresh')
  assignNumber(options, 'classifierBatchSize', classifier, 'cls_batch_num')
  options.classifierLabels = getClassifierLabelList(classifier)
  assignNumber(options, 'batchSize', recognition, 'rec_batch_num')
  assignString(options, 'fontPath', global, 'font_path')
  assignString(options, 'modelDirectory', global, 'model_root_dir')
  const logLevel = optionalString(global, 'log_level')
  if (logLevel !== undefined)
    options.logLevel = normalizeLogLevel(logLevel)
  assignString(options, 'detectionModelPath', detection, 'model_path')
  assignString(options, 'classifierModelPath', classifier, 'model_path')
  assignString(options, 'recognitionModelPath', recognition, 'model_path')
  assignString(options, 'dictionaryPath', recognition, 'rec_keys_path')
  assignEnum(options, 'limitType', detection, 'limit_type', ['min', 'max'] as const)
  assignEnum(options, 'scoreMode', detection, 'score_mode', ['fast', 'slow'] as const)
  assignTriplet(options, 'detectionMean', detection, 'mean')
  assignTriplet(options, 'detectionStd', detection, 'std')
  // Upstream v3.10.0 declares Cls.cls_image_shape but TextClassifier chooses
  // its input shape from OCRVersion and ignores this config value.
  assignImageShape(options, 'recognitionImageShape', recognition, 'rec_img_shape')
  assignModelSelection(options, 'detectionModel', detection)
  assignModelSelection(options, 'classifierModel', classifier)
  assignModelSelection(options, 'recognitionModel', recognition)

  assignPositiveThread(options, 'intraOpNumThreads', onnxRuntimeConfig, 'intra_op_num_threads')
  assignPositiveThread(options, 'interOpNumThreads', onnxRuntimeConfig, 'inter_op_num_threads')
  assignBoolean(options, 'enableCpuMemArena', onnxRuntimeConfig, 'enable_cpu_mem_arena')
  const providers = getProviders(onnxRuntimeConfig)
  options.executionProviders = providers as RapidOCROptions['executionProviders']
  return options
}

function assignModelSelection(
  options: RapidOCROptions,
  key: 'detectionModel' | 'classifierModel' | 'recognitionModel',
  section: ConfigRecord,
): void {
  const langType = optionalString(section, 'lang_type')
  const modelType = optionalString(section, 'model_type')
  const ocrVersion = optionalString(section, 'ocr_version')
  if (langType || modelType || ocrVersion) {
    options[key] = {
      ...(langType === undefined ? {} : { langType }),
      ...(modelType === undefined ? {} : { modelType }),
      ...(ocrVersion === undefined ? {} : { ocrVersion: ocrVersion as NonNullable<RapidOCROptions[typeof key]>['ocrVersion'] }),
    }
  }
}

function getProviders(engine: ConfigRecord): ExecutionProviderSelection {
  const providers: ExecutionProviderConfig[] = []
  if (optionalBoolean(engine, 'use_cuda')) {
    const deviceId = getDeviceId(readSection(engine, 'cuda_ep_cfg'), 'device_id')
    providers.push({ name: 'cuda', deviceId })
  }
  if (optionalBoolean(engine, 'use_dml')) {
    const deviceId = getDeviceId(readSection(engine, 'dml_ep_cfg'), 'device_id')
    providers.unshift({ name: 'dml', deviceId })
  }
  if (optionalBoolean(engine, 'use_coreml')) {
    providers.unshift({ name: 'coreml', coreMlFlags: getCoreMlFlags(readSection(engine, 'coreml_ep_cfg')) })
  }
  if (providers.length === 0)
    return ['cpu']
  providers.push('cpu')
  return providers
}

function getCoreMlFlags(config: ConfigRecord): number {
  let flags = 0
  const modelFormat = optionalString(config, 'ModelFormat') ?? 'MLProgram'
  if (modelFormat === 'MLProgram')
    flags |= 0x010
  else if (modelFormat !== 'NeuralNetwork')
    throw new TypeError('EngineConfig.onnxruntime.coreml_ep_cfg.ModelFormat must be MLProgram or NeuralNetwork.')

  const computeUnits = optionalString(config, 'MLComputeUnits') ?? 'ALL'
  if (computeUnits === 'CPU_ONLY')
    flags |= 0x001
  else if (computeUnits === 'CPU_AND_GPU')
    flags |= 0x020
  else if (computeUnits !== 'ALL')
    throw new TypeError('EngineConfig.onnxruntime.coreml_ep_cfg.MLComputeUnits must be ALL, CPU_ONLY, or CPU_AND_GPU.')

  if (readOnOffFlag(config, 'RequireStaticInputShapes'))
    flags |= 0x008
  if (readOnOffFlag(config, 'EnableOnSubgraphs'))
    flags |= 0x002
  return flags
}

function readOnOffFlag(config: ConfigRecord, key: string): boolean {
  const value = optionalNumber(config, key)
  if (value === undefined || value === 0)
    return false
  if (value === 1)
    return true
  throw new TypeError(`${key} must be 0 or 1.`)
}

function getDeviceId(config: ConfigRecord, key: string): number {
  const value = optionalNumber(config, key)
  if (value === undefined)
    return 0
  if (!Number.isInteger(value) || value < 0)
    throw new RangeError(`${key} must be a non-negative integer.`)
  return value
}

function assignImageShape<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = section[field]
  if (value === undefined || value === null)
    return
  if (!Array.isArray(value) || value.length !== 3 || value[0] !== 3 || value.slice(1).some(entry => typeof entry !== 'number' || !Number.isInteger(entry) || entry < 1))
    throw new TypeError(`${field} must contain [3, positive height, positive width].`)
  const defaultShape = [3, 48, 320]
  if (areEqual(value, defaultShape))
    return
  target[key] = value as T[K]
}

function getClassifierLabelList(section: ConfigRecord): readonly [string, string] {
  const value = section.label_list === undefined ? DEFAULT_CONFIG.Cls.label_list : section.label_list
  if (!Array.isArray(value) || value.length !== 2 || value.some(label => typeof label !== 'string')) {
    throw new TypeError('Cls.label_list must contain exactly two strings.')
  }
  return [value[0]!, value[1]!]
}

function validateConfig(config: ConfigRecord): void {
  validateRapidOCRConfigShape(config)
  const global = readSection(config, 'Global')
  const logLevel = optionalString(global, 'log_level')
  if (logLevel !== undefined)
    normalizeLogLevel(logLevel)

  const engines = readSection(config, 'EngineConfig')
  for (const [backend, defaults] of Object.entries(INACTIVE_UPSTREAM_ENGINE_DEFAULTS)) {
    if (backend in engines && !areEqual(engines[backend], defaults))
      throw new TypeError(`EngineConfig.${backend} is not supported by RapidOCR Node.`)
  }
  const onnx = readSection(engines, 'onnxruntime')
  if (optionalBoolean(onnx, 'use_cann')) {
    throw new TypeError('EngineConfig.onnxruntime.use_cann is not supported by RapidOCR Node; the ONNX Runtime Node binding does not expose CANN.')
  }
  assertKnownStage(readSection(config, 'Det'), 'Det', 'det')
  const classifier = readSection(config, 'Cls')
  assertKnownStage(classifier, 'Cls', 'cls')
  getClassifierLabelList(classifier)
  assertKnownStage(readSection(config, 'Rec'), 'Rec', 'rec')
  assertUnsupportedValue(onnx, 'cpu_ep_cfg', 'arena_extend_strategy', 'kSameAsRequested')
  assertUnsupportedValue(onnx, 'cuda_ep_cfg', 'arena_extend_strategy', 'kNextPowerOfTwo')
  assertUnsupportedValue(onnx, 'cuda_ep_cfg', 'cudnn_conv_algo_search', 'EXHAUSTIVE')
  assertUnsupportedValue(onnx, 'cuda_ep_cfg', 'do_copy_in_default_stream', true)
  for (const [key, defaultValue] of Object.entries({
    device_id: 0,
    arena_extend_strategy: 'kNextPowerOfTwo',
    npu_mem_limit: 21474836480,
    op_select_impl_mode: 'high_performance',
    optypelist_for_implmode: 'Gelu',
    enable_cann_graph: true,
  })) assertUnsupportedValue(onnx, 'cann_ep_cfg', key, defaultValue)
  for (const [key, defaultValue] of Object.entries({
    SpecializationStrategy: 'FastPrediction',
    ProfileComputePlan: 0,
    AllowLowPrecisionAccumulationOnGPU: 0,
    ModelCacheDirectory: '/tmp/RapidOCR',
  })) assertUnsupportedValue(onnx, 'coreml_ep_cfg', key, defaultValue)
}

function assertKnownStage(section: ConfigRecord, name: string, task: string): void {
  const configuredTask = optionalString(section, 'task_type')
  if (configuredTask !== undefined && configuredTask !== task)
    throw new TypeError(`${name}.task_type must be ${task}.`)
  if (section.model_dir !== undefined && section.model_dir !== null)
    throw new TypeError(`${name}.model_dir is not supported; use model_path or Global.model_root_dir.`)
  assertEngine(section, name)
}

function areEqual(left: unknown, right: unknown): boolean {
  if (left === right)
    return true
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => areEqual(value, right[index]))
  }
  if (isRecord(left) || isRecord(right)) {
    if (!isRecord(left) || !isRecord(right))
      return false
    const leftKeys = Object.keys(left).sort()
    const rightKeys = Object.keys(right).sort()
    return areEqual(leftKeys, rightKeys)
      && leftKeys.every(key => areEqual(left[key], right[key]))
  }
  return false
}

function assertUnsupportedValue(parent: ConfigRecord, sectionKey: string, key: string, supportedDefault: unknown): void {
  const section = readSection(parent, sectionKey)
  if (key in section && JSON.stringify(section[key]) !== JSON.stringify(supportedDefault)) {
    throw new TypeError(`EngineConfig.onnxruntime.${sectionKey}.${key} is not supported by RapidOCR Node.`)
  }
}

function assertEngine(section: ConfigRecord, name: string): void {
  const value = optionalString(section, 'engine_type') ?? 'onnxruntime'
  if (value !== 'onnxruntime')
    throw new TypeError(`${name}.engine_type=${value} is not supported; RapidOCR Node uses ONNX Runtime.`)
}

function assignBoolean<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = optionalBoolean(section, field)
  if (value !== undefined)
    target[key] = value as T[K]
}

function assignNumber<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = optionalNumber(section, field)
  if (value !== undefined)
    target[key] = value as T[K]
}

function assignString<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = optionalString(section, field)
  if (value !== undefined)
    target[key] = value as T[K]
}

function assignEnum<T extends object, K extends keyof T, V extends string>(
  target: T,
  key: K,
  section: ConfigRecord,
  field: string,
  valid: readonly V[],
): void {
  const value = optionalString(section, field)
  if (value === undefined)
    return
  if (!valid.includes(value as V))
    throw new TypeError(`${field} must be one of: ${valid.join(', ')}.`)
  target[key] = value as T[K]
}

function assignPositiveThread<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = optionalNumber(section, field)
  if (value !== undefined && value > 0)
    target[key] = value as T[K]
}

function assignTriplet<T extends object, K extends keyof T>(target: T, key: K, section: ConfigRecord, field: string): void {
  const value = section[field]
  if (value === undefined || value === null)
    return
  if (!Array.isArray(value) || value.length !== 3 || value.some(entry => typeof entry !== 'number' || !Number.isFinite(entry)))
    throw new TypeError(`${field} must contain three finite numbers.`)
  target[key] = value as T[K]
}

function optionalBoolean(section: ConfigRecord, key: string): boolean | undefined {
  const value = section[key]
  if (value === undefined || value === null)
    return undefined
  if (typeof value !== 'boolean')
    throw new TypeError(`${key} must be a boolean.`)
  return value
}

function optionalNumber(section: ConfigRecord, key: string): number | undefined {
  const value = section[key]
  if (value === undefined || value === null)
    return undefined
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new TypeError(`${key} must be a finite number.`)
  return value
}

function optionalString(section: ConfigRecord, key: string): string | undefined {
  const value = section[key]
  if (value === undefined || value === null)
    return undefined
  if (typeof value !== 'string')
    throw new TypeError(`${key} must be a string.`)
  return value
}

function normalizeLogLevel(value: unknown): NonNullable<RapidOCROptions['logLevel']> {
  if (typeof value !== 'string')
    throw new TypeError('Global.log_level must be debug, info, warning, error, or critical.')
  const normalized = value.toLowerCase()
  if (normalized !== 'debug' && normalized !== 'info' && normalized !== 'warning' && normalized !== 'error' && normalized !== 'critical') {
    throw new TypeError('Global.log_level must be debug, info, warning, error, or critical.')
  }
  return normalized
}

function readSection(parent: ConfigRecord, key: string): ConfigRecord {
  const value = parent[key]
  if (value === undefined || value === null)
    return {}
  if (!isRecord(value))
    throw new TypeError(`${key} must be a mapping.`)
  return value
}

function mergeRecords(base: ConfigRecord, override: ConfigRecord): ConfigRecord {
  const merged = { ...base }
  for (const [key, value] of Object.entries(override)) {
    const current = merged[key]
    merged[key] = isRecord(current) && isRecord(value) ? mergeRecords(current, value) : value
  }
  return merged
}

function setDottedValue(config: ConfigRecord, key: string, value: unknown): void {
  const parts = key.split('.')
  if (parts.length < 2 || parts.some(part => part.length === 0))
    throw new TypeError(`${key} is not a valid RapidOCR config key.`)
  let section: ConfigRecord = config
  for (const part of parts.slice(0, -1)) {
    const nested = section[part]
    if (!isRecord(nested))
      throw new TypeError(`${key} is not a valid RapidOCR config key.`)
    section = nested
  }
  const field = parts.at(-1)!
  if (!(field in section))
    throw new TypeError(`${key} is not a valid RapidOCR config key.`)
  section[field] = value
}

function isRecord(value: unknown): value is ConfigRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
