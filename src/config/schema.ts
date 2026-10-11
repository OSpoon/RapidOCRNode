import { z } from 'zod'

const optionalString = z.string().nullable().optional()
const optionalNumber = z.number().finite().nullable().optional()
const optionalBoolean = z.boolean().nullable().optional()
const optionalArray = z.array(z.unknown()).nullable().optional()
const optionalRecord = z.record(z.string(), z.unknown()).nullable().optional()

function optionalSection(shape: z.ZodRawShape): z.ZodType {
  return z.strictObject(shape).nullable().optional()
}

const configSchema = z.strictObject({
  Global: optionalSection({
    text_score: optionalNumber,
    use_det: optionalBoolean,
    use_cls: optionalBoolean,
    use_rec: optionalBoolean,
    use_preprocess_img: optionalBoolean,
    min_side_len: optionalNumber,
    max_side_len: optionalNumber,
    use_vertical_padding: optionalBoolean,
    min_height: optionalNumber,
    width_height_ratio: optionalNumber,
    return_word_box: optionalBoolean,
    return_single_char_box: optionalBoolean,
    font_path: optionalString,
    log_level: optionalString,
    model_root_dir: optionalString,
  }),
  EngineConfig: optionalSection({
    onnxruntime: optionalSection({
      intra_op_num_threads: optionalNumber,
      inter_op_num_threads: optionalNumber,
      enable_cpu_mem_arena: optionalBoolean,
      cpu_ep_cfg: optionalSection({ arena_extend_strategy: optionalString }),
      use_cuda: optionalBoolean,
      cuda_ep_cfg: optionalSection({
        device_id: optionalNumber,
        arena_extend_strategy: optionalString,
        cudnn_conv_algo_search: optionalString,
        do_copy_in_default_stream: optionalBoolean,
      }),
      use_dml: optionalBoolean,
      dml_ep_cfg: optionalSection({ device_id: optionalNumber }),
      use_cann: optionalBoolean,
      cann_ep_cfg: optionalSection({
        device_id: optionalNumber,
        arena_extend_strategy: optionalString,
        npu_mem_limit: optionalNumber,
        op_select_impl_mode: optionalString,
        optypelist_for_implmode: optionalString,
        enable_cann_graph: optionalBoolean,
      }),
      use_coreml: optionalBoolean,
      coreml_ep_cfg: optionalSection({
        ModelFormat: optionalString,
        MLComputeUnits: optionalString,
        RequireStaticInputShapes: optionalNumber,
        EnableOnSubgraphs: optionalNumber,
        SpecializationStrategy: optionalString,
        ProfileComputePlan: optionalNumber,
        AllowLowPrecisionAccumulationOnGPU: optionalNumber,
        ModelCacheDirectory: optionalString,
      }),
    }),
    openvino: optionalRecord,
    paddle: optionalRecord,
    torch: optionalRecord,
    tensorrt: optionalRecord,
    mnn: optionalRecord,
  }),
  Det: optionalSection({
    engine_type: optionalString,
    lang_type: optionalString,
    model_type: optionalString,
    ocr_version: optionalString,
    task_type: optionalString,
    model_path: optionalString,
    model_dir: optionalString,
    limit_side_len: optionalNumber,
    limit_type: optionalString,
    mean: optionalArray,
    std: optionalArray,
    thresh: optionalNumber,
    box_thresh: optionalNumber,
    max_candidates: optionalNumber,
    unclip_ratio: optionalNumber,
    use_dilation: optionalBoolean,
    score_mode: optionalString,
  }),
  Cls: optionalSection({
    engine_type: optionalString,
    lang_type: optionalString,
    model_type: optionalString,
    ocr_version: optionalString,
    task_type: optionalString,
    model_path: optionalString,
    model_dir: optionalString,
    cls_image_shape: optionalArray,
    cls_batch_num: optionalNumber,
    cls_thresh: optionalNumber,
    label_list: optionalArray,
  }),
  Rec: optionalSection({
    engine_type: optionalString,
    lang_type: optionalString,
    model_type: optionalString,
    ocr_version: optionalString,
    task_type: optionalString,
    model_path: optionalString,
    model_dir: optionalString,
    rec_keys_path: optionalString,
    rec_img_shape: optionalArray,
    rec_batch_num: optionalNumber,
  }),
})

export function validateRapidOCRConfigShape(value: unknown): void {
  const result = configSchema.safeParse(value)
  if (result.success)
    return

  const issue = result.error.issues[0]!
  if (issue.code === 'unrecognized_keys') {
    const parent = issue.path.length > 0 ? `${issue.path.join('.')}.` : 'configuration.'
    throw new TypeError(`${parent}${issue.keys[0]} is not supported by RapidOCR Node.`)
  }

  const path = issue.path.map(String).join('.')
  if (issue.code === 'invalid_type') {
    if (issue.expected === 'object')
      throw new TypeError(`${path} must be a mapping.`)
    if (issue.expected === 'string' || issue.expected === 'number' || issue.expected === 'boolean') {
      const field = issue.path.length > 0 ? String(issue.path.at(-1)) : 'configuration'
      const suffix = issue.expected === 'number' ? 'a finite number' : `a ${issue.expected}`
      throw new TypeError(`${field} must be ${suffix}.`)
    }
  }

  throw new TypeError(`${path || 'configuration'} is invalid: ${issue.message}`)
}
