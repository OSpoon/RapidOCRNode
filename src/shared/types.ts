import type { Buffer } from 'node:buffer'
import type * as ort from 'onnxruntime-node'
import type { ModelSelection } from '../models/registry.js'

export type ExecutionProviderConfig = NonNullable<ort.InferenceSession.SessionOptions['executionProviders']>[number]

/** Upstream-compatible diagnostic threshold. Messages are written to stderr. */
export type RapidOCRLogLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical'

/** A local path, HTTP(S) image URL, encoded bytes, or raw pixel data. */
export type ImageInput = string | Uint8Array | RawRgbImage | RawPixelImage

/** Raw, interleaved RGB pixels. `data` must contain exactly width * height * 3 bytes. */
export interface RawRgbImage {
  data: Uint8Array
  height: number
  width: number
}

export type RawPixelFormat
  = | 'GRAY'
    | 'GRAY8'
    | 'GRAY_ALPHA'
    | 'GRAYA'
    | 'RGB'
    | 'RGB24'
    | 'BGR'
    | 'BGR24'
    | 'RGBA'
    | 'RGBA32'
    | 'RGBX'
    | 'BGRA'
    | 'BGRA32'
    | 'BGRX'

/** Raw grayscale, BGR/RGB, or four-channel image with optional row padding. */
export interface RawPixelImage {
  bottomUp?: boolean
  data: Uint8Array
  format: RawPixelFormat
  height: number
  stride?: number
  width: number
}

/** Options for the PP-OCR compatible text-line recognizer. */
export interface TextRecognizerOptions {
  /** Maximum number of image lines sent to the model in one inference call. */
  batchSize?: number
  /** Downloads the visualization font on the first `recognizeLines().vis()` call. Defaults to true. */
  autoDownloadFonts?: boolean
  /** Upstream Rec.rec_img_shape as channels, height, and minimum batch width. */
  imageShape?: readonly [3, number, number]
  /** Defaults to true when the default recognition model or dictionary is needed. */
  autoDownloadModels?: boolean
  /** Caller-created ONNX Runtime session. When set, `modelPath` is not resolved or downloaded; the caller retains ownership. */
  session?: ort.InferenceSession
  /** Path to the character dictionary that matches the recognition model. */
  dictionaryPath?: string
  /** Number of ONNX Runtime CPU threads. The runtime default is used when omitted. */
  intraOpNumThreads?: number
  /** Number of ONNX Runtime inter-op threads. The runtime default is used when omitted. */
  interOpNumThreads?: number
  /** ONNX Runtime CPU memory arena setting. */
  enableCpuMemArena?: boolean
  /** ONNX Runtime execution providers in priority order. Defaults to CPU. */
  executionProviders?: readonly ExecutionProviderConfig[]
  /** Directory used to cache default models. */
  modelDirectory?: string
  /** Optional TrueType/OpenType font file used by `recognizeLines().vis()`. */
  fontPath?: string
  /** Path to a PP-OCR ONNX model. */
  modelPath?: string
  /** Upstream version, language, and model type for automatic model selection. */
  modelSelection?: ModelSelection
}

/** Options for collecting CTC character positions during recognition. */
export interface TextRecognitionCallOptions {
  /** Include word-level CTC positions in the result. */
  returnWordPositions?: boolean
  /** Split Latin and numeric groups into one result per character. */
  returnSingleCharBoxes?: boolean
}

/** Text and mean per-character CTC confidence for one cropped text line. */
export interface RecognitionResult {
  confidence: number
  text: string
  /** Optional normalized horizontal positions used to build word boxes. */
  wordPositions?: WordPosition[]
}

export interface WordPosition extends RecognitionResultBase {
  /** CTC time-step indices that contributed to this word or character. */
  columns: number[]
  end: number
  /** Effective text-line length after recognizer batch padding. */
  lineWidth: number
  start: number
}

interface RecognitionResultBase {
  confidence: number
  text: string
}

export interface RecognizedWord extends RecognitionResultBase {
  box: TextBox
}

export interface Point {
  x: number
  y: number
}

/** A quadrilateral in top-left, top-right, bottom-right, bottom-left order. */
export type TextBox = readonly [Point, Point, Point, Point]

export interface RecognizedLine extends RecognitionResult {
  box: TextBox
  words?: RecognizedWord[]
}

export interface RapidOCRResultBase {
  /** Decoded source image in interleaved RGB byte order. */
  img: RawRgbImage
  lines: RecognizedLine[]
  /** Stage timings in upstream order (seconds): detection, classification, recognition. */
  elapseList: [number, number, number]
  /** Sum of the enabled stage timings in seconds, matching RapidOCR. */
  elapse: number
  /** Named stage timings in seconds, matching `elapseList`. */
  timings: {
    classification: number
    detection: number
    recognition: number
  }
}

export interface RapidOCRJsonLine {
  box: number[][]
  score: number
  txt: string
}

export interface RapidOCRExportMethods {
  /** Converts recognized lines to RapidOCR's `{ box, txt, score }` JSON shape. */
  toJSON: () => RapidOCRJsonLine[] | null
  /** Reconstructs a readable Markdown approximation from line box positions. */
  toMarkdown: () => string
  /** Renders the RapidOCR side-by-side visualization as a PNG buffer, optionally saving it. */
  vis: (savePath?: string) => Promise<Buffer | undefined>
}

export interface RapidOCRStageVisualization {
  /** Renders and optionally saves this stage's upstream-style visualization as a PNG buffer. */
  vis: (savePath?: string) => Promise<Buffer | undefined>
}

export interface RapidOCRFullResult extends RapidOCRResultBase, RapidOCRExportMethods {
  type: 'ocr'
  boxes: TextBox[]
  txts: string[]
  scores: number[]
  wordResults: RecognizedWord[][]
}

export interface RapidOCRDetectionResult extends RapidOCRResultBase, RapidOCRStageVisualization {
  type: 'detection'
  lines: []
  boxes: TextBox[]
  scores: number[]
}

export interface RapidOCRClassificationResult extends RapidOCRResultBase, RapidOCRStageVisualization {
  type: 'classification'
  lines: []
  clsRes: Array<[string, number]>
  results: Array<{ angle: 0 | 180, confidence: number }>
  imgList: RawRgbImage[]
}

export interface RapidOCRRecognitionResult extends RapidOCRResultBase, RapidOCRStageVisualization {
  type: 'recognition'
  lines: []
  results: RecognitionResult[]
  imgs: RawRgbImage[]
  txts: string[]
  scores: number[]
}

export interface RapidOCREmptyResult extends RapidOCRResultBase, RapidOCRExportMethods {
  type: 'empty'
  lines: []
  boxes: null
  txts: null
  scores: null
  wordResults: []
}

export type RapidOCRResult
  = RapidOCRFullResult
    | RapidOCRDetectionResult
    | RapidOCRClassificationResult
    | RapidOCRRecognitionResult
    | RapidOCREmptyResult

/** Select the stages used for one recognition call. */
export interface RapidOCRCallOptions {
  /** Applies the 180-degree line classifier. Defaults to the value in RapidOCROptions. */
  useClassifier?: boolean
  /** Finds text regions before recognition. Defaults to the value in RapidOCROptions. */
  useDetection?: boolean
  /** Runs CTC recognition. When disabled, return detection or classification results. Defaults to true. */
  useRecognition?: boolean
  /** Minimum text probability for each detection box. Defaults to the configured boxThreshold. */
  boxThreshold?: number
  /** DB polygon expansion ratio. Defaults to the configured unclipRatio. */
  unclipRatio?: number
  /** Minimum recognition confidence for a line to be included. Defaults to the configured threshold. */
  textScoreThreshold?: number
  /** Returns Chinese character boxes and Latin/numeric word boxes. Defaults to false. */
  returnWordBoxes?: boolean
  /** Splits Latin/numeric word boxes into individual character boxes. Defaults to false. */
  returnSingleCharBoxes?: boolean
}

/** Persistent defaults updated on a reusable RapidOCR instance. */
export type RapidOCRUpdateParams = Pick<
  RapidOCRCallOptions,
  | 'boxThreshold'
  | 'returnSingleCharBoxes'
  | 'returnWordBoxes'
  | 'textScoreThreshold'
  | 'unclipRatio'
  | 'useClassifier'
  | 'useDetection'
  | 'useRecognition'
>

/** Paths to the PP-OCR models and dictionary used by the full-page pipeline. */
export interface RapidOCROptions {
  /** Maximum number of crops sent to the recognition model in one call. */
  batchSize?: number
  /** Path to a RapidOCR-compatible YAML or JSON configuration file. */
  configPath?: string
  /** Minimum diagnostic level, matching Global.log_level. Defaults to info. */
  logLevel?: RapidOCRLogLevel
  /** Defaults to true when a required default model path is missing. */
  autoDownloadModels?: boolean
  /** Downloads the upstream visualization font when `vis()` is first used. Defaults to true. */
  autoDownloadFonts?: boolean
  /** Caller-created ONNX Runtime detector session. The caller retains ownership. */
  detectionSession?: ort.InferenceSession
  /** Caller-created ONNX Runtime direction-classifier session. The caller retains ownership. */
  classifierSession?: ort.InferenceSession
  /** Caller-created ONNX Runtime text-recognition session. The caller retains ownership. */
  recognitionSession?: ort.InferenceSession
  /** Enables the detector by default. Defaults to true. */
  useDetection?: boolean
  /** Enables the recognizer by default. Defaults to true. */
  useRecognition?: boolean
  /** Optional PP-OCR 180-degree direction classifier model. */
  classifierModelPath?: string
  /** Path to a PP-OCR ONNX model. */
  detectionModelPath?: string
  /** Upstream detection model route. Defaults to PP-OCRv6 small for Chinese. */
  detectionModel?: ModelSelection
  /** Matching recognition model dictionary file. */
  dictionaryPath?: string
  /** Upstream recognition model route and its matching dictionary. Defaults to PP-OCRv6 small for Chinese. */
  recognitionModel?: ModelSelection
  /** Upstream Rec.rec_img_shape as channels, height, and minimum batch width. */
  recognitionImageShape?: readonly [3, number, number]
  /** Number of ONNX Runtime CPU threads. The runtime default is used when omitted. */
  intraOpNumThreads?: number
  /** Number of ONNX Runtime inter-op threads. The runtime default is used when omitted. */
  interOpNumThreads?: number
  /** ONNX Runtime CPU memory arena setting, applied to every enabled stage. */
  enableCpuMemArena?: boolean
  /** Fallback execution-provider order for stages without a stage-specific setting. Defaults to CPU. */
  executionProviders?: readonly ExecutionProviderConfig[]
  /** ONNX Runtime execution providers for detection. */
  detectionExecutionProviders?: readonly ExecutionProviderConfig[]
  /** ONNX Runtime execution providers for recognition. */
  recognitionExecutionProviders?: readonly ExecutionProviderConfig[]
  /** ONNX Runtime execution providers for classification. */
  classifierExecutionProviders?: readonly ExecutionProviderConfig[]
  /** DB probability-map threshold. Defaults to 0.3. */
  detectionThreshold?: number
  /** Minimum text probability inside a detection box. Defaults to 0.5. */
  boxThreshold?: number
  /** Detector normalization mean in tensor channel order. Defaults to [0.5, 0.5, 0.5]. */
  detectionMean?: readonly [number, number, number]
  /** DB polygon expansion ratio. Defaults to 1.6. */
  unclipRatio?: number
  /** Maximum number of detection contours to process. Defaults to 1000. */
  maxCandidates?: number
  /** Dilate the probability map before contour extraction. Defaults to true. */
  useDilation?: boolean
  /** Fast rectangle or slow contour-region mean score for DB boxes. Defaults to fast. */
  scoreMode?: 'fast' | 'slow'
  /** Detector-side resize limit. Defaults to 736. */
  limitSideLength?: number
  /** Resize the short side up or long side down to limitSideLength. Defaults to min. */
  limitType?: 'min' | 'max'
  /** Detector normalization standard deviation. Defaults to [0.5, 0.5, 0.5]. */
  detectionStd?: readonly [number, number, number]
  /** Directory used to cache default models. */
  modelDirectory?: string
  /** Optional TrueType/OpenType font file for rendering visualization labels. */
  fontPath?: string
  /** Path to a PP-OCR CTC ONNX model. */
  recognitionModelPath?: string
  /** Upstream classifier model route. Defaults to PP-OCRv4 mobile. */
  classifierModel?: ModelSelection
  /** Node-specific classifier input-size override; upstream YAML Cls.cls_image_shape is ignored. */
  classifierImageShape?: readonly [3, number, number]
  /** Upstream Cls.label_list; exactly two labels for the orientation classifier output. */
  classifierLabels?: readonly [string, string]
  /** Maximum number of text lines sent to the direction classifier per batch. */
  classifierBatchSize?: number
  /** Rotate text lines when the 180-degree score exceeds this threshold. Defaults to 0.9. */
  classifierThreshold?: number
  /** Minimum recognition confidence for a line to be included. Defaults to 0.5. */
  textScoreThreshold?: number
  /** Dot-path configuration values applied over `configPath`, such as `Global.text_score`. */
  params?: Record<string, unknown>
  /** Returns Chinese character boxes and Latin/numeric word boxes. Defaults to false. */
  returnWordBoxes?: boolean
  /** Splits Latin/numeric word boxes into individual character boxes. Defaults to false. */
  returnSingleCharBoxes?: boolean
  /** Enables the 180-degree direction classifier by default. Defaults to true, matching RapidOCR. */
  useClassifier?: boolean
  /** Resizes inputs within minSideLength/maxSideLength bounds. Defaults to true. */
  usePreprocessImage?: boolean
  /** Minimum image side before detection. Defaults to 30. */
  minSideLength?: number
  /** Maximum image side before detection. Defaults to 2000. */
  maxSideLength?: number
  /** Adds vertical padding for very short or unusually wide images. Defaults to true. */
  useVerticalPadding?: boolean
  /** Minimum image height used by vertical padding. Defaults to 30. */
  minHeight?: number
  /** Maximum text-line width/height ratio before vertical padding. Defaults to 8; -1 disables ratio limit. */
  widthHeightRatio?: number
}
