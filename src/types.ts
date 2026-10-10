/** An image encoded as bytes, a local image path, or raw RGB pixels. */
export type ImageInput = string | Uint8Array | RawRgbImage

/** Raw, interleaved RGB pixels. `data` must contain exactly width * height * 3 bytes. */
export interface RawRgbImage {
  data: Uint8Array
  height: number
  width: number
}

/** Options for the PP-OCR compatible text-line recognizer. */
export interface TextRecognizerOptions {
  /** Maximum number of image lines sent to the model in one inference call. */
  batchSize?: number
  /** Defaults to true when the default recognition model or dictionary is needed. */
  autoDownloadModels?: boolean
  /** Path to the character dictionary that matches the recognition model. */
  dictionaryPath?: string
  /** Number of ONNX Runtime CPU threads. The runtime default is used when omitted. */
  intraOpNumThreads?: number
  /** Directory used to cache default models. */
  modelDirectory?: string
  /** Path to a PP-OCR compatible ONNX recognition model. */
  modelPath?: string
}

/** Text and mean per-character CTC confidence for one cropped text line. */
export interface RecognitionResult {
  confidence: number
  text: string
}

export interface Point {
  x: number
  y: number
}

/** A quadrilateral in top-left, top-right, bottom-right, bottom-left order. */
export type TextBox = readonly [Point, Point, Point, Point]

export interface RecognizedLine extends RecognitionResult {
  box: TextBox
}

export interface RapidOCRResult {
  lines: RecognizedLine[]
  timings: {
    classification: number
    detection: number
    recognition: number
  }
}

/** Select the stages used for one recognition call. */
export interface RapidOCRCallOptions {
  /** Applies the optional 180-degree line classifier. Enabled when a classifier model was loaded. */
  useClassifier?: boolean
  /** Finds text regions before recognition. When disabled, the full image is treated as one text line. */
  useDetection?: boolean
}

/** Paths to the PP-OCR models and dictionary used by the full-page pipeline. */
export interface RapidOCROptions {
  /** Maximum number of crops sent to the recognition model in one call. */
  batchSize?: number
  /** Defaults to true when a required default model path is missing. */
  autoDownloadModels?: boolean
  /** Optional PP-OCR 180-degree direction classifier model. */
  classifierModelPath?: string
  /** Path to a PP-OCR compatible DB text detection ONNX model. */
  detectionModelPath?: string
  /** Matching recognition model dictionary file. */
  dictionaryPath?: string
  /** Number of ONNX Runtime CPU threads. The runtime default is used when omitted. */
  intraOpNumThreads?: number
  /** Directory used to cache default models. */
  modelDirectory?: string
  /** Path to a PP-OCR compatible CTC recognition ONNX model. */
  recognitionModelPath?: string
  /** Minimum recognition confidence for a line to be included. Defaults to 0.5. */
  textScoreThreshold?: number
  /** Loads the default 180-degree direction classifier when no classifier path is supplied. */
  useClassifier?: boolean
}
