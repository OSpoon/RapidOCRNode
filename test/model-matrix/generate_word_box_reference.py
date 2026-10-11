#!/usr/bin/env python3
"""Capture full-page inference-backend word-box references from RapidOCR Python."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
FULLPAGE_REFERENCES = ROOT / "test/fixtures/rapidocr-v3.10.0-recognition-fullpages.expected.json"
DEFAULT_OUTPUT = ROOT / "test/fixtures/onnx-word-boxes.expected.json"
DEFAULT_SINGLE_CHAR_OUTPUT = ROOT / "test/fixtures/onnx-single-char-word-boxes.expected.json"
ROUTES = {
    "rec/PP-OCRv4/ch_PP-OCRv4_rec_mobile": ("PP-OCRv4", "mobile", "ch_PP-OCRv4_rec_mobile", "ch"),
    "rec/PP-OCRv4/en_PP-OCRv4_rec_mobile": ("PP-OCRv4", "mobile", "en_PP-OCRv4_rec_mobile", "en"),
}


def run_route(
    key: str,
    reference: dict[str, Any],
    model_dir: Path,
    upstream_python: Path,
    return_single_char_box: bool,
) -> dict[str, Any]:
    if str(upstream_python) not in sys.path:
        sys.path.insert(0, str(upstream_python))
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    version_name, model_type, model_key, language = ROUTES[key]
    engine_type = EngineType.ONNXRUNTIME
    params = {
        "Global.model_root_dir": str(model_dir / "python-onnx-word-box-reference"),
        "Global.return_word_box": True,
        "Global.return_single_char_box": return_single_char_box,
        "Det.engine_type": engine_type,
        "Det.ocr_version": OCRVersion("PP-OCRv6"),
        "Det.model_type": ModelType("small"),
        "Det.lang_type": LangDet("ch"),
        "Cls.engine_type": engine_type,
        "Cls.ocr_version": OCRVersion("PP-OCRv4"),
        "Cls.model_type": ModelType("mobile"),
        "Cls.lang_type": LangCls("ch"),
        "Rec.engine_type": engine_type,
        "Rec.ocr_version": OCRVersion(version_name),
        "Rec.model_type": ModelType(model_type),
        "Rec.lang_type": LangRec(language),
    }
    ocr = RapidOCR(params=params)
    try:
        result = ocr(str(model_dir / reference["fixture"]))
    finally:
        del ocr
        gc.collect()

    if result.boxes is None or result.scores is None or result.txts is None or result.word_results is None:
        raise RuntimeError(f"RapidOCR returned incomplete word-box output for {key}")
    boxes = np.asarray(result.boxes, dtype=float).tolist()
    scores = [float(score) for score in result.scores]
    texts = list(result.txts)
    if not (len(boxes) == len(scores) == len(texts) == len(result.word_results)):
        raise RuntimeError(f"RapidOCR returned mismatched OCR and word-box lines for {key}")
    lines = []
    for box, score, text, words in zip(boxes, scores, texts, result.word_results, strict=True):
        lines.append({
            "box": box,
            "confidence": score,
            "text": text,
            "words": [
                {"box": word_box, "confidence": float(word_score), "text": word_text}
                for word_text, word_score, word_box in words
                if word_box is not None
            ],
        })
    return {
        "fixture": reference["fixture"],
        "language": language,
        "lines": lines,
        "modelKey": model_key,
        "modelType": model_type,
        "version": version_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="RapidOCR v3.10.0 source checkout")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, help="Output JSON path; defaults to an engine-specific fixture")
    parser.add_argument("--only", action="append", default=[], help="Capture one recognition route; repeatable")
    parser.add_argument("--return-single-char-box", action="store_true", help="Split English and numeric words into characters")
    args = parser.parse_args()

    upstream_python = args.upstream.resolve() / "python"
    if not (upstream_python / "rapidocr").is_dir():
        parser.error(f"RapidOCR Python source was not found under {upstream_python}")
    model_dir = args.model_dir.resolve()
    if args.output:
        output_path = args.output
    else:
        output_path = DEFAULT_SINGLE_CHAR_OUTPUT if args.return_single_char_box else DEFAULT_OUTPUT
    page_references = json.loads(FULLPAGE_REFERENCES.read_text(encoding="utf-8"))["models"]
    keys = args.only or list(ROUTES)
    unknown = sorted(set(keys) - ROUTES.keys())
    if unknown:
        parser.error(f"Unknown ONNX word-box routes: {', '.join(unknown)}")

    models = {}
    for key in keys:
        models[key] = run_route(key, page_references[key], model_dir, upstream_python, args.return_single_char_box)
        print(f"Captured ONNX Runtime {key} ({len(models[key]['lines'])} lines)", flush=True)

    runtime_version = version("onnxruntime")
    payload = {
        "engine": "ONNX Runtime CPU",
        "models": models,
        "referenceRuntime": {
            "onnxruntime": runtime_version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "rapidocr": version("rapidocr"),
        },
        "upstream": "RapidOCR v3.10.0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(models)} ONNX Runtime word-box references to {output_path}")


if __name__ == "__main__":
    main()
