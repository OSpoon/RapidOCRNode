#!/usr/bin/env python3
"""Capture the pinned RapidOCR full-page reference for its registration scan."""

import argparse
import hashlib
from importlib.metadata import version as installed_version
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.utils.typings import TaskType


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
IMAGE_NAME = "check_return_word_len.jpeg"
IMAGE_SHA256 = "c369e8b92dce43e4e525be90f4766de3203e208616003fbd438b0ff08a529b14"
DETECTION_KEY = "multi_PP-OCRv6_det_small"
CLASSIFIER_KEY = "ch_ppocr_mobile_v2.0_cls_mobile"
RECOGNITION_KEY = "multi_PP-OCRv6_rec_small"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runtime = {
        "rapidocr": installed_version("rapidocr"),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
    }
    expected_runtime = {"rapidocr": "3.10.0", "onnxruntime": "1.30.0", "opencv": "5.0.0"}
    if runtime != expected_runtime:
        raise RuntimeError(f"Reference generation requires {expected_runtime}; found {runtime}")

    image_path = MODEL_DIR / IMAGE_NAME
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing {image_path}; run `pnpm test:e2e` to download pinned fixtures.")
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_sha256 != IMAGE_SHA256:
        raise RuntimeError(f"Unexpected SHA-256 for {IMAGE_NAME}: {image_sha256}")

    upstream_config = yaml.safe_load(
        (ROOT / "test/fixtures/rapidocr-v3.10.0-default_models.yaml").read_text(encoding="utf-8")
    )
    detection_path = model_path(upstream_config, "det", DETECTION_KEY)
    classifier_path = model_path(upstream_config, "cls", CLASSIFIER_KEY)
    recognition_path = model_path(upstream_config, "rec", RECOGNITION_KEY)
    dictionary_path = dictionary_path_for(upstream_config, RECOGNITION_KEY)
    for path in (detection_path, classifier_path, recognition_path, dictionary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model asset {path}; run `pnpm test:e2e` first.")

    params = {
        "Global.model_root_dir": str(MODEL_DIR),
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.task_type": TaskType.DET,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.model_type": ModelType.SMALL,
        "Det.lang_type": LangRec.CH,
        "Det.model_path": str(detection_path),
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.task_type": TaskType.CLS,
        "Cls.ocr_version": OCRVersion.PPOCRV4,
        "Cls.model_type": ModelType.MOBILE,
        "Cls.lang_type": LangRec.CH,
        "Cls.model_path": str(classifier_path),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.task_type": TaskType.REC,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": ModelType.SMALL,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_path": str(recognition_path),
        "Rec.rec_keys_path": str(dictionary_path),
    }
    ocr = RapidOCR(params=params)
    try:
        result = ocr(str(image_path))
    finally:
        del ocr

    if result.boxes is None or result.txts is None or result.scores is None:
        raise RuntimeError("Upstream RapidOCR returned no full-page OCR result")
    if not (len(result.boxes) == len(result.txts) == len(result.scores)):
        raise RuntimeError("Upstream registration scan result lengths differ")

    lines = [
        {
            "text": text,
            "confidence": float(score),
            "box": np.asarray(box, dtype=float).tolist(),
        }
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
    ]
    if not lines or any(not line["text"].strip() for line in lines):
        raise RuntimeError("Expected non-empty recognized lines from the registration scan")

    reference = {
        "upstream": "RapidOCR v3.10.0",
        "input": {"file": IMAGE_NAME, "sha256": image_sha256},
        "referenceRuntime": {
            "python": platform.python_version(),
            **runtime,
            "numpy": np.__version__,
        },
        "models": {
            "detection": detection_path.name,
            "classification": classifier_path.name,
            "recognition": recognition_path.name,
            "dictionary": dictionary_path.name,
        },
        "lines": lines,
    }
    output_path = args.output or ROOT / "test/fixtures/check_return_word_len.expected.json"
    output_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} upstream lines to {output_path}")


def model_path(config: dict, stage: str, model_key: str) -> Path:
    url = config["onnxruntime"]["PP-OCRv6" if stage != "cls" else "PP-OCRv4"][stage][model_key]["model_dir"]
    return MODEL_DIR / Path(url).name


def dictionary_path_for(config: dict, model_key: str) -> Path:
    url = config["paddle"]["PP-OCRv6"]["rec"][model_key]["dict_url"]
    return MODEL_DIR / Path(url).name


if __name__ == "__main__":
    main()
