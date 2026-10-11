#!/usr/bin/env python3
"""Capture upstream output for blank images at preprocessing size boundaries."""

import argparse
import json
from pathlib import Path

from importlib.metadata import version as installed_version

import cv2
import numpy as np
from rapidocr import EngineType, RapidOCR


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
SIZES = ((29, 29), (30, 2000), (2000, 30), (31, 2048))
MODEL_FILES = {
    "detection": "PP-OCRv6_det_small.onnx",
    "classification": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "recognition": "PP-OCRv6_rec_small.onnx",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    model_paths = {stage: MODEL_DIR / name for stage, name in MODEL_FILES.items()}
    for model_path in (*model_paths.values(), MODEL_DIR / "ppocrv6_dict.txt"):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model asset {model_path}; run `pnpm run test:models` first.")

    cases = capture_cases(SIZES, model_paths)

    reference = {
        "upstream": "RapidOCR v3.10.0",
        "referenceRuntime": {
            "engine": "ONNX Runtime CPU",
            "opencv": cv2.__version__,
            "rapidocr": "3.10.0",
        },
        "models": {stage: path.name for stage, path in model_paths.items()},
        "cases": cases,
    }
    reference["referenceRuntime"]["onnxruntime"] = installed_version("onnxruntime")
    output_path = ROOT / "test/fixtures" / "blank-boundary.expected.json"
    output_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote upstream ONNX Runtime blank image references for {len(cases)} boundary sizes")


def capture_cases(sizes: tuple[tuple[int, int], ...], model_paths: dict[str, Path]) -> list[dict[str, object]]:
    params = {
        "Global.model_root_dir": str(MODEL_DIR),
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_path": str(model_paths["detection"]),
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Cls.model_path": str(model_paths["classification"]),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.model_path": str(model_paths["recognition"]),
        "Rec.rec_keys_path": str(MODEL_DIR / "ppocrv6_dict.txt"),
    }
    ocr = RapidOCR(params=params)
    cases = []
    for width, height in sizes:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        result = ocr(image)
        texts = list(result.txts or [])
        boxes = np.asarray(result.boxes).tolist() if result.boxes is not None else []
        if texts or boxes:
            raise RuntimeError(f"Expected a blank result for {width}x{height}, got {texts}")
        cases.append({"height": height, "texts": texts, "type": "empty", "width": width})
    return cases


if __name__ == "__main__":
    main()
