#!/usr/bin/env python3
"""Capture a second Chinese full-page reference for each ONNX Runtime recognizer route."""

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
FIXTURE = "check_return_word_len.jpeg"
FIXTURE_URL = "https://raw.githubusercontent.com/RapidAI/RapidOCR/v3.10.0/python/tests/test_files/check_return_word_len.jpeg"
FIXTURE_SHA256 = "c369e8b92dce43e4e525be90f4766de3203e208616003fbd438b0ff08a529b14"
DEFAULT_OUTPUT = ROOT / "test/fixtures/check_return_word_len-recognition-routes.expected.json"
ROUTES = (
    ("PP-OCRv4", "mobile", "ch_PP-OCRv4_rec_mobile"),
    ("PP-OCRv4", "server", "ch_PP-OCRv4_rec_server"),
    ("PP-OCRv5", "mobile", "ch_PP-OCRv5_rec_mobile"),
    ("PP-OCRv5", "server", "ch_PP-OCRv5_rec_server"),
    ("PP-OCRv6", "tiny", "multi_PP-OCRv6_rec_tiny"),
    ("PP-OCRv6", "small", "multi_PP-OCRv6_rec_small"),
    ("PP-OCRv6", "medium", "multi_PP-OCRv6_rec_medium"),
)

ROUTE_BY_KEY = {
    f"rec/{route[0]}/{route[2]}": route
    for route in ROUTES
}


def prepare_fixture(model_dir: Path) -> Path:
    import hashlib
    import urllib.request

    path = model_dir / FIXTURE
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA256:
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(FIXTURE_URL, timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != FIXTURE_SHA256:
            raise RuntimeError(f"SHA-256 mismatch while downloading {FIXTURE}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_route(
    route: tuple[str, str, str],
    image_path: Path,
    model_dir: Path,
    upstream_python: Path,
) -> dict[str, Any]:
    if str(upstream_python) not in sys.path:
        sys.path.insert(0, str(upstream_python))
    from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    version_name, model_type, model_key = route
    engine_type = EngineType.ONNXRUNTIME
    params = {
        "Global.model_root_dir": str(model_dir / "python-check-return-route-reference-onnxruntime"),
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
        "Rec.lang_type": LangRec("ch"),
    }
    ocr = RapidOCR(params=params)
    try:
        result = ocr(str(image_path))
    finally:
        del ocr
        gc.collect()

    if result.boxes is None or result.scores is None or result.txts is None:
        raise RuntimeError(f"RapidOCR returned no OCR output for {model_key}")
    boxes = np.asarray(result.boxes, dtype=float).tolist()
    scores = [float(score) for score in result.scores]
    texts = list(result.txts)
    if not texts or not (len(boxes) == len(scores) == len(texts)):
        raise RuntimeError(f"RapidOCR returned an invalid full-page result for {model_key}")
    return {
        "fixture": FIXTURE,
        "language": "ch",
        "lines": [
            {"box": box, "confidence": score, "text": text}
            for box, score, text in zip(boxes, scores, texts, strict=True)
        ],
        "modelKey": model_key,
        "modelType": model_type,
        "version": version_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="RapidOCR v3.10.0 source checkout")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, help="Output JSON path; defaults to an engine-specific fixture")
    parser.add_argument("--only", action="append", default=[], help="Capture one route key; repeatable")
    args = parser.parse_args()

    upstream_python = args.upstream.resolve() / "python"
    if not (upstream_python / "rapidocr").is_dir():
        parser.error(f"RapidOCR Python source was not found under {upstream_python}")
    model_dir = args.model_dir.resolve()
    output_path = args.output or DEFAULT_OUTPUT
    image_path = prepare_fixture(model_dir)
    runtime_version = version("onnxruntime")
    models = {}
    default_keys = [
        f"rec/{version_name}/{model_key}"
        for version_name, _, model_key in ROUTES
    ]
    route_keys = args.only or default_keys
    unknown_keys = sorted(set(route_keys) - ROUTE_BY_KEY.keys())
    if unknown_keys:
        parser.error(f"Unknown full-page route keys: {', '.join(unknown_keys)}")
    for route_key in route_keys:
        route = ROUTE_BY_KEY[route_key]
        models[route_key] = run_route(
            route, image_path, model_dir, upstream_python
        )
        print(f"Captured ONNX Runtime {route_key}", flush=True)
    reference = {
        "engine": "ONNX Runtime CPU",
        "fixture": FIXTURE,
        "models": models,
        "pipeline": {
            "classifier": "PP-OCRv4 ch mobile",
            "detection": "PP-OCRv6 ch small",
        },
        "referenceRuntime": {
            "numpy": np.__version__,
            "onnxruntime": runtime_version,
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "rapidocr": version("rapidocr"),
        },
        "upstream": "RapidOCR v3.10.0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(models)} ONNX Runtime full-page references to {output_path}")


if __name__ == "__main__":
    main()
