#!/usr/bin/env python3
"""Generate a vertical Chinese/Japanese page and capture pinned upstream output."""

import argparse
import hashlib
from importlib.metadata import version as installed_version
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
FONT_PATH = MODEL_DIR / "chinese_cht.ttf"
FONT_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
    "resources/fonts/chinese_cht.ttf"
)
FONT_SHA256 = "5ce814960d0cdea1dd647180636babc1cf6a0acf0a9a9019424f4689acedd9ea"
OCR_VERSIONS = {
    "PP-OCRv4": OCRVersion.PPOCRV4,
    "PP-OCRv5": OCRVersion.PPOCRV5,
    "PP-OCRv6": OCRVersion.PPOCRV6,
}
MODEL_TYPES = {
    "mobile": ModelType.MOBILE,
    "small": ModelType.SMALL,
}
LANGUAGES = {
    "chinese": {
        "dictionary": "ppocrv6_dict.txt",
        "lang_type": "ch",
        "ocr_version": "PP-OCRv6",
        "model_type": "small",
        "columns": [
            (460, "縱排中文模型測試"),
            (340, "真實頁面識別驗證"),
            (220, "文字方向由上而下"),
            (100, "節點版完整流程"),
        ],
        "models": {
            "onnxruntime": {
                "detection": "PP-OCRv6_det_small.onnx",
                "classification": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                "recognition": "PP-OCRv6_rec_small.onnx",
            },
        },
    },
    "japanese": {
        "dictionary": "japan_PP-OCRv4_rec_mobile-dictionary.txt",
        "lang_type": "japan",
        "ocr_version": "PP-OCRv4",
        "model_type": "mobile",
        "columns": [
            (460, "日本語の文字"),
            (340, "日本語テスト"),
            (220, "上から下へ"),
            (100, "結果を確認"),
        ],
        "models": {
            "onnxruntime": {
                "detection": "multi_PP-OCRv3_det_mobile.onnx",
                "classification": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                "recognition": "japan_PP-OCRv4_rec_mobile.onnx",
            },
        },
    },
}


def ensure_font() -> None:
    if FONT_PATH.is_file() and hashlib.sha256(FONT_PATH.read_bytes()).hexdigest() == FONT_SHA256:
        return

    FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    font_bytes = urllib.request.urlopen(FONT_URL, timeout=180).read()
    if hashlib.sha256(font_bytes).hexdigest() != FONT_SHA256:
        raise RuntimeError("RapidOCR chinese_cht.ttf SHA-256 mismatch")
    FONT_PATH.write_bytes(font_bytes)


def generate_image(language: str) -> Path:
    font = ImageFont.truetype(str(FONT_PATH), 46)
    image = Image.new("RGB", (520, 720), "white")
    draw = ImageDraw.Draw(image)
    for x, text in LANGUAGES[language]["columns"]:
        for row, character in enumerate(text):
            draw.text((x, 58 + row * 78), character, font=font, fill="black", anchor="mm")
    image_path = ROOT / f"test/fixtures/vertical-{language}.png"
    image.save(image_path, format="PNG", optimize=False)
    return image_path


def capture_reference(language: str, image_path: Path, output_path: Path) -> None:
    language_config = LANGUAGES[language]
    model_paths = {stage: MODEL_DIR / name for stage, name in language_config["models"]["onnxruntime"].items()}
    dictionary = MODEL_DIR / language_config["dictionary"]
    engine_type = EngineType.ONNXRUNTIME
    params = {
        "Global.model_root_dir": str(MODEL_DIR),
        "Det.lang_type": language_config["lang_type"],
        "Det.ocr_version": OCR_VERSIONS[language_config["ocr_version"]],
        "Det.model_type": MODEL_TYPES[language_config["model_type"]],
        "Det.engine_type": engine_type,
        "Det.model_path": str(model_paths["detection"]),
        "Cls.engine_type": engine_type,
        "Cls.model_path": str(model_paths["classification"]),
        "Rec.engine_type": engine_type,
        "Rec.lang_type": language_config["lang_type"],
        "Rec.ocr_version": OCR_VERSIONS[language_config["ocr_version"]],
        "Rec.model_type": MODEL_TYPES[language_config["model_type"]],
        "Rec.model_path": str(model_paths["recognition"]),
        "Rec.rec_keys_path": str(dictionary),
    }
    ocr = RapidOCR(params=params)
    try:
        result = ocr(str(image_path))
    finally:
        del ocr

    if result.boxes is None or result.txts is None or result.scores is None:
        raise RuntimeError("Upstream RapidOCR returned no vertical-page OCR result")
    if not (len(result.boxes) == len(result.txts) == len(result.scores)):
        raise RuntimeError("Upstream vertical-page result lengths differ")

    lines = [
        {
            "box": np.asarray(box, dtype=float).tolist(),
            "confidence": float(score),
            "text": text,
        }
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
    ]
    if len(lines) < len(language_config["columns"]) or any(not line["text"].strip() for line in lines):
        raise RuntimeError(f"Expected at least four non-empty vertical {language} detections, got {len(lines)}")

    reference_runtime = {
        "engine": "ONNX Runtime CPU",
        "opencv": cv2.__version__,
        "rapidocr": "3.10.0",
    }
    reference_runtime["onnxruntime"] = installed_version("onnxruntime")

    reference = {
        "upstream": "RapidOCR v3.10.0",
        "input": {
            "file": image_path.name,
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "font": FONT_PATH.name,
            "fontSha256": FONT_SHA256,
        },
        "language": language,
        "referenceRuntime": reference_runtime,
        "models": {stage: path.name for stage, path in model_paths.items()},
        "lines": lines,
    }
    output_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote vertical {language} page and Python reference: {len(lines)} lines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LANGUAGES), default="chinese")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    language_config = LANGUAGES[args.language]
    model_paths = (MODEL_DIR / name for name in language_config["models"]["onnxruntime"].values())
    dictionary = MODEL_DIR / language_config["dictionary"]
    for model_path in (*model_paths, dictionary):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model asset {model_path}; run `pnpm run test:models` first.")
    ensure_font()
    image_path = generate_image(args.language)
    output_path = args.output or ROOT / "test/fixtures" / f"vertical-{args.language}.expected.json"
    capture_reference(args.language, image_path, output_path)


if __name__ == "__main__":
    main()
