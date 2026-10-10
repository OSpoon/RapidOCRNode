# Model and test asset provenance

The runtime downloads required default models on first use and checks each
file against its pinned SHA-256. `pnpm test:e2e` prepares the same models plus
test fixtures automatically. The shared pins live in `model-assets.mjs`. Model
files and test images are not included in the npm package.

## Models and dictionary

The ONNX models and PP-OCRv6 recognition dictionary below are fetched from the
RapidOCR v3.10.0 ModelScope repository. RapidOCR's model attribution notice
identifies the PaddleOCR-derived ONNX artifacts as Apache-2.0 and attributes
the model weights to Baidu and/or the relevant PaddleOCR rights holders. The
dictionary is part of the RapidOCR v3.10.0 distribution, which is licensed
under Apache-2.0.

| File | Pinned source | SHA-256 | Upstream attribution and license |
| --- | --- | --- | --- |
| `PP-OCRv6_det_small.onnx` | [ModelScope v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx) | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | Baidu and/or PaddleOCR rights holders; Apache-2.0 |
| `PP-OCRv6_rec_small.onnx` | [ModelScope v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx) | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | Baidu and/or PaddleOCR rights holders; Apache-2.0 |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | [ModelScope v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx) | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | Baidu and/or PaddleOCR rights holders; Apache-2.0 |
| `ppocrv6_dict.txt` | [ModelScope v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt) | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` | RapidOCR distribution; Apache-2.0 |

References: [RapidOCR v3.10.0 model registry](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/rapidocr/default_models.yaml), [RapidOCR v3.10.0 license](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/LICENSE), and [RapidOCR model attribution notice](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/).

## Test images

These images are downloaded only for regression tests. They are not included
in the npm package. Their separate image licensing is not asserted here.

| File | Pinned source | SHA-256 |
| --- | --- | --- |
| `ch_en_num.jpg` | [ModelScope v3.10.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/resources/test_files/ch_en_num.jpg) | `b94c1bf68af9ceb3c550b86931d7b48d4181c26c9f5ace2492e64042f36a02ca` |
| `text_cls.jpg` | [RapidOCR v3.10.0](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/tests/test_files/text_cls.jpg) | `872200f57a1408e7aab2856d5f2c687b3a937805e0c4ff74bd7de21df1f742b9` |
| `text_rec.jpg` | [RapidOCR v3.10.0](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/tests/test_files/text_rec.jpg) | `05921c624274eedbfc1208d12eba5b06acbdd1291d1b1a2129770199f430d6ae` |
| `ch_doc_server.png` | [RapidOCR v3.10.0](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/tests/test_files/ch_doc_server.png) | `8b5726a172d848dbbc679fef93cf7a521cd67914ebf13ff4d43049c0b6cd9f73` |
