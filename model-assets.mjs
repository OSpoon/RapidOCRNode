export const MODEL_VERSION = 'v3.10.0'

export const MODEL_ASSETS = {
  detectionModelPath: {
    file: 'PP-OCRv6_det_small.onnx',
    sha256: '090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f',
    url: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx',
  },
  recognitionModelPath: {
    file: 'PP-OCRv6_rec_small.onnx',
    sha256: '6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884',
    url: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx',
  },
  classifierModelPath: {
    file: 'ch_ppocr_mobile_v2.0_cls_mobile.onnx',
    sha256: 'e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c',
    url: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx',
  },
  dictionaryPath: {
    file: 'ppocrv6_dict.txt',
    sha256: 'b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d',
    url: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt',
  },
}

export const TEST_FIXTURE_ASSETS = [
  {
    file: 'ch_en_num.jpg',
    sha256: 'b94c1bf68af9ceb3c550b86931d7b48d4181c26c9f5ace2492e64042f36a02ca',
    url: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/resources/test_files/ch_en_num.jpg',
  },
  {
    file: 'text_cls.jpg',
    sha256: '872200f57a1408e7aab2856d5f2c687b3a937805e0c4ff74bd7de21df1f742b9',
    url: 'https://raw.githubusercontent.com/RapidAI/RapidOCR/v3.10.0/python/tests/test_files/text_cls.jpg',
  },
  {
    file: 'text_rec.jpg',
    sha256: '05921c624274eedbfc1208d12eba5b06acbdd1291d1b1a2129770199f430d6ae',
    url: 'https://raw.githubusercontent.com/RapidAI/RapidOCR/v3.10.0/python/tests/test_files/text_rec.jpg',
  },
  {
    file: 'ch_doc_server.png',
    sha256: '8b5726a172d848dbbc679fef93cf7a521cd67914ebf13ff4d43049c0b6cd9f73',
    url: 'https://raw.githubusercontent.com/RapidAI/RapidOCR/v3.10.0/python/tests/test_files/ch_doc_server.png',
  },
]
