# RapidOCR Node

面向 Node.js 的 RapidOCR 实现。使用 ONNX Runtime 在本地运行 PP-OCR 模型，支持中文、英文及数字识别。

[![CI](https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml/badge.svg)](https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/rapidocr-node)](https://www.npmjs.com/package/rapidocr-node)
[![License](https://img.shields.io/github/license/OSpoon/RapidOCRNode)](./LICENSE.md)

## 📝 简介

RapidOCR Node 按照 [RapidOCR](https://github.com/RapidAI/RapidOCR) 的模型与处理流程实现，可在 Node.js 应用中本地运行文本检测、方向分类和文本识别。

首次初始化时，如果默认模型尚未缓存，程序会下载并校验 PP-OCRv6 small 模型及字典。模型保存在操作系统缓存目录，后续推理可离线运行；npm 安装不会下载模型。

## 📦 安装

```sh
npm install rapidocr-node
```

需要 Node.js 22 或更高版本。推理使用 ONNX Runtime CPU。首次下载默认模型需要联网；也可预先调用 `downloadModels()`，或指定本地模型路径和缓存目录。

## 🚀 快速开始

```ts
import { RapidOCR } from 'rapidocr-node'

const ocr = await RapidOCR.create()

try {
  const result = await ocr.recognize('./receipt.jpg')
  console.log(result.lines)
}
finally {
  await ocr.close()
}
```

输入可使用本地文件路径、编码后的 `Buffer` / `Uint8Array`，或 RGB 像素数据。识别结果包含文字、置信度、原图坐标文本框和各阶段耗时。可用 `useDetection: false` 识别已裁切的单行文字。

单独识别文本行时可使用 `TextRecognizer`：

```ts
import { TextRecognizer } from 'rapidocr-node'

const recognizer = await TextRecognizer.create()

try {
  console.log(await recognizer.recognizeLine('./text-line.png'))
}
finally {
  await recognizer.close()
}
```

## 🧠 模型

默认模型和配套字典来自 RapidOCR v3.10.0，首次使用时自动下载并校验 SHA-256：

- `PP-OCRv6_det_small.onnx`：文本检测
- `PP-OCRv6_rec_small.onnx` 与 `ppocrv6_dict.txt`：文本识别
- `ch_ppocr_mobile_v2.0_cls_mobile.onnx`：可选的 180° 方向分类；设置 `useClassifier: true` 可自动下载并加载，或通过 `classifierModelPath` 指定本地文件

默认缓存目录遵循 macOS、Windows 和 Linux 的用户缓存目录规则，也可通过 `modelDirectory` 或 `RAPIDOCR_MODEL_DIR` 指定。设置 `autoDownloadModels: false` 可禁止自动下载；此时须传入所需的模型路径。

需要提前下载全部默认模型时，可调用：

```ts
import { downloadModels, RapidOCR } from 'rapidocr-node'

const models = await downloadModels({ modelDirectory: './models' })
const ocr = await RapidOCR.create({ ...models })
```

模型来源、SHA-256 和许可信息见[模型与测试资源说明](./MODEL-LICENSES.md)。

## 🧪 开发与测试

```sh
pnpm install
pnpm lint && pnpm knip && pnpm test && pnpm typecheck && pnpm build
pnpm test:e2e
```

端到端命令会自动下载并校验真实 ONNX 模型及测试图片，再将输出与 RapidOCR v3.10.0 参考结果比对。贡献和发布流程见[贡献指南](./CONTRIBUTING.md)。

## 📜 许可证

代码采用 Apache-2.0 许可证。RapidOCR 上游归属见 [NOTICE.md](./NOTICE.md)；模型、字典和测试图片的说明见 [MODEL-LICENSES.md](./MODEL-LICENSES.md)。
