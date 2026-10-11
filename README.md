# RapidOCR Node

面向 Node.js 的 RapidOCR 实现，按 [RapidOCR](https://github.com/RapidAI/RapidOCR) 的模型路由和 OCR 流程处理文本检测、方向分类与文字识别。推理使用 [ONNX Runtime Node](https://github.com/microsoft/onnxruntime)，默认在 CPU 上运行。

[![CI](https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml/badge.svg)](https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/rapidocr-node)](https://www.npmjs.com/package/rapidocr-node)
[![License](https://img.shields.io/github/license/OSpoon/RapidOCRNode)](./LICENSE.md)

## 安装

```sh
npm install rapidocr-node
```

需要 Node.js 24 或更高版本。首次 OCR 会按所选模型路由下载所需 ONNX 模型和识别字典，校验 SHA-256 后放入本机缓存；npm 安装和实例创建本身不会下载模型。模型可预先用 `downloadModels()` 下载，也可指定本地模型路径。

CPU 是默认执行提供程序。可通过 `executionProviders` 配置 ONNX Runtime 执行提供程序；是否可用取决于当前 `onnxruntime-node` 构建、操作系统和硬件。不可用的提供程序会发出警告并回退到 CPU。

## 快速开始

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

输入支持本地文件路径、HTTP(S) URL、`Buffer` / `Uint8Array`，以及指定像素格式的原始图像数据。默认执行检测、方向分类和识别。结果含 `lines`、`boxes`、`txts`、`scores`、`elapseList`、阶段 `timings`，完整 OCR 结果还支持 `toJSON()`、`toMarkdown()` 和 `vis()`。

```ts
const result = await ocr.recognize({
  data: bgrPixels,
  format: 'BGR24',
  width,
  height,
  stride: width * 3,
})
```

## 模型

模型路由对应 RapidOCR v3.10.0 的 `default_models.yaml`，覆盖 PP-OCRv4、PP-OCRv5 和 PP-OCRv6 中上游登记的 ONNX 检测、分类及识别模型。默认使用 PP-OCRv6 small 中文检测和识别模型，以及 PP-OCRv4 mobile 方向分类模型。可用语言通过 `listSupportedLanguages(stage, version, modelType?)` 查询。

识别字典会与所选识别模型一同下载。模型和字典的来源、许可说明及完整 SHA-256 清单见 [`MODEL-LICENSES.md`](./MODEL-LICENSES.md) 和源码模型注册表。

## API 和 CLI

```ts
import { downloadModels, listSupportedLanguages, RapidOCR } from 'rapidocr-node'

const paths = await downloadModels()
console.log(paths)
console.log(listSupportedLanguages('rec', 'PP-OCRv5', 'mobile'))

const ocr = await RapidOCR.create({
  executionProviders: ['cpu'],
  returnWordBoxes: true,
})
```

命令行提供图片识别、配置生成、模型下载、语言查询和安装检查：

```sh
rapidocr-node receipt.jpg
rapidocr-node receipt.jpg --output markdown --vis
rapidocr-node config --save-config rapidocr.yaml
rapidocr-node download-models
rapidocr-node check
rapidocr-node --list-langs --task rec --ocr-version PP-OCRv5 --model-type mobile
```

CLI 参数和配置文件可通过以下命令查看：

```sh
rapidocr-node --help
rapidocr-node config --help
```

## 开发和验证

```sh
pnpm install
pnpm run ci
pnpm run test:e2e
pnpm run test:models
```

`test:e2e` 对照 RapidOCR v3.10.0 固定参考验证完整流程。`test:models` 使用真实图片验证上游登记的 42 个 ONNX 阶段模型路由及 30 个识别路由的整页结果；可选执行提供程序需在匹配的系统和硬件上单独验证。覆盖范围和待完成事项见 [`ROADMAP.md`](./ROADMAP.md)。

## 许可

本项目代码按 Apache-2.0 许可发布。RapidOCR 上游代码、模型、字典、字体和测试图片分别保留其上游许可与归属信息；详见 [`NOTICE.md`](./NOTICE.md) 和 [`MODEL-LICENSES.md`](./MODEL-LICENSES.md)。本项目不是 RapidOCR 官方发行版。
