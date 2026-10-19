<div align="center">
  <h1>RapidOCR Node</h1>
  <p>面向 Node.js 的 RapidOCR 实现，按 <a href="https://github.com/RapidAI/RapidOCR">RapidOCR</a> 流程执行 OCR，推理由 <a href="https://github.com/microsoft/onnxruntime">ONNX Runtime Node</a> 提供。</p>
  <p>支持 JavaScript API、命令行和 HTTP 服务。</p>
  <p>
    <a href="https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml"><img src="https://github.com/OSpoon/RapidOCRNode/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://www.npmjs.com/package/rapidocr-node"><img src="https://img.shields.io/npm/v/rapidocr-node" alt="npm version"></a>
    <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  </p>
</div>

## 选择使用方式

| 方式 | 适合场景 | 从这里开始 |
| --- | --- | --- |
| JavaScript API | 在 Node.js 程序中调用 OCR，读取文字、置信度和坐标 | [JavaScript API](#javascript-api) |
| CLI | 在终端识别图片、生成配置或下载模型 | [命令行](#命令行) |
| HTTP 服务 | 让其他进程通过 HTTP 上传图片并获取 OCR JSON | [HTTP 服务](#http-服务) |

## 能做什么

- 完整 OCR：文本检测、文本行裁切与排序、可选的 180 度方向分类、文字识别。
- 单独使用检测器、方向分类器或识别器，接入自有图像处理流程。
- 选择 RapidOCR 上游登记的 PP-OCRv4、PP-OCRv5、PP-OCRv6 模型路由、语言和模型规格。
- 使用默认模型、指定本地 ONNX 模型，或注入由调用方创建的 ONNX Runtime session。
- 输入本地图片、HTTP(S) 图片地址、字节数据、Blob 或原始像素。
- 获取文本、置信度、四点坐标、逐字/词信息、阶段耗时；导出 JSON、Markdown 或可视化图片。
- 可选地将 OCR 文字交给独立的小型文本模型整理为字段和值，建议仅用于发票、收据等票据。
- 通过 CLI 运行单张图片任务，或启动、查看、重启和停止 HTTP 服务。

## 安装和模型

需要 Node.js 24 或更高版本：

~~~sh
npm install rapidocr-node
~~~

模型按需下载并缓存。可通过 `modelDirectory`、`RAPIDOCR_MODEL_DIR` 或 CLI 的 `--model-dir` 指定缓存目录。

也可以提前下载模型：

~~~ts
import { downloadModels } from 'rapidocr-node'

const paths = await downloadModels({
  modelDirectory: './models',
})
console.log(paths)
~~~

模型来源、许可证和 SHA-256 清单见 [`MODEL-LICENSES.md`](./MODEL-LICENSES.md)。

## JavaScript API

下面的例子识别一张本地图片，并在使用完后释放模型资源：

~~~ts
import { RapidOCR } from 'rapidocr-node'

const ocr = await RapidOCR.create()
try {
  const result = await ocr.recognize('./receipt.jpg')

  if (result.type === 'ocr') {
    for (const line of result.lines) {
      console.log(line.text, line.confidence, line.box)
    }
  }
  else {
    console.log('没有识别到文字')
  }
}
finally {
  await ocr.close()
}
~~~

### 输入与结果

`recognize()` 接受以下输入：

| 输入 | 示例 |
| --- | --- |
| 本地路径、HTTP(S) URL 或 URL 对象 | `./receipt.jpg`、`https://example.com/page.png`、`new URL(...)` |
| 字节数据 | `Buffer`、`Uint8Array`、`ArrayBuffer`、`Blob` |
| 原始像素 | 带有 `data`、`width`、`height`、`format` 的对象 |

原始像素需要说明像素格式和图像尺寸：

~~~ts
const result = await ocr.recognize({
  data: bgrPixels,
  format: 'BGR24',
  width,
  height,
  stride: width * 3,
})
~~~

完整 OCR 结果的 `lines` 中，每行包含文字、置信度和四点坐标；`boxes` 是 `[N][4][2]` 坐标数组，`img` 是解码后的 BGR 像素。结果还提供与 RapidOCR 对齐的 `txts`、`scores`、`word_results`、`elapse_list` 等字段，以及 Node.js 风格的 `boxPoints`、`wordResults`、`timings`。

~~~ts
const jsonLines = result.toJSON()
const markdown = result.toMarkdown()
await result.vis('./receipt-result.png')
~~~

`toJSON()` / `to_json()` 返回便于保存的文字与坐标数组；`toMarkdown()` / `to_markdown()` 按文本位置生成 Markdown；`vis()` 可保存可视化图片，`visBuffer()` 返回 PNG 字节。

没有检测到文字时，结果类型为 `empty`。通过单阶段调用获得的结果类型可能是 `detection`、`classification` 或 `recognition`。

默认识别调用会执行检测、方向分类和识别。可按单次调用关闭阶段，例如只检测文本框：

~~~ts
const detection = await ocr.recognize('./receipt.jpg', {
  useClassifier: false,
  useRecognition: false,
})
console.log(detection.type, detection.boxes)
~~~

也可以通过 `useClassifier: false` 跳过方向分类。

### 票据字段结构化（可选）

处理发票、收据时，可以在 OCR 后按需要提取字段。普通文档识别直接使用 `recognize()` 即可。

在调用 `rapidocr-node` 的应用项目目录中安装可选依赖：

~~~sh
npm install @huggingface/transformers
~~~

~~~ts
import { RapidOCR } from 'rapidocr-node'
import { OCRTextStructurer } from 'rapidocr-node/structured-output'

const ocr = await RapidOCR.create()
const structurer = new OCRTextStructurer()
try {
  const result = await ocr.recognize('./invoice.jpg')
  const fields = result.type === 'ocr'
    ? await structurer.extract(result.lines, {
        invoiceNumber: '发票号码',
        issueDate: '开票日期',
        buyerName: '购买方名称',
        totalAmount: '价税合计',
      })
    : null

  console.log(fields)
}
finally {
  await Promise.all([ocr.close(), structurer.close()])
}
~~~

`values` 返回提取结果；无法找到的字段为 `null`。`evidence` 标出对应的 OCR 文字行，方便核对。

### 选模型和识别语言

默认完整流程使用 PP-OCRv6 small 中文检测与识别模型，以及 PP-OCRv4 mobile 方向分类模型。各阶段可以分别选择模型：

~~~ts
import { RapidOCR } from 'rapidocr-node'

const ocr = await RapidOCR.create({
  detectionModel: { ocrVersion: 'PP-OCRv6', langType: 'ch', modelType: 'small' },
  classifierModel: { ocrVersion: 'PP-OCRv4', langType: 'ch', modelType: 'mobile' },
  recognitionModel: { ocrVersion: 'PP-OCRv5', langType: 'en', modelType: 'mobile' },
  returnWordBoxes: true,
  executionProviders: ['cpu'],
})
~~~

可通过 `listSupportedLanguages()` 查询某个阶段和模型规格支持的语言：

~~~ts
import { listSupportedLanguages } from 'rapidocr-node'

console.log(listSupportedLanguages('rec', 'PP-OCRv5', 'mobile'))
~~~

### 使用单个处理阶段

需要自行组合图像流程时，可以使用单阶段类：

| 类 | 用途 |
| --- | --- |
| **TextDetector** | 检测图片中的文本区域并返回文本框 |
| **TextClassifier** | 判断文本行是否需要旋转 180 度 |
| **TextRecognizer** | 识别已裁切的文本行 |

各阶段类接受对应的 ONNX 模型路径和推理选项；`TextRecognizer` 还可以按模型路由自动选择识别模型及匹配字典。完整选项与类型定义可在源码的 [`src/shared/types.ts`](./src/shared/types.ts) 和各阶段实现中查看。

例如，单独识别已经裁切好的文本行：

~~~ts
import { TextRecognizer } from 'rapidocr-node'

const recognizer = await TextRecognizer.create({
  modelSelection: { ocrVersion: 'PP-OCRv5', langType: 'en', modelType: 'mobile' },
})
try {
  const lines = await recognizer.recognizeLines(['./cropped-line.png'])
  console.log(lines[0]?.text)
}
finally {
  await recognizer.close()
}
~~~

## 命令行

安装到当前项目后，可用 `npx rapidocr-node` 调用 CLI：

~~~sh
# 输出 OCR JSON
npx rapidocr-node ./receipt.jpg

# 输出 Markdown
npx rapidocr-node ./receipt.jpg --output markdown

# 另存可视化结果，并返回词框
npx rapidocr-node ./receipt.jpg --vis ./receipt-result.png --word-boxes

# 检查安装并对示例图片执行一次 OCR
npx rapidocr-node check
~~~

常用命令：

~~~sh
# 生成完整的 RapidOCR YAML 配置
npx rapidocr-node config --save-config ./rapidocr.yaml

# 按配置下载所需模型
npx rapidocr-node --config ./rapidocr.yaml download-models

# 列出 PP-OCRv5 mobile 识别模型支持的语言
npx rapidocr-node --list-langs --task rec --ocr-version PP-OCRv5 --model-type mobile

# 显示完整帮助
npx rapidocr-node --help
~~~

运行 `npx rapidocr-node --help` 查看图片识别选项。

处理发票或收据时，可选用结构化提取。使用 `npx` 时需同时提供两个依赖：

~~~sh
npx --yes \
  --package=rapidocr-node@latest \
  --package=@huggingface/transformers \
  rapidocr-node ./invoice.jpg \
  --structured-fields '{"invoiceNumber":"发票号码","issueDate":"开票日期","totalAmount":"价税合计"}'
~~~

全局安装方式：

~~~sh
npm install -g rapidocr-node@latest @huggingface/transformers
rapidocr-node ./invoice.jpg --structured-fields '{"invoiceNumber":"发票号码"}'
~~~

结构化提取默认使用 CPU，传入 `--structured-fields` 时按需加载模型；其他选项见 `--help`。

模型版本和规格通过配置文件选择；可用路由通过 `--list-langs` 查询。

生成配置后，编辑 `Det`、`Cls`、`Rec` 下的 `ocr_version`、`lang_type` 和 `model_type`。下面只列出模型路由字段，其他参数继续使用默认值：

~~~yaml
Det:
  ocr_version: PP-OCRv6
  lang_type: ch
  model_type: small
Cls:
  ocr_version: PP-OCRv4
  lang_type: ch
  model_type: mobile
Rec:
  ocr_version: PP-OCRv5
  lang_type: en
  model_type: mobile
~~~

使用该配置识别图片：

~~~sh
npx rapidocr-node --config ./rapidocr.yaml ./receipt.jpg
~~~

## HTTP 服务

HTTP 服务适合让本机或其他应用重复提交图片。服务进程会在启动时加载并预热模型，之后复用同一套 OCR 实例处理请求。

### 启动和管理

默认监听 `127.0.0.1:8000`。`start` 在后台启动，并在模型准备就绪后返回；`run` 在前台运行，按 Ctrl-C 停止：

~~~sh
# 后台启动
npx rapidocr-node serve start

# 查看地址、进程号和日志文件
npx rapidocr-node serve status

# 使用原配置重启，或优雅停止
npx rapidocr-node serve restart
npx rapidocr-node serve stop

# 前台运行
npx rapidocr-node serve run --host 127.0.0.1 --port 8000
~~~

启动时可调整监听地址、端口、模型配置和资源限制：

~~~sh
npx rapidocr-node --config ./rapidocr.yaml serve start \
  --host 127.0.0.1 \
  --port 8000 \
  --max-file-size 10485760 \
  --max-image-pixels 20000000 \
  --max-concurrency 1
~~~

启用结构化提取时，服务环境需安装 `@huggingface/transformers`。可在启动时设置模型、设备和 token 上限：

~~~sh
npx --yes \
  --package=rapidocr-node@latest \
  --package=@huggingface/transformers \
  rapidocr-node serve start \
  --structured-model onnx-community/Qwen3-0.6B-ONNX \
  --structured-max-new-tokens 512

# 如果两个包都全局安装，则直接使用全局 CLI
rapidocr-node serve start \
  --structured-model onnx-community/Qwen3-0.6B-ONNX \
  --structured-max-new-tokens 512
~~~

请求带有 `structured_fields` 时会运行字段提取：

~~~sh
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F 'file=@invoice.jpg' \
  -F 'options={"include_image":false,"structured_fields":{"invoiceNumber":"发票号码","issueDate":"开票日期","totalAmount":"价税合计"}};type=application/json'
~~~

OCR 与结构化结果及耗时均包含在响应中；完整请求和响应定义见 `/openapi.json`。

默认限制为 10 MiB、20 MP 和 1 个并发请求，可用 `--max-file-size`、`--max-image-pixels` 和 `--max-concurrency` 调整。

服务默认仅监听本机且无身份验证；远程访问请在反向代理设置访问控制。

### 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | **/health/live** | 确认服务进程正在运行 |
| GET | **/health/ready** | 确认模型已就绪；未就绪时返回 503 |
| GET | **/openapi.json** | 获取 OpenAPI 3.1 描述 |
| POST | **/v1/ocr** | 上传图片并执行完整 OCR |

`POST /v1/ocr` 使用 `multipart/form-data` 上传图片。

错误采用 Problem Details 格式；服务只接受图片文件，不接受本地路径或图片 URL。

## 模型和执行后端

模型路由与 RapidOCR v3.10.0 对齐，支持 PP-OCRv4、PP-OCRv5 和 PP-OCRv6；可用路由见 **listSupportedLanguages()**。

OCR 和结构化提取默认使用 **CPU**。可在 API 或配置文件中为 OCR 显式指定执行后端，也可通过 API 的 **device** 或 CLI/服务的 **--structured-device** 为结构化提取指定后端。

### 本地速度对照

macOS · Node 24.15.0 · ONNX Runtime 1.30.0 · `test.png`（1588×1060）· 模型已缓存。

OCR 使用 PP-OCRv6 small 检测/识别和 PP-OCRv4 mobile 分类，热运行 12 次：

| 执行后端 | 端到端耗时 P50 / P95 | 识别结果 |
| --- | ---: | --- |
| CPU | 1.756 / 1.807 秒 | 43 行，基准 |
| CoreML | 3.429 / 3.934 秒 | 39 行，文本有差异 |
| WebGPU | 1.109 / 1.147 秒 | 43 行，文本一致 |

结构化提取使用 Qwen3-0.6B Q4F16（Transformers.js 4.3.1），从同一张图片提取 3 个字段；模型已缓存，每个设备测 1 次首次调用和 5 次热运行。

| 结构化设备 | 首次总耗时（模型加载） | 热运行总耗时 P50 / P95 |
| --- | ---: | ---: |
| CPU | 9.31 秒（0.73 秒） | 10.44 / 10.65 秒 |
| CoreML | 12.11 秒（0.97 秒） | 9.65 / 10.53 秒 |
| WebGPU | 3.70 秒（1.41 秒） | 2.11 / 2.14 秒 |

本次 3 个字段结果一致。手动指定设备：API 用 `new OCRTextStructurer({ device: 'webgpu' })`，CLI 或 HTTP 服务启动时用 `--structured-device webgpu`。

## 工作流程

完整 OCR 的主要步骤如下。方向分类默认启用，可按需关闭。

~~~mermaid
flowchart LR
  A[接收请求<br/>JavaScript API / CLI / HTTP] --> B[图片解码与预处理]
  B --> C[文本检测模型]
  C --> D[文本区域裁切与排序]
  D --> E{启用方向分类？}
  E -- 是 --> F[方向分类模型<br/>判断方向，必要时旋转]
  E -- 否 --> G[文字识别模型]
  F --> G
  G --> H[整理并返回结果<br/>文本、分数、坐标与阶段耗时]

  classDef model fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#111827
  classDef process fill:#f8fafc,stroke:#64748b,color:#111827
  classDef decision fill:#fff7ed,stroke:#ea580c,color:#111827
  class A,B,D,H process
  class C,F,G model
  class E decision
~~~

蓝紫色节点表示模型推理，浅灰色节点表示普通处理，橙色节点表示分支判断。方向分类默认启用，可按需关闭。

## 开发和验证

~~~sh
pnpm install
pnpm run ci
pnpm run test:e2e
pnpm run test:models
~~~

`test:e2e` 对照 RapidOCR v3.10.0 固定参考验证端到端结果。`test:models` 使用真实图片验证上游登记的 42 个 ONNX 阶段模型路由及 30 个识别路由；可选执行提供程序需在匹配的系统和硬件上单独验证。覆盖范围见 [`ROADMAP.md`](./ROADMAP.md)。

## 许可

### 项目代码

本项目代码按 Apache-2.0 许可发布，详见 [`LICENSE.md`](./LICENSE.md)。

### 模型与上游资源

RapidOCR 上游代码、模型、字典、字体和测试图片分别保留其上游许可与归属信息。模型来源、校验值和许可见 [`MODEL-LICENSES.md`](./MODEL-LICENSES.md)，其他归属说明见 [`NOTICE.md`](./NOTICE.md)。本项目不是 RapidOCR 官方发行版。
