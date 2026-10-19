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

npm 安装不会下载模型。第一次处理图片时，程序会按所选路由检查模型和识别字典；缺少的文件会下载、校验 SHA-256 并缓存。默认缓存目录按操作系统选择。可通过 JavaScript 的 `modelDirectory` 或环境变量 `RAPIDOCR_MODEL_DIR` 指定目录；CLI 使用 `--model-dir`。

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

常用图片选项：

| 选项 | 作用 |
| --- | --- |
| **--config** | 读取 YAML/JSON 配置；可设置模型版本、规格、阶段和推理参数 |
| **--model-dir** | 指定模型缓存目录 |
| **--lang** | 覆盖配置中的识别语言 |
| **--text-score** | 设置输出文字的最低置信度 |
| **--word-boxes** | 返回中文逐字框和英文/数字词框 |
| **--no-det**、**--no-cls**、**--no-rec** | 关闭检测、方向分类或识别阶段 |
| **--vis [路径]** | 保存可视化图片；省略路径时使用默认文件名 |
| **--output json\|markdown** | 选择 JSON 或 Markdown 输出 |
| **--no-download** | 禁止自动下载缺失模型 |

处理发票或收据时，可额外请求字段提取。字段名和说明可用 JSON 传入，也可以放在 JSON 文件中。若使用 `npx` 临时运行，RapidOCR Node 和结构化模型依赖都要放进这次 `npx` 环境：

~~~sh
npx --yes \
  --package=rapidocr-node@latest \
  --package=@huggingface/transformers \
  rapidocr-node ./invoice.jpg \
  --structured-fields '{"invoiceNumber":"发票号码","issueDate":"开票日期","totalAmount":"价税合计"}'
~~~

如果使用全局 CLI，请把两个包都全局安装，然后运行全局命令：

~~~sh
npm install -g rapidocr-node@latest @huggingface/transformers
rapidocr-node ./invoice.jpg --structured-fields '{"invoiceNumber":"发票号码"}'
~~~

`npx` 会在临时包环境中运行 CLI，因此单独全局安装 `@huggingface/transformers` 不会让它对 `npx rapidocr-node@latest` 可见。请使用上面的双 `--package` 命令，或全局安装两个包后直接运行全局 `rapidocr-node` 命令。

输出对象包含 `ocr` 和 `structured`：`structured.values` 给出字段值，无法从 OCR 文本核对的值为 `null`；`structured.evidence` 给出对应的 OCR 行索引。可用 `--structured-model` 选择文本模型，也可用 `--structured-max-new-tokens` 设置生成上限；默认不限制生成长度。普通识别不传 `--structured-fields` 时不会加载结构化模型。

命令行的默认模型版本和规格写在配置文件中。`--ocr-version`、`--model-type` 与 `--task` 用于 `--list-langs` 查询；运行图片识别时，请通过配置文件选择各阶段的模型路由。

`--no-det` 会把整张图片作为一个文本行送入后续阶段，适用于已裁切好的单行图片；`--no-rec` 则返回检测结果。

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

需要在 HTTP 请求中启用字段提取时，运行服务的同一 npm 环境也需要安装 `@huggingface/transformers`。如果通过 `npx` 临时启动，请同时指定两个包；如果全局安装，则使用全局 `rapidocr-node` 命令。可在启动时选择结构化模型；token 上限是可选的：

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

服务只在请求带有 `structured_fields` 时运行字段提取。字段由请求方按票据类型提供：

~~~sh
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F 'file=@invoice.jpg' \
  -F 'options={"include_image":false,"structured_fields":{"invoiceNumber":"发票号码","issueDate":"开票日期","totalAmount":"价税合计"}};type=application/json'
~~~

启用后，响应会额外包含 `structured.values` 和 `structured.evidence`。`values` 给出字段值，缺失或无法在 OCR 文本中核对的值为 `null`；`evidence` 给出支撑字段值的 OCR 行索引。未传 `structured_fields` 时，响应格式与普通 OCR 相同。

上传大小默认上限为 10 MiB，解码后的图片默认最多 20,000,000 像素，并发 OCR 请求默认上限为 1。可通过 `--max-file-size`、`--max-image-pixels` 和 `--max-concurrency` 调整；图片字节数或像素数超限时返回 **413**，超出并发上限的请求收到 **429** 和 **Retry-After**。如果指定了 `--runtime-dir`，查询、重启和停止时也要指定相同目录。

服务默认只监听本机，API 不提供身份验证。需要从其他机器访问时，请在受信网络中部署，并在反向代理或网络层配置访问控制。

### 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | **/health/live** | 确认服务进程正在运行 |
| GET | **/health/ready** | 确认模型已就绪；未就绪时返回 503 |
| GET | **/openapi.json** | 获取 OpenAPI 3.1 描述 |
| POST | **/v1/ocr** | 上传图片并执行完整 OCR |

`POST /v1/ocr` 使用 `multipart/form-data`：

- **file**：必填，图片文件。
- **options**：可选，包含 OCR 参数的 JSON 字符串。

最小请求（不在响应中包含源图像数据）：

~~~sh
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F 'file=@receipt.jpg' \
  -F 'options={"include_image":false};type=application/json'
~~~

带选项的请求：

~~~sh
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F 'file=@receipt.jpg' \
  -F 'options={"return_word_boxes":true,"include_image":false};type=application/json'
~~~

支持的请求选项：

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| **box_threshold** | 使用模型配置 | 文本框置信度阈值，范围 0 到 1 |
| **unclip_ratio** | 使用模型配置 | 扩展文本框的比例，必须大于或等于 0 |
| **text_score_threshold** | 使用模型配置 | 识别结果最低置信度，范围 0 到 1 |
| **return_word_boxes** | false | 返回中文逐字框和英文/数字词框 |
| **return_single_char_boxes** | false | 将英文/数字词框拆为单字符框 |
| **include_image** | true | 是否在响应中包含图像像素；设为 false 可减小响应 |
| **structured_fields** | 不启用 | 按字段名映射说明，例如 `{"invoiceNumber":"发票号码"}`；建议用于发票和收据 |

成功响应包含 **request_id**、**type**、**boxes**、**txts**、**scores**、**word_results**、**lang_rec** 和 **elapse_list** 等字段。**boxes[i]**、**txts[i]** 和 **scores[i]** 对应同一行结果；**elapse_list** 的阶段顺序为检测、分类、识别，单位为秒。默认的 **img.data** 是 base64 编码的 BGR 像素；设置 **include_image: false** 后 **img** 为 null。无文字时 **type** 为 **empty**。

错误使用 Problem Details JSON 格式，包含错误码和请求 ID。常见状态码有 400（请求参数无效）、413（上传超限）、415（媒体类型不支持）、422（图片无法解码）、429（并发已满）和 503（模型未就绪）。服务只接受上传的图片文件，不接受客户端传来的本地路径或图片 URL。

## 模型和执行后端

模型路由对应 RapidOCR v3.10.0 的 **default_models.yaml**，包括上游登记的 PP-OCRv4、PP-OCRv5 和 PP-OCRv6 ONNX 检测、分类与识别模型。各阶段可单独选择版本、语言和规格；并非每种组合都受支持，可用路由通过 **listSupportedLanguages()** 查询。

ONNX Runtime 的 CPU 执行提供程序默认启用。JavaScript API 可通过 **executionProviders** 设置执行提供程序及优先级；按阶段配置时可分别使用 **detectionExecutionProviders**、**classifierExecutionProviders** 和 **recognitionExecutionProviders**。实际可用项取决于安装的 ONNX Runtime 构建、操作系统和硬件。

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
