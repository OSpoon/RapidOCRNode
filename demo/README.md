# RapidOCR Node npm 安装演示

这是一个独立的 Node.js 项目，从 npm 安装已发布的 `rapidocr-node@0.1.1`，并用附带的示例图片执行 OCR；它不引用仓库中的本地源码。

需要 Node.js 22 或更高版本。

```sh
cd demo
npm ci
npm run demo
```

也可以传入自己的图片路径：

```sh
npm run demo -- /path/to/receipt.jpg
```

首次运行时，RapidOCR 会自动下载并校验默认 PP-OCRv6 small 模型和字典，缓存在用户目录中；后续运行会复用缓存，支持离线识别。

示例会输出识别文本、置信度和各阶段耗时。`sample.svg` 是项目自带的示例图片，无需额外下载测试图片。

输出示例：

```text
1. RapidOCR Node (confidence: 1.000)
2. Install from npm (confidence: 1.000)
3. Node.js 22+ (confidence: 0.999)
4. Text recognition demo (confidence: 1.000)
```
