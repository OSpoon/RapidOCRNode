# 回归测试资源

本目录保存与 RapidOCR v3.10.0 对照的参考结果和少量合成图片。测试图片及 ONNX 模型按需下载到本机模型缓存，不放进 npm 包。

## 上游清单

- `rapidocr-v3.10.0-default_models.yaml` 是上游原始模型路由清单，用于核对语言、别名、阶段和模型键。
- `rapidocr-v3.10.0-config.yaml` 是上游原始默认配置，用于确认配置读取器可导入默认 YAML。文件中的其他引擎默认区块只是兼容性测试数据，不表示本项目实现了这些后端。

## 参考输出

- `rapidocr-v3.10.0-model-reference.json`：42 个 ONNX 阶段路由的 Python RapidOCR 参考结果。
- `rapidocr-v3.10.0-recognition-fullpages.expected.json`：30 个识别路由的整页识别参考结果。
- `rapidocr-v3.10.0-language-pages.expected.json`：多语言整页识别参考结果。
- `onnx-word-boxes.expected.json` 和 `onnx-single-char-word-boxes.expected.json`：中文、英文词框及逐字符框参考。
- 其余 `*.expected.json`：边界尺寸、纵排文本、模型变体、图像解码和识别选项等回归结果。

路由矩阵由 `test/model-matrix/routes.test.ts` 执行；Python 参考生成脚本位于 `test/model-matrix/generate_*.py`。生成器参数可通过 `python <script> --help` 查看。更新参考时记录 RapidOCR、ONNX Runtime 和 Python 依赖版本，并确保模型文件按清单校验通过。

测试资源来源和许可边界见 [`MODEL-LICENSES.md`](../../MODEL-LICENSES.md)。
