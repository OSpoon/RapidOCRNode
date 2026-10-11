# 模型及测试资源

运行时按模型路由从 RapidOCR v3.10.0 的 ModelScope 仓库下载 ONNX 模型和识别字典。下载文件不包含在 npm 包中；注册表为每项资源固定来源 URL 与 SHA-256，供下载时校验。完整路由和校验值见 [`src/models/registry-data.ts`](./src/models/registry-data.ts)。

RapidOCR 上游将 PP-OCR 模型资源标注为 Apache-2.0，并说明模型权利归模型原始权利人及相关 PaddleOCR 权利方所有。格式、许可和归属以[上游模型许可说明](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)及[上游 v3.10.0 许可文件](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/LICENSE)为准。本项目不主张模型权利，也不把模型作为 npm 包的一部分再分发。

## 可视化字体

可视化字体按需从 RapidOCR v3.10.0 资源仓库下载，并按注册表中的 SHA-256 校验。字体有各自的权利和许可条款；上游模型许可说明不将字体纳入模型许可。使用者应遵守对应字体许可。

## 测试资源

测试图像、合成图像及上游参考输出用于回归验证，不随 npm 包发布。上游测试图片的权利和许可仍归相应来源方；参考输出仅用于比较实现结果。资源清单和生成方式见 [`test/fixtures/README.md`](./test/fixtures/README.md)。
