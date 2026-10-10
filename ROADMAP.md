# 开发路线

## 首次发布范围

- [x] Node.js 22+、TypeScript ESM 与 ONNX Runtime CPU。
- [x] 默认模型首次使用时自动下载、校验并缓存；支持本地模型路径和关闭自动下载。
- [x] 整页检测、文本行裁切、180° 方向分类及 CTC 识别。
- [x] 文本行直接识别、原图坐标、置信度、调用耗时和会话释放。
- [x] 固定 RapidOCR v3.10.0 模型、字典和测试资源，并校验 SHA-256。
- [x] 以真实模型和上游输出回归两张页面图、低分辨率页面、空白图、放大图、倒置文本行及裁切文本行。
- [x] 配置 npm Trusted Publishing 发布工作流，并确认 npm 包不包含模型和测试文件。

首次发布前需在 GitHub Actions 观察 Node.js 22/24 与 Linux、macOS、Windows 矩阵通过；首个 npm 包需由维护者手动发布以启用 Trusted Publishing。具体步骤见[贡献指南](./CONTRIBUTING.md#release)。

## 后续方向

- 按平台验证 GPU 等 ONNX Runtime 执行提供程序。
- 根据用户需求评估更多语言模型、文字框细分、批量 API 与 PDF 输入。

实现和默认模型以 [RapidOCR](https://github.com/RapidAI/RapidOCR) 为上游参照。真实模型回归方式见[贡献指南](./CONTRIBUTING.md#local-setup)。
