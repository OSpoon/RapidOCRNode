# 路线图

项目目标是用 Node.js 实现 RapidOCR v3.10.0 的模型路由、图像处理和 OCR 结果流程。推理后端集中为 ONNX Runtime，避免维护多套原生适配层；ONNX Runtime 自身提供的执行提供程序按平台和硬件逐步验证。

## 已完成

- 推理代码、公开选项、CLI、依赖和 GitHub Actions 只保留 ONNX Runtime 路径。
- 默认使用 ONNX Runtime CPU；支持通过 ONNX Runtime `executionProviders` 选项指定运行时已提供的执行提供程序，并在不可用时回退到 CPU。
- 按上游 v3.10.0 路由表登记 PP-OCRv4、PP-OCRv5、PP-OCRv6 的 42 个 ONNX 阶段模型组合，模型和字典首次使用时按需下载并校验 SHA-256。
- 建立真实图片回归：42 个阶段路由与固定 Python RapidOCR 参考对照，30 个识别路由对照整页识别结果；另有词框、逐字符框、竖排文本、图像边界和配置导入用例。
- 包要求 Node.js 24 或更高版本；CI 配置 Ubuntu、macOS、Windows 上的 Node.js 24 和 26。
- 主 OCR 流程及独立检测、分类、识别结果提供上游字段名和坐标结构；Node 输入支持路径、URL、编码字节和原始像素数据的原生等价形式，结果图像以 BGR 像素对象返回，并保留 Node 风格别名。

## 本机验证记录

2026-10-11 在 macOS arm64、Node.js 24.15.0 上：`pnpm run ci` 全部通过（94 项单测）；`pnpm run test:e2e` 通过 132 项、跳过 3 项硬件用例；设置 `RAPIDOCR_TEST_COREML=1` 和 `RAPIDOCR_TEST_WEBGPU=1` 后，`pnpm run test:models` 通过 220 项，覆盖 ONNX Runtime CPU、CoreML 与 WebGPU。构建中的 publint 和 npm tarball 内容检查也通过。这些结果仅代表本机环境。

## 尚需验证

- 等待并检查 GitHub 托管 CI 的各操作系统与 Node.js 版本结果。
- 在匹配设备上补充 ONNX Runtime CUDA 和 DirectML 提供程序实机验证；提供程序是否存在取决于相应 `onnxruntime-node` 构建。
- 增加更多真实图像和边界输入，并逐项比对上游默认参数、错误行为、输出选项及语言路由。
- 验证干净 consumer 安装、首次模型下载、离线运行、CLI 主要命令和正式 npm 发布流程。

上游登记的路由全部有自动化样例，不表示所有模型已在所有设备、输入图像和执行提供程序上获得相同覆盖。达到完整上游流程兼容仍需完成以上验证并处理发现的差异。

## 上游配置兼容

配置生成器只输出 ONNX Runtime 配置。为便于导入上游 v3.10.0 默认 YAML，配置读取器接受其中未修改的其他引擎默认区块；非默认设置会报错。这只兼容默认配置文件格式，不提供 OpenVINO、MNN、Paddle、Torch 或独立 TensorRT 推理。
