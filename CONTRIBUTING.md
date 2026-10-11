# 贡献指南

## 环境

- Node.js 24 或更高版本
- pnpm 12.8.1

```sh
pnpm install
pnpm run ci
```

`pnpm test` 运行单元测试。`pnpm test:e2e` 下载并校验固定模型和测试资源，然后运行真实模型端到端回归。`pnpm test:models` 对照 RapidOCR v3.10.0 参考运行 42 个 ONNX 阶段路由及 30 个识别路由的整页用例。可选 ONNX Runtime 执行提供程序需要匹配的运行环境。

模型路由、来源 URL 或资源校验值有变更时，请更新真实模型参考或说明为何参考不变，并同步维护 [`MODEL-LICENSES.md`](./MODEL-LICENSES.md)。

## 发布

发布由 `v*` Git 标签触发。发布前确认版本号和变更记录，推送标签后，GitHub Actions 会安装依赖、运行 CI、端到端与 ONNX 真实模型测试、检查 npm 包并发布到 npm。仓库的 npm 包需配置 GitHub Actions Trusted Publisher（仓库 `OSpoon/RapidOCRNode`、工作流 `release.yml`）。

维护者可运行 `pnpm run release` 使用 `bumpp` 更新版本并创建标签。提交信息使用 [Conventional Commits](https://www.conventionalcommits.org/)。
