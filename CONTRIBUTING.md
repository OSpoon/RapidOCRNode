# Contributing

## Local setup

Use Node.js 22.18 or newer and pnpm 12. Install dependencies with:

```sh
pnpm install
```

Before opening a pull request, run the project gates:

```sh
pnpm lint && pnpm knip && pnpm test && pnpm typecheck && pnpm build
```

`pnpm test` runs unit tests without downloading models. `pnpm test:e2e` downloads and verifies the pinned models and fixtures before running the real-model regression. The tests compare two upstream pages and a downscaled page, and exercise direct line recognition, direction correction, image scaling, blank-page handling, and session shutdown.

## Release

The first npm release must be published once by a maintainer with npm access (`pnpm publish`) so the package exists. Then configure npm Trusted Publishing for `OSpoon/RapidOCRNode` with the GitHub Actions workflow file `.github/workflows/release.yml` and no environment. The workflow uses GitHub Actions OIDC, so no npm token secret is needed.

For later releases, start from a clean working tree and run:

```sh
pnpm run release
```

`bumpp` prompts for the next version, updates the package version, creates a commit and `v*` tag, and pushes them. Pushing the tag starts the release workflow. It runs the project gates and real-model E2E checks before publishing to npm, then creates the GitHub release and changelog.

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. Record the immutable model source, checksum, and license details when changing model assets. Preserve the RapidOCR attribution in `NOTICE.md` when modifying adapted code.
