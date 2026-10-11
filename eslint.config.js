// @ts-check
import antfu from '@antfu/eslint-config'

export default antfu(
  {
    type: 'lib',
    pnpm: true,
  },
  {
    ignores: [
      '**/__pycache__/**',
      '**/.mypy_cache/**',
      'test/fixtures/rapidocr-v3.10.0-default_models.yaml',
    ],
  },
  {
    files: ['scripts/**/*.mjs'],
    rules: {
      'no-unused-vars': 'error',
      'unused-imports/no-unused-vars': 'off',
    },
  },
)
