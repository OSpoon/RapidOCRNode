// @ts-check
import antfu from '@antfu/eslint-config'

export default antfu(
  {
    type: 'lib',
    pnpm: true,
  },
  {
    files: ['scripts/**/*.mjs'],
    rules: {
      'no-unused-vars': 'error',
      'unused-imports/no-unused-vars': 'off',
    },
  },
)
