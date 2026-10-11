import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    hookTimeout: 180_000,
    include: ['test/model-matrix/**/*.test.ts'],
    testTimeout: 180_000,
  },
})
