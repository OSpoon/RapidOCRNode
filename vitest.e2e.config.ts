import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    hookTimeout: 120_000,
    include: ['test/e2e/**/*.test.ts'],
    testTimeout: 60_000,
  },
})
