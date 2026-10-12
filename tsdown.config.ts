import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/cli.ts',
    'src/server.ts',
  ],
  dts: true,
  exports: true,
  publint: true,
})
