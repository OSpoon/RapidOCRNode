import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/cli.ts',
    'src/server.ts',
    'src/structured-output.ts',
  ],
  dts: true,
  exports: true,
  publint: true,
})
