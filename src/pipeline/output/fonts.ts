import type { ModelAsset } from '../../../model-assets.mjs'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { downloadVerifiedAsset, getModelDirectory } from '../../models/download.js'

const MODEL_SCOPE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/resources/fonts/'

const FONT_ASSETS: Record<string, ModelAsset> = {
  ch: { file: 'FZYTK.TTF', sha256: '4065a23df6823c8e2b69a0e76d02f02a6470b8774a5e91086609701ad95cc33f', url: `${MODEL_SCOPE}FZYTK.TTF` },
  japan: { file: 'japan.ttc', sha256: '11122490a5e3a862015c8894183750de59abf95c3936d63d5978293d92f23dba', url: `${MODEL_SCOPE}japan.ttc` },
  arabic: { file: 'arabic.ttf', sha256: 'e38162048fca91187a7c3d73a4a1109ca0490d6bac8acf3ff32a57f5fff7e5b6', url: `${MODEL_SCOPE}arabic.ttf` },
  chinese_cht: { file: 'chinese_cht.ttf', sha256: '5ce814960d0cdea1dd647180636babc1cf6a0acf0a9a9019424f4689acedd9ea', url: `${MODEL_SCOPE}chinese_cht.ttf` },
  cyrillic: { file: 'cyrillic.ttf', sha256: '084768d29859a62b735387fb5946dfe61fb3d844031c7c51c1668d8afbe3b802', url: `${MODEL_SCOPE}cyrillic.ttf` },
  devanagari: { file: 'devanagari_Martel-Regular.ttf', sha256: 'b37a04d22ee4700f5018d39678ebb2b9fc64c9d06cd9ec6f9f3fb4c27870a63b', url: `${MODEL_SCOPE}devanagari_Martel-Regular.ttf` },
  korean: { file: 'korean.ttf', sha256: '0897316bdb2e308cea2841c54940f2ef5707856000aa07910c8bff39a47e40bd', url: `${MODEL_SCOPE}korean.ttf` },
  ka: { file: 'kannada.ttf', sha256: 'b337386a8e853ccba53c0c248bd06f025d7667b800ba74c72c66040d67315c6e', url: `${MODEL_SCOPE}kannada.ttf` },
  latin: { file: 'latin.ttf', sha256: '1562fe5cbdaacab4a5880d6404ba05245d12f3a4478fe16021e976bc725ce5d5', url: `${MODEL_SCOPE}latin.ttf` },
  ta: { file: 'tamil.ttf', sha256: 'b771ac413157f6b1f1a52fb8ff1b56057f4b492fcce385ddd32ca12eee0c73b0', url: `${MODEL_SCOPE}tamil.ttf` },
  te: { file: 'telugu.ttf', sha256: '7f82ab141b77d263f9ea9b31b47faf50c11310f42fce6d9dffeaaa334909bbf9', url: `${MODEL_SCOPE}telugu.ttf` },
  th: { file: 'th.ttf', sha256: 'b8150084e25734e6f31696c57ff009f5564efa09d295848b717d9e2328c0311d', url: `${MODEL_SCOPE}th.ttf` },
  el: { file: 'el.ttf', sha256: '892e6c81071fc85923f8d0ff925edab6049dd4495a2fdedffddc04753bf348b3', url: `${MODEL_SCOPE}el.ttf` },
}

const FONT_ALIASES: Record<string, string> = {
  'ch_doc': 'ch',
  'en': 'ch',
  'eslav': 'cyrillic',
  'ja': 'japan',
  'jp': 'japan',
  'zh': 'ch',
  'zh_cn': 'ch',
  'zh-cn': 'ch',
  'zh_tw': 'chinese_cht',
  'zh-tw': 'chinese_cht',
}

export async function loadVisualizationFont(options: {
  autoDownload?: boolean
  fontPath?: string
  langType?: string
  modelDirectory?: string
}): Promise<Uint8Array> {
  if (options.fontPath)
    return readFile(options.fontPath)

  const language = (options.langType ?? 'ch').trim().toLowerCase()
  const font = FONT_ASSETS[FONT_ALIASES[language] ?? language] ?? FONT_ASSETS.ch!
  const modelDirectory = getModelDirectory(options.modelDirectory)
  if (options.autoDownload !== false) {
    const path = await downloadVerifiedAsset(font, modelDirectory)
    return readFile(path)
  }

  const cachedPath = join(modelDirectory, font.file)
  const contents = await readFile(cachedPath)
  const actualHash = createHash('sha256').update(contents).digest('hex')
  if (actualHash !== font.sha256)
    throw new Error(`Cached visualization font failed SHA-256 verification: ${cachedPath}`)
  return contents
}
