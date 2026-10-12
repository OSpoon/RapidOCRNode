import { realpathSync } from 'node:fs'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

/** Detects direct execution when Node was launched through a package-manager symlink. */
export function isMainModule(moduleUrl: string, entryPath: string | undefined): boolean {
  if (!entryPath)
    return false

  try {
    return normalizePath(realpathSync(fileURLToPath(moduleUrl))) === normalizePath(realpathSync(entryPath))
  }
  catch {
    return false
  }
}

function normalizePath(path: string): string {
  if (process.platform !== 'win32')
    return path

  return path
    .replace(/^\\\\\?\\UNC\\/iu, '\\\\')
    .replace(/^\\\\\?\\/u, '')
    .replaceAll('/', '\\')
    .toLowerCase()
}
