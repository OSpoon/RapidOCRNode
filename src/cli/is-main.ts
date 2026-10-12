import { realpathSync, statSync } from 'node:fs'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

/** Detects direct execution when Node was launched through a package-manager symlink. */
export function isMainModule(moduleUrl: string, entryPath: string | undefined): boolean {
  if (!entryPath)
    return false

  try {
    const modulePath = realpathSync(fileURLToPath(moduleUrl))
    const entryRealPath = realpathSync(entryPath)
    if (normalizePath(modulePath) === normalizePath(entryRealPath))
      return true

    const moduleStats = statSync(modulePath)
    const entryStats = statSync(entryRealPath)
    return moduleStats.dev === entryStats.dev && moduleStats.ino !== 0 && moduleStats.ino === entryStats.ino
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
