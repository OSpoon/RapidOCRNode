import { realpathSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

/** Detects direct execution when Node was launched through a package-manager symlink. */
export function isMainModule(moduleUrl: string, entryPath: string | undefined): boolean {
  if (!entryPath)
    return false

  try {
    return realpathSync(fileURLToPath(moduleUrl)) === realpathSync(entryPath)
  }
  catch {
    return false
  }
}
