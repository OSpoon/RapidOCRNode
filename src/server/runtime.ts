import type { Server } from 'node:net'
import type { RapidOCROptions } from '../shared/types.js'
import { Buffer } from 'node:buffer'
import { spawn } from 'node:child_process'
import { randomBytes, timingSafeEqual } from 'node:crypto'
import { chmod, mkdir, open, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { connect, createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import process from 'node:process'
import { RapidOCR } from '../pipeline/rapidocr.js'
import { createRapidOCRServer } from './api.js'

const STATE_FILE = 'rapidocr-node-server.json'
const LOCK_FILE = 'rapidocr-node-server.lock'
const STOP_TIMEOUT_MS = 30_000
const START_TIMEOUT_MS = 300_000

interface RuntimeState {
  controlPort: number
  cwd: string
  host: string
  httpPort: number
  logFile: string
  pid: number
  restartArgs: string[]
  startedAt: string
  token: string
}

export interface RuntimeStatus {
  baseUrl: string
  host: string
  httpPort: number
  logFile: string
  pid: number
  startedAt: string
  status: 'running'
}

export interface ServerRuntimeOptions {
  host: string
  port: number
  maxFileSize: number
  maxConcurrency: number
  runtimeDir: string
  rapidOptions: RapidOCROptions
  restartArgs: string[]
}

export function defaultRuntimeDirectory(): string {
  return resolve(process.env.XDG_RUNTIME_DIR || join(tmpdir(), 'rapidocr-node'))
}

export async function startServerProcess(
  entryPath: string,
  runArgs: string[],
  restartArgs: string[],
  runtimeDirectory: string,
  workingDirectory = process.cwd(),
): Promise<RuntimeStatus> {
  const runtimeDir = resolve(runtimeDirectory)
  await ensureRuntimeDirectory(runtimeDir)
  const current = await queryRuntimeStatus(runtimeDir).catch(() => undefined)
  if (current)
    return current

  const logFile = join(runtimeDir, 'server.log')
  const logFd = await open(logFile, 'a', 0o600)
  let childError: Error | undefined
  let childExit: { code: number | null, signal: NodeJS.Signals | null } | undefined
  try {
    const child = spawn(process.execPath, [resolve(entryPath), ...runArgs], {
      cwd: workingDirectory,
      detached: true,
      env: {
        ...process.env,
        RAPIDOCR_SERVE_RESTART_ARGS: JSON.stringify(restartArgs),
      },
      stdio: ['ignore', logFd.fd, logFd.fd],
    })
    child.once('error', (error) => {
      childError = error
    })
    child.once('exit', (code, signal) => {
      childExit = { code, signal }
    })
    child.unref()

    const deadline = Date.now() + START_TIMEOUT_MS
    let lastError: unknown
    while (Date.now() < deadline) {
      try {
        return await queryRuntimeStatus(runtimeDir)
      }
      catch (error) {
        lastError = error
      }
      if (childError || childExit) {
        const logTail = await readLogTail(logFile)
        const reason = childError?.message ?? `process exited with ${childExit?.signal ?? `code ${childExit?.code}`}`
        throw new Error(`The OCR server failed to start: ${reason}.${logTail ? `\n${logTail}` : ''}`, { cause: childError ?? lastError })
      }
      await delay(250)
    }
    child.kill('SIGTERM')
    const logTail = await readLogTail(logFile)
    throw new Error(`The OCR server did not become ready within ${START_TIMEOUT_MS / 1000} seconds.${logTail ? `\n${logTail}` : ''}`, { cause: lastError })
  }
  finally {
    await logFd.close()
  }
}

export async function runServer(options: ServerRuntimeOptions): Promise<void> {
  const runtimeDir = resolve(options.runtimeDir)
  await ensureRuntimeDirectory(runtimeDir)
  await clearDeadRuntimeState(runtimeDir)
  const lockPath = join(runtimeDir, LOCK_FILE)
  let lock
  try {
    lock = await open(lockPath, 'wx', 0o600)
    await lock.writeFile(`${process.pid}\n`)
  }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'EEXIST')
      throw new Error(`An OCR server is already running or starting in ${runtimeDir}. Use "rapidocr-node serve status" to inspect it.`)
    throw error
  }

  let controlServer: Server | undefined
  let ocr: RapidOCR | undefined
  let app: Awaited<ReturnType<typeof createRapidOCRServer>> | undefined
  let state: RuntimeState | undefined
  let finished = false
  let resolveShutdown: (() => void) | undefined
  let shutdownPromise: Promise<void> | undefined

  function shutdown(): Promise<void> {
    if (shutdownPromise)
      return shutdownPromise
    shutdownPromise = (async () => {
      if (finished)
        return
      finished = true
      process.removeListener('SIGINT', signalHandler)
      process.removeListener('SIGTERM', signalHandler)
      const errors: unknown[] = []
      if (controlServer)
        await closeNetServer(controlServer).catch(error => errors.push(error))
      if (app)
        await app.close().catch(error => errors.push(error))
      if (ocr)
        await ocr.close().catch(error => errors.push(error))
      await rm(join(runtimeDir, STATE_FILE), { force: true }).catch(error => errors.push(error))
      await rm(lockPath, { force: true }).catch(error => errors.push(error))
      resolveShutdown?.()
      if (errors.length > 0)
        throw new AggregateError(errors, 'One or more OCR server shutdown steps failed.')
    })()
    return shutdownPromise
  }
  function signalHandler(): void {
    void shutdown().catch((error: unknown) => {
      process.stderr.write(`RapidOCR API shutdown failed: ${error instanceof Error ? error.message : String(error)}\n`)
    })
  }

  try {
    ocr = await RapidOCR.create({
      ...options.rapidOptions,
      useDetection: true,
      useClassifier: true,
      useRecognition: true,
    })
    await ocr.warmup()
    app = await createRapidOCRServer({
      maxConcurrency: options.maxConcurrency,
      maxFileSize: options.maxFileSize,
      ocr,
    })
    const address = await app.listen({ host: options.host, port: options.port })
    const httpPort = new URL(address).port ? Number(new URL(address).port) : options.port
    const control = await createControlServer(shutdown)
    controlServer = control.server
    state = {
      controlPort: control.port,
      cwd: process.cwd(),
      host: options.host,
      httpPort,
      logFile: join(runtimeDir, 'server.log'),
      pid: process.pid,
      restartArgs: options.restartArgs,
      startedAt: new Date().toISOString(),
      token: control.token,
    }
    await writeRuntimeState(runtimeDir, state)
    process.stdout.write(`RapidOCR API listening at ${address}\n`)
    process.on('SIGINT', signalHandler)
    process.on('SIGTERM', signalHandler)
    await new Promise<void>((resolvePromise) => {
      resolveShutdown = resolvePromise
    })
  }
  catch (error) {
    if (!finished) {
      finished = true
      if (controlServer)
        await closeNetServer(controlServer).catch(() => {})
      if (app)
        await app.close().catch(() => {})
      if (ocr)
        await ocr.close().catch(() => {})
      await rm(join(runtimeDir, STATE_FILE), { force: true })
      await rm(lockPath, { force: true })
    }
    throw error
  }
  finally {
    await lock?.close()
  }
  await shutdownPromise
  if (state)
    process.stdout.write('RapidOCR API stopped.\n')
}

export async function queryRuntimeStatus(runtimeDirectory: string): Promise<RuntimeStatus> {
  const runtimeDir = resolve(runtimeDirectory)
  const state = await readRuntimeState(runtimeDir)
  const response = await sendControlRequest(state, 'status')
  if (!response.ok)
    throw new Error(response.error ?? 'The OCR server control request failed.')
  return toStatus(state)
}

export async function stopServer(runtimeDirectory: string, timeoutMs = STOP_TIMEOUT_MS): Promise<void> {
  const runtimeDir = resolve(runtimeDirectory)
  const state = await readRuntimeState(runtimeDir)
  const response = await sendControlRequest(state, 'stop')
  if (!response.ok)
    throw new Error(response.error ?? 'The OCR server refused to stop.')
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      await queryRuntimeStatus(runtimeDir)
    }
    catch {
      let current: RuntimeState
      try {
        current = await readRuntimeState(runtimeDir)
      }
      catch (error) {
        if (error instanceof Error && error.message.startsWith('No OCR server is running'))
          return
        throw error
      }
      if (processIsAlive(current.pid)) {
        await delay(100)
        continue
      }
      await rm(join(runtimeDir, STATE_FILE), { force: true })
      await rm(join(runtimeDir, LOCK_FILE), { force: true })
      return
    }
    await delay(100)
  }
  throw new Error(`Timed out waiting ${timeoutMs}ms for the OCR server to stop.`)
}

export async function restartServer(runtimeDirectory: string): Promise<RuntimeStatus> {
  const runtimeDir = resolve(runtimeDirectory)
  const state = await readRuntimeState(runtimeDir)
  const args = withRuntimeDirectory(state.restartArgs, runtimeDir)
  await stopServer(runtimeDir)
  const cliEntry = process.argv[1]
  if (!cliEntry)
    throw new Error('Cannot restart the OCR server because the CLI entrypoint is unavailable.')
  const runArgs = [...args]
  const serveIndex = runArgs.indexOf('serve')
  if (serveIndex < 0 || runArgs[serveIndex + 1] !== 'start')
    throw new Error('Saved server restart arguments are invalid.')
  runArgs[serveIndex + 1] = 'run'
  return startServerProcess(cliEntry, runArgs, args, runtimeDir, state.cwd)
}

async function createControlServer(shutdown: () => Promise<void>): Promise<{ port: number, server: Server, token: string }> {
  const token = randomBytes(32).toString('hex')
  const server = createServer((socket) => {
    let data = ''
    socket.setEncoding('utf8')
    socket.setTimeout(5_000, () => socket.destroy())
    socket.on('data', (chunk) => {
      data += chunk
      if (data.length > 8192) {
        socket.end(`${JSON.stringify({ ok: false, error: 'Request too large.' })}\n`)
        return
      }
      const newline = data.indexOf('\n')
      if (newline < 0)
        return
      let request: { action?: string, token?: string }
      try {
        request = JSON.parse(data.slice(0, newline)) as { action?: string, token?: string }
      }
      catch {
        socket.end(`${JSON.stringify({ ok: false, error: 'Invalid control request.' })}\n`)
        return
      }
      if (!safeTokenEqual(token, request.token)) {
        socket.end(`${JSON.stringify({ ok: false, error: 'Unauthorized.' })}\n`)
        return
      }
      if (request.action === 'status') {
        socket.end(`${JSON.stringify({ ok: true })}\n`)
        return
      }
      if (request.action === 'stop') {
        socket.end(`${JSON.stringify({ ok: true })}\n`)
        setImmediate(() => {
          void shutdown().catch((error: unknown) => {
            process.stderr.write(`RapidOCR API shutdown failed: ${error instanceof Error ? error.message : String(error)}\n`)
          })
        })
        return
      }
      socket.end(`${JSON.stringify({ ok: false, error: 'Unknown control action.' })}\n`)
    })
  })
  await new Promise<void>((resolvePromise, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0 }, resolvePromise)
  })
  const address = server.address()
  if (!address || typeof address === 'string')
    throw new Error('Unable to open the local OCR server control channel.')
  return { port: address.port, server, token }
}

async function sendControlRequest(state: RuntimeState, action: 'status' | 'stop'): Promise<{ ok: boolean, error?: string }> {
  return await new Promise((resolvePromise, reject) => {
    const socket = connect({ host: '127.0.0.1', port: state.controlPort })
    let data = ''
    const timeout = setTimeout(() => {
      socket.destroy(new Error('Timed out connecting to the OCR server control channel.'))
    }, 1500)
    socket.setEncoding('utf8')
    socket.on('connect', () => socket.write(`${JSON.stringify({ action, token: state.token })}\n`))
    socket.on('data', chunk => data += chunk)
    socket.on('end', () => {
      clearTimeout(timeout)
      try {
        resolvePromise(JSON.parse(data.trim()) as { ok: boolean, error?: string })
      }
      catch (error) {
        reject(error)
      }
    })
    socket.on('error', (error) => {
      clearTimeout(timeout)
      reject(error)
    })
  })
}

async function writeRuntimeState(runtimeDir: string, state: RuntimeState): Promise<void> {
  const path = join(runtimeDir, STATE_FILE)
  const tempPath = `${path}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`
  await writeFile(tempPath, `${JSON.stringify(state, null, 2)}\n`, { flag: 'wx', mode: 0o600 })
  await chmod(tempPath, 0o600)
  await rename(tempPath, path)
}

async function readRuntimeState(runtimeDir: string): Promise<RuntimeState> {
  const path = join(runtimeDir, STATE_FILE)
  let parsed: unknown
  try {
    parsed = JSON.parse(await readFile(path, 'utf8')) as unknown
  }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT')
      throw new Error(`No OCR server is running in ${runtimeDir}.`)
    throw new Error(`Cannot read OCR server state from ${path}.`, { cause: error })
  }
  if (parsed === null || typeof parsed !== 'object')
    throw new Error(`OCR server state in ${path} is invalid.`)
  const state = parsed as RuntimeState
  if (!Number.isInteger(state.pid)
    || !Number.isInteger(state.controlPort)
    || typeof state.host !== 'string'
    || !Number.isInteger(state.httpPort)
    || typeof state.cwd !== 'string'
    || typeof state.logFile !== 'string'
    || typeof state.startedAt !== 'string'
    || typeof state.token !== 'string'
    || !Array.isArray(state.restartArgs)
    || !state.restartArgs.every(argument => typeof argument === 'string')) {
    throw new Error(`OCR server state in ${path} is invalid.`)
  }
  return state
}

async function clearDeadRuntimeState(runtimeDir: string): Promise<void> {
  try {
    const state = await readRuntimeState(runtimeDir)
    if (processIsAlive(state.pid))
      throw new Error(`An OCR server process (PID ${state.pid}) is already running or starting in ${runtimeDir}.`)
  }
  catch (error) {
    if (error instanceof Error && error.message.includes('already running or starting'))
      throw error
  }
  const lockPath = join(runtimeDir, LOCK_FILE)
  try {
    const pid = Number((await readFile(lockPath, 'utf8')).trim())
    if (Number.isInteger(pid) && pid > 0 && processIsAlive(pid))
      throw new Error(`An OCR server process (PID ${pid}) is already starting in ${runtimeDir}.`)
  }
  catch (error) {
    if (error instanceof Error && error.message.includes('already starting'))
      throw error
  }
  await rm(join(runtimeDir, STATE_FILE), { force: true })
  await rm(lockPath, { force: true })
}

async function ensureRuntimeDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true, mode: 0o700 })
}

function toStatus(state: RuntimeState): RuntimeStatus {
  const host = state.host === '0.0.0.0' || state.host === '::' ? '127.0.0.1' : state.host
  const urlHost = host.includes(':') && !host.startsWith('[') ? `[${host}]` : host
  return {
    baseUrl: `http://${urlHost}:${state.httpPort}`,
    host: state.host,
    httpPort: state.httpPort,
    logFile: state.logFile,
    pid: state.pid,
    startedAt: state.startedAt,
    status: 'running',
  }
}

function withRuntimeDirectory(args: string[], runtimeDir: string): string[] {
  const result = [...args]
  const optionIndex = result.indexOf('--runtime-dir')
  if (optionIndex >= 0 && result[optionIndex + 1] !== undefined)
    result[optionIndex + 1] = runtimeDir
  else
    result.push('--runtime-dir', runtimeDir)
  return result
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  }
  catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM'
  }
}

function safeTokenEqual(expected: string, actual: string | undefined): boolean {
  if (actual === undefined)
    return false
  const left = Buffer.from(expected)
  const right = Buffer.from(actual)
  return left.length === right.length && timingSafeEqual(left, right)
}

async function readLogTail(path: string): Promise<string> {
  try {
    const content = await readFile(path, 'utf8')
    return content.trim().split('\n').slice(-20).join('\n')
  }
  catch {
    return ''
  }
}

function closeNetServer(server: Server): Promise<void> {
  return new Promise((resolvePromise, reject) => {
    server.close(error => error ? reject(error) : resolvePromise())
  })
}

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolvePromise => setTimeout(resolvePromise, milliseconds))
}
