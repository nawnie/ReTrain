/**
 * ReTrain desktop shell.
 *
 * Responsibilities, in order of importance:
 *   1. Start and supervise the local FastAPI backend, and stop only the one it
 *      started.
 *   2. Serve the built frontend over a private protocol.
 *   3. Forward the frontend's API calls to that backend, so the renderer uses
 *      exactly the same relative /api/retrain paths in the desktop app as it
 *      does behind the Vite dev proxy. The previous shell instead had the
 *      client sniff `window.location.protocol` and swap in an absolute URL;
 *      that puts deployment knowledge in the data layer and gives dev and
 *      production two different code paths through the thing most likely to
 *      break.
 */

const { app, BrowserWindow, ipcMain, nativeTheme, net, protocol, shell } = require('electron')
const { spawn, spawnSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')
const { pathToFileURL } = require('node:url')

// One host and port for both ends. The shell spawns the backend and forwards to
// it, so it owns both sides of this and they cannot drift apart.
const API_HOST = process.env.RETRAIN_API_HOST ?? '127.0.0.1'
const API_PORT = Number(process.env.RETRAIN_API_PORT ?? 8787)
const API_ORIGIN = `http://${API_HOST}:${API_PORT}`
const API_PREFIX = '/api/retrain'

const projectRoot = path.resolve(__dirname, '..', '..', '..')
const distRoot = path.resolve(__dirname, '..', 'dist')
const iconPath = path.join(__dirname, 'retrain.ico')

let mainWindow = null
let backendProcess = null

// Caption-button colours, sent from the renderer when the theme changes. These
// must track the app header or the native buttons sit on the wrong ground.
const TITLE_BAR_HEIGHT = 48
let titleBarColors = { color: '#fbfaf9', symbolColor: '#1c1a17' }

/* ---------------------------------------------------------------------------
   The private protocol has to be registered as privileged before the app is
   ready, or fetch() from the renderer is not allowed to use it.
   --------------------------------------------------------------------------- */

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'retrain',
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  },
])

/* ---------------------------------------------------------------------------
   Backend supervision
   --------------------------------------------------------------------------- */

/**
 * Is something already answering on the API port?
 *
 * A healthy answer here is adopted rather than replaced. That covers two real
 * cases: an operator running uvicorn themselves in a terminal, and a backend
 * orphaned by a previous shell that was force-killed rather than closed.
 * `before-quit` cannot run when a process is killed outright, so that orphan
 * keeps the port; adopting it is correct, because it serves the same app from
 * the same source. Only a shell that started its own backend ever stops one.
 */
async function apiReady() {
  try {
    const response = await fetch(`${API_ORIGIN}${API_PREFIX}/bootstrap`, {
      signal: AbortSignal.timeout(1500),
    })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Start uvicorn, unless the operator already has it running.
 *
 * Launched through the project's pythonw interpreter to keep the API headless.
 *
 * Returns a reason string on failure rather than throwing. A backend that will
 * not start is not a reason to deny the operator the window: the UI degrades to
 * an explanatory offline state and recovers on its own once the API answers.
 */
async function startBackend() {
  if (await apiReady()) return null

  const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
  const pythonwExe = path.join(projectRoot, '.venv', 'Scripts', 'pythonw.exe')

  for (const required of [pythonExe, pythonwExe]) {
    if (!fs.existsSync(required)) return `Required runtime is missing: ${required}`
  }

  const logRoot = path.join(projectRoot, 'training', 'gui_runs')
  const logPath = path.join(logRoot, 'desktop-api.log')
  try {
    fs.mkdirSync(logRoot, { recursive: true })
  } catch (error) {
    return `Could not create the log directory: ${error.message}`
  }

  // Preflight the import in a normal, short-lived python.exe.
  //
  // pythonw has no console standard handles. Log the API directly to a file;
  // use python.exe only for this bounded import check that captures errors.
  //
  // An import-time failure (a missing dependency, a syntax error in the API) is
  // exactly the case where a log matters most, and it is a traceback on stderr
  // rather than anything the logging config would catch. So it is checked here
  // first, in a process whose output can actually be read. This is a one-shot
  // check, not a standing helper, so a plain interpreter is appropriate;
  // windowsHide keeps it from flashing a console.
  const preflight = spawnSync(pythonExe, ['-c', 'import gui.api.app'], {
    cwd: projectRoot,
    windowsHide: true,
    encoding: 'utf8',
    timeout: 60000,
  })
  if (preflight.status !== 0) {
    const detail = (preflight.stderr || preflight.stdout || 'no output').trim()
    try {
      fs.appendFileSync(logPath, `\n[preflight ${new Date().toISOString()}]\n${detail}\n`)
    } catch {
      // Reporting the failure matters more than recording it.
    }
    return `The ReTrain API failed to import. See ${logPath}`
  }

  // uvicorn writes its own log file rather than being piped, for the same
  // reason: nothing survives the pythonw hop. This config is generated rather
  // than shipped because the path is only known at runtime.
  const logConfigPath = path.join(logRoot, 'desktop-api-logging.json')
  try {
    fs.writeFileSync(
      logConfigPath,
      JSON.stringify(
        {
          version: 1,
          disable_existing_loggers: false,
          formatters: { plain: { format: '%(asctime)s %(levelname)s %(name)s %(message)s' } },
          handlers: {
            file: {
              class: 'logging.handlers.RotatingFileHandler',
              formatter: 'plain',
              filename: logPath,
              maxBytes: 1048576,
              backupCount: 2,
              encoding: 'utf-8',
            },
          },
          loggers: {
            uvicorn: { handlers: ['file'], level: 'INFO', propagate: false },
            'uvicorn.error': { handlers: ['file'], level: 'INFO', propagate: false },
            'uvicorn.access': { handlers: ['file'], level: 'INFO', propagate: false },
          },
        },
        null,
        2,
      ),
    )
  } catch (error) {
    return `Could not write the backend log config: ${error.message}`
  }

  backendProcess = spawn(
    pythonwExe,
    [
      '-m',
      'uvicorn',
      'gui.api.app:app',
      '--host',
      API_HOST,
      '--port',
      String(API_PORT),
      '--log-config',
      logConfigPath,
    ],
    { cwd: projectRoot, windowsHide: true, stdio: ['ignore', 'ignore', 'ignore'] },
  )

  // Poll rather than sleep a fixed amount: a warm start answers in well under a
  // second, a cold one can take several.
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (await apiReady()) return null
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  return `The ReTrain API did not answer at ${API_ORIGIN} within 15 seconds.`
}

/** Stop only a backend this shell started. An operator's own uvicorn is left alone. */
function stopOwnedBackend() {
  if (!backendProcess || backendProcess.exitCode !== null) return
  // Stop the owned API and any children it created.
  spawnSync('taskkill.exe', ['/pid', String(backendProcess.pid), '/t', '/f'], {
    windowsHide: true,
    stdio: 'ignore',
  })
  backendProcess = null
}

/* ---------------------------------------------------------------------------
   The retrain:// protocol: static files, plus an API passthrough
   --------------------------------------------------------------------------- */

function registerAppProtocol() {
  protocol.handle('retrain', async (request) => {
    const requestUrl = new URL(request.url)

    // API passthrough. This is what lets the renderer call the same relative
    // paths it uses in dev.
    if (requestUrl.pathname.startsWith(API_PREFIX)) {
      const target = `${API_ORIGIN}${requestUrl.pathname}${requestUrl.search}`
      try {
        return await net.fetch(target, {
          method: request.method,
          headers: request.headers,
          body: request.body,
          // Required by Node's fetch whenever a body is streamed.
          duplex: 'half',
        })
      } catch (error) {
        // Shaped like the backend's own errors so the client's normal failure
        // path handles it, instead of a protocol-level crash.
        return new Response(
          JSON.stringify({ status: 'error', message: `Local API unreachable: ${error.message}` }),
          { status: 502, headers: { 'Content-Type': 'application/json' } },
        )
      }
    }

    // Static asset. Anything that is not a real file falls back to index.html so
    // a deep link keeps working.
    const requested = decodeURIComponent(
      requestUrl.pathname === '/' ? '/index.html' : requestUrl.pathname,
    )
    let filePath = path.resolve(distRoot, `.${requested}`)

    // Never serve outside dist, whatever the URL claims.
    if (filePath !== distRoot && !filePath.startsWith(`${distRoot}${path.sep}`)) {
      return new Response('Blocked path', { status: 403 })
    }
    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      filePath = path.join(distRoot, 'index.html')
    }
    return net.fetch(pathToFileURL(filePath).toString())
  })
}

/* ---------------------------------------------------------------------------
   Window
   --------------------------------------------------------------------------- */

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 940,
    // Below this the layout is still correct, but the projection column stacks
    // and the console stops being usable as a side-by-side workbench.
    minWidth: 900,
    minHeight: 600,
    show: false,
    icon: iconPath,
    backgroundColor: titleBarColors.color,
    // The native caption buttons, overlaid on the app's own header. This keeps
    // the header design while leaving minimise, maximise, snap layouts, and
    // their keyboard and accessibility behaviour to the OS -- all of which a
    // hand-rolled frameless title bar has to reimplement and usually does not.
    titleBarStyle: 'hidden',
    titleBarOverlay: { ...titleBarColors, height: TITLE_BAR_HEIGHT },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())
  mainWindow.on('closed', () => {
    mainWindow = null
  })

  // Anything that wants a new window is an external link; hand it to the real
  // browser rather than opening an unsandboxed window inside the app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://') || url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })

  // The renderer only ever loads the local app. A navigation anywhere else is
  // either a mistake or an attack.
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('retrain://')) event.preventDefault()
  })

  mainWindow.loadURL('retrain://app/index.html')
}

/* ---------------------------------------------------------------------------
   Renderer bridge
   --------------------------------------------------------------------------- */

// The renderer reports its resolved theme colours so the native caption buttons
// match the header in both themes.
ipcMain.on('retrain:title-bar-theme', (_event, colors) => {
  if (!colors || typeof colors.color !== 'string' || typeof colors.symbolColor !== 'string') return
  titleBarColors = { color: colors.color, symbolColor: colors.symbolColor }
  try {
    mainWindow?.setTitleBarOverlay({ ...titleBarColors, height: TITLE_BAR_HEIGHT })
  } catch {
    // setTitleBarOverlay is Windows-only; elsewhere the native frame is fine.
  }
  nativeTheme.themeSource = colors.dark ? 'dark' : 'light'
})

/* ---------------------------------------------------------------------------
   Lifecycle
   --------------------------------------------------------------------------- */

// A second launch should focus the existing window, not start a second backend
// on a port the first one already holds.
if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  })

  app.whenReady().then(async () => {
    app.setAppUserModelId('AiEmbeddedSystems.ReTrain')
    registerAppProtocol()

    // The window is created regardless of the backend's fate. The UI has a
    // designed offline state that names the problem and recovers on its own.
    const failure = await startBackend()
    createWindow()
    if (failure) {
      console.error(`[retrain] ${failure}`)
    }
  })

  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', stopOwnedBackend)
  // Covers the paths that skip before-quit, such as a signal or a crash.
  process.on('exit', stopOwnedBackend)
}
