/**
 * The application shell.
 *
 * Owns all state and every request; the panes below it are presentational.
 *
 * Two layout decisions are load-bearing and deliberate:
 *
 *  1. The shell is a three-row grid — header, work area, launch bar — and only
 *     the work area flexes. Everything that spends GPU time lives in the
 *     launch bar, which is a shell row rather than page content, so it cannot
 *     scroll out of reach at any window size.
 *
 *  2. Each pane is its own scroll container. The page itself never scrolls.
 *     That keeps the header and launch bar fixed without any of the content
 *     becoming unreachable, which is only true because every pane is an
 *     explicit `overflow-y: auto`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as api from './api'
import { readableError } from './api'
import type {
  Bootstrap,
  CheckpointInventory,
  CompareSummary,
  JobReceipt,
  JobStatus,
  LayerMap,
  LocalModel,
  RuntimeStatus,
  TensorBoardStatus,
  TrainingPlan,
  TrainingSettings,
} from './types'
import { Chip, Notice, toneForState, type Tone } from './ui'
import { Configure } from './panels/Configure'
import { Layers } from './panels/Layers'
import { Live } from './panels/Live'
import { Projection } from './panels/Projection'
import { Runs } from './panels/Runs'
import { Runtime } from './panels/Runtime'

type PaneId = 'configure' | 'live' | 'layers' | 'runs' | 'runtime'

const PANES: { id: PaneId; label: string }[] = [
  { id: 'configure', label: 'Configure' },
  { id: 'live', label: 'Live' },
  { id: 'layers', label: 'Layers' },
  { id: 'runs', label: 'Runs' },
  { id: 'runtime', label: 'Runtime' },
]

/**
 * The open pane is kept in the URL hash rather than only in component state.
 *
 * Three things depend on it: the browser back button behaves, a pane can be
 * linked to directly, and — the reason it is not optional — an automated
 * review can actually visit every pane. A single-page app whose views exist
 * only in client state renders one view per URL, so any sweep that loads the
 * app once has only ever checked the first pane, and the rest ship unverified.
 */
function paneFromHash(): PaneId {
  const requested = window.location.hash.replace('#', '').toLowerCase()
  return PANES.find((item) => item.id === requested)?.id ?? 'configure'
}

/** How often the shell re-reads live job and machine state, in milliseconds. */
const POLL_INTERVAL_MS = 5000
/** Quiet period after a control changes before the plan is re-requested. */
const PLAN_DEBOUNCE_MS = 350

type ThemeChoice = 'system' | 'light' | 'dark'

/** A transient message with a tone. Cleared by the operator or by the next action. */
interface Message {
  tone: Tone
  text: string
}

export default function App() {
  const [pane, setPane] = useState<PaneId>(paneFromHash)
  const [theme, setTheme] = useState<ThemeChoice>('system')

  const [boot, setBoot] = useState<Bootstrap | null>(null)
  const [settings, setSettings] = useState<TrainingSettings | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [plan, setPlan] = useState<TrainingPlan | null>(null)
  const [planPending, setPlanPending] = useState(false)
  const [jobs, setJobs] = useState<JobStatus>({ status: 'idle', active_count: 0, jobs: [] })
  const [receipt, setReceipt] = useState<JobReceipt | null>(null)
  const [inventory, setInventory] = useState<CheckpointInventory | null>(null)
  const [compare, setCompare] = useState<CompareSummary | null>(null)
  const [tensorboard, setTensorboard] = useState<TensorBoardStatus | null>(null)

  // Layer map. `folder` is resolved once from the configured model, then owned
  // by the pane so the operator can inspect a checkpoint other than the one the
  // run is currently pointed at.
  const [localModels, setLocalModels] = useState<LocalModel[]>([])
  const [layerFolder, setLayerFolder] = useState('')
  const [layerMap, setLayerMap] = useState<LayerMap | null>(null)
  const [layerSample, setLayerSample] = useState(true)
  const [layerLoading, setLayerLoading] = useState(false)
  const [layerNotice, setLayerNotice] = useState<string | null>(null)

  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState('')
  const [message, setMessage] = useState<Message | null>(null)
  /**
   * Connection trouble is kept separate from action results on purpose. They
   * have different lifetimes: a failed poll repeats every few seconds, while an
   * action result is something the operator asked for and should stay put. Held
   * in one variable, the poll would overwrite the answer within five seconds.
   */
  const [offline, setOffline] = useState<string | null>(null)

  /* -- Pane routing --------------------------------------------------------
     Two directions, kept in sync: the tabs write the hash, and the browser's
     own navigation (back button, a pasted link) writes the pane. The listener
     is removed on unmount so the handler cannot outlive the component.
     ----------------------------------------------------------------------- */

  useEffect(() => {
    const onHashChange = () => setPane(paneFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  /* -- Theme -------------------------------------------------------------- */

  // The choice is written to <html>, where theme.css reads it. "system" removes
  // the attribute entirely so the prefers-color-scheme media query applies.
  //
  // The desktop shell then needs the resolved colours, because the native
  // caption buttons are painted by the OS and would otherwise keep the light
  // palette on a dark header. They are read back from the cascade rather than
  // duplicated here, so the token layer stays the only source of colour.
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)

    const shell = window.retrainDesktop
    if (!shell) return

    const styles = getComputedStyle(root)
    shell.setTitleBarTheme({
      color: styles.getPropertyValue('--surface').trim(),
      symbolColor: styles.getPropertyValue('--ink').trim(),
      dark: styles.colorScheme.includes('dark'),
    })
  }, [theme])

  // "Match system" has to follow the system actually changing, not only the
  // moment the choice was made.
  useEffect(() => {
    if (theme !== 'system') return undefined
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      const shell = window.retrainDesktop
      if (!shell) return
      const styles = getComputedStyle(document.documentElement)
      shell.setTitleBarTheme({
        color: styles.getPropertyValue('--surface').trim(),
        symbolColor: styles.getPropertyValue('--ink').trim(),
        dark: query.matches,
      })
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [theme])

  /* -- Initial load -------------------------------------------------------- */

  useEffect(() => {
    const controller = new AbortController()
    Promise.all([api.fetchBootstrap(controller.signal), api.fetchRuntime(controller.signal)])
      .then(([nextBoot, nextRuntime]) => {
        setBoot(nextBoot)
        setSettings(nextBoot.defaults)
        setRuntime(nextRuntime)
        setOffline(null)
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        setOffline(readableError(error))
      })
    return () => controller.abort()
  }, [])

  /* -- Live polling --------------------------------------------------------
     Job and machine state are the only things that change without the operator
     doing anything, so they are the only things polled. The interval is torn
     down and the in-flight request aborted on unmount, so a late response can
     never write into a component that has gone.
     ----------------------------------------------------------------------- */

  useEffect(() => {
    const controller = new AbortController()

    const refresh = () => {
      Promise.all([api.fetchJobStatus(controller.signal), api.fetchRuntime(controller.signal)])
        .then(([nextJobs, nextRuntime]) => {
          setJobs(nextJobs)
          setRuntime(nextRuntime)
          setOffline(null)
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return
          setOffline(readableError(error))
        })
    }

    refresh()
    const timer = window.setInterval(refresh, POLL_INTERVAL_MS)
    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [])

  /* -- The plan loop -------------------------------------------------------
     The projection is the point of the Configure pane, so it re-requests
     whenever a control changes rather than waiting for a button. Debounced so
     that holding an arrow key on a number field does not queue a request per
     keystroke, and aborted on change so an older estimate can never land after
     a newer one.
     ----------------------------------------------------------------------- */

  const planRequest = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!settings) return undefined

    const timer = window.setTimeout(() => {
      planRequest.current?.abort()
      const controller = new AbortController()
      planRequest.current = controller
      setPlanPending(true)
      api
        .fetchPlan(settings, controller.signal)
        .then((next) => {
          setPlan(next)
          setOffline(null)
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return
          setOffline(readableError(error))
        })
        .finally(() => {
          if (!controller.signal.aborted) setPlanPending(false)
        })
    }, PLAN_DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [settings])

  /* -- Action plumbing ------------------------------------------------------ */

  const update = useCallback(
    <K extends keyof TrainingSettings>(key: K, value: TrainingSettings[K]) => {
      setSettings((current) => (current ? { ...current, [key]: value } : current))
      // Any configuration change invalidates the armed state. Arming is consent
      // for one specific run, not a mode the session stays in.
      setArmed(false)
    },
    [],
  )

  /** Run one action, with a busy label, and surface both outcomes as a message. */
  const run = useCallback(
    async <T,>(
      label: string,
      call: (signal: AbortSignal) => Promise<T>,
      onDone: (value: T) => Message | null,
    ) => {
      const controller = new AbortController()
      setBusy(label)
      setMessage(null)
      try {
        setMessage(onDone(await call(controller.signal)))
      } catch (error: unknown) {
        if (!controller.signal.aborted) setMessage({ tone: 'risk', text: readableError(error) })
      } finally {
        setBusy('')
      }
    },
    [],
  )

  const loadInventory = useCallback(
    () =>
      run('inventory', api.fetchCheckpoints, (value) => {
        setInventory(value)
        return null
      }),
    [run],
  )

  const loadCompare = useCallback(
    () =>
      run('compare', api.fetchCompare, (value) => {
        setCompare(value)
        return null
      }),
    [run],
  )

  // The Runs pane is empty until something loads it, so entering it loads it.
  // Guarded on `inventory` so switching tabs repeatedly does not re-fetch.
  useEffect(() => {
    if (pane === 'runs' && inventory === null) {
      void loadInventory()
      void loadCompare()
    }
  }, [pane, inventory, loadInventory, loadCompare])

  // The catalogue of local checkpoints, loaded once the pane is first opened.
  useEffect(() => {
    if (pane !== 'layers' || localModels.length > 0) return undefined
    const controller = new AbortController()
    api
      .fetchLocalModels(controller.signal)
      .then((value) => setLocalModels(value.models))
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setLayerNotice(readableError(error))
      })
    return () => controller.abort()
  }, [pane, localModels.length])

  // The layer map itself. Keyed on the folder and the sampling choice, and
  // aborted on change so a slow read of a large checkpoint cannot land after
  // the operator has already moved to another one.
  useEffect(() => {
    if (pane !== 'layers') return undefined
    const target = layerFolder ? { folder: layerFolder } : { modelId: settings?.modelId }
    if (!target.folder && !target.modelId) return undefined

    const controller = new AbortController()
    setLayerLoading(true)
    setLayerNotice(null)
    api
      .fetchLayerMap(target, layerSample, controller.signal)
      .then((value) => {
        if ('status' in value && value.status === 'blocked') {
          setLayerMap(null)
          setLayerNotice(value.message)
          return
        }
        const map = value as LayerMap
        setLayerMap(map)
        // Adopt the resolved folder so later requests address it directly,
        // and so the picker shows what is actually loaded.
        if (!layerFolder) setLayerFolder(map.name)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setLayerNotice(readableError(error))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLayerLoading(false)
      })
    return () => controller.abort()
  }, [pane, layerFolder, layerSample, settings?.modelId])

  useEffect(() => {
    if (pane !== 'runtime' || tensorboard !== null) return undefined
    const controller = new AbortController()
    api
      .fetchTensorBoard(controller.signal)
      .then(setTensorboard)
      .catch(() => {
        // A TensorBoard that is not running is the normal case, not an error
        // worth putting in front of the operator.
        if (!controller.signal.aborted) setTensorboard({ status: 'not running' })
      })
    return () => controller.abort()
  }, [pane, tensorboard])

  /* -- Derived state -------------------------------------------------------- */

  const activeJob = jobs.jobs.find((job) => job.running) ?? null
  const runState = activeJob ? 'Training' : (runtime?.state ?? 'Connecting')
  const canStart = Boolean(plan?.start_enabled) && armed && !activeJob && busy === ''

  const launchLine = useMemo(() => {
    if (!plan) return 'Waiting for the first estimate from the local API.'
    const { summary } = plan
    return `${summary.model} · ${summary.method} · ${summary.estimated_gb.toFixed(1)} GB estimated · fit ${summary.fit_state}`
  }, [plan])

  /* -- Launch actions ------------------------------------------------------- */

  const doDryRun = () => {
    if (!settings) return
    void run(
      'dry-run',
      (signal) => api.dryRun(settings, signal),
      (value) => {
        setReceipt(value)
        setInventory(null) // a new receipt exists; let the Runs pane re-read
        return { tone: 'ok', text: `Rehearsal complete. Receipt written: ${value.outputs.receipt_path ?? value.status}` }
      },
    )
  }

  const doStart = () => {
    if (!settings) return
    void run(
      'start',
      (signal) => api.startJob(settings, armed, signal),
      (value) => {
        setReceipt(value)
        setArmed(false)
        setInventory(null)
        if (value.status === 'blocked' || value.status === 'rejected' || value.status === 'failed') {
          return { tone: 'warn', text: value.error || `Training was not started: ${value.status}.` }
        }
        return { tone: 'warn', text: `Training started. ${value.outputs.output_root ?? ''}`.trim() }
      },
    )
  }

  const doStop = () => {
    void run(
      'stop',
      (signal) => api.stopJob(activeJob?.pid ?? receipt?.process?.pid ?? null, signal),
      (value) => ({ tone: value.status === 'stopped' ? 'ok' : 'warn', text: value.message }),
    )
  }

  /* -- Render --------------------------------------------------------------- */

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1 className="brand-name">ReTrain</h1>
          <span className="brand-note">
            {boot ? boot.subtitle : 'Local training console'}
          </span>
        </div>
        <div className="header-status">
          <Chip tone={activeJob ? 'warn' : toneForState(runtime?.state ?? '')}>{runState}</Chip>
          {runtime ? (
            <span className="brand-note">
              {runtime.vramFreeGb.toFixed(1)} GB free · {runtime.device}
            </span>
          ) : null}
          <label className="field-label" htmlFor="theme-choice">
            <span className="visually-hidden">Colour theme</span>
          </label>
          <select
            id="theme-choice"
            value={theme}
            onChange={(event) => setTheme(event.target.value as ThemeChoice)}
          >
            <option value="system">Match system</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      </header>

      <nav className="tabs" aria-label="Workspace sections">
        {PANES.map((item) => (
          <button
            key={item.id}
            type="button"
            className="tab"
            // aria-current is what tells a screen reader which pane is open.
            // The underline alone communicates nothing without sight.
            aria-current={pane === item.id ? 'page' : undefined}
            onClick={() => {
              // Writing the hash drives the listener above, which sets the
              // pane. One direction of truth, so the two cannot disagree.
              window.location.hash = item.id
              setPane(item.id)
            }}
          >
            {item.label}
            {item.id === 'runs' && inventory ? (
              <span className="tab-count">{inventory.count}</span>
            ) : null}
          </button>
        ))}
      </nav>

      <main className="app-main">
        <div className="pane">
          <div className="pane-inner">
            {offline ? (
              <Notice tone="warn">
                Local API unavailable — {offline} Start it with the ReTrain launcher, then this
                view will recover on its own.
              </Notice>
            ) : null}
            {message ? (
              <Notice tone={message.tone} onDismiss={() => setMessage(null)}>
                {message.text}
              </Notice>
            ) : null}

            {pane === 'configure' && boot && settings ? (
              <div className="configure">
                <Configure boot={boot} settings={settings} onChange={update} />
                <div className="projection-column">
                  <Projection plan={plan} pending={planPending} />
                </div>
              </div>
            ) : null}

            {pane === 'configure' && !settings ? (
              <Notice tone="neutral">Loading the model and dataset catalogue.</Notice>
            ) : null}

            {pane === 'layers' ? (
              <Layers
                models={localModels}
                folder={layerFolder}
                map={layerMap}
                loading={layerLoading}
                notice={layerNotice}
                sample={layerSample}
                onFolder={(next) => {
                  setLayerFolder(next)
                  setLayerMap(null)
                }}
                onSample={setLayerSample}
                onApplyScope={(lastNLayers) => {
                  // Two settings move together: the trainer only honours a
                  // layer count when the scope says to use one.
                  setSettings((current) =>
                    current ? { ...current, tuneScope: 'Last layers', lastNLayers } : current,
                  )
                  setArmed(false)
                  setMessage({
                    tone: 'ok',
                    text: `Training scope set to the last ${lastNLayers} layer${lastNLayers === 1 ? '' : 's'}. The estimate on Configure has been updated.`,
                  })
                  setPane('configure')
                  window.location.hash = 'configure'
                }}
              />
            ) : null}

            {pane === 'live' ? <Live /> : null}

            {pane === 'runs' ? (
              <Runs
                inventory={inventory}
                compare={compare}
                busy={busy}
                onRefresh={() => {
                  void loadInventory()
                  void loadCompare()
                }}
                onExport={(path) =>
                  void run(
                    'export',
                    (signal) => api.exportArtifact(path, signal),
                    (value) => ({ tone: 'ok', text: value.message ?? 'Exported.' }),
                  )
                }
                onDelete={(path) =>
                  void run(
                    'delete',
                    (signal) => api.deleteArtifact(path, signal),
                    (value) => {
                      setInventory(null)
                      return { tone: 'ok', text: value.message ?? 'Deleted.' }
                    },
                  )
                }
              />
            ) : null}

            {pane === 'runtime' && runtime ? (
              <Runtime
                runtime={runtime}
                tensorboard={tensorboard}
                busy={busy}
                onRefresh={() =>
                  void run('runtime', api.fetchRuntime, (value) => {
                    setRuntime(value)
                    return null
                  })
                }
                onStartTensorBoard={() =>
                  void run('tensorboard', api.startTensorBoard, (value) => {
                    setTensorboard(value)
                    return { tone: 'ok', text: `TensorBoard ${value.status}${value.url ? ` at ${value.url}` : ''}` }
                  })
                }
              />
            ) : null}

            {pane === 'runtime' && !runtime ? (
              <Notice tone="neutral">Reading machine state.</Notice>
            ) : null}
          </div>
        </div>
      </main>

      {/* The launch bar. A shell row, so it is present at every window size and
          on every pane — the operator never has to go looking for Stop. */}
      <div className="launch-bar">
        <div className="launch-summary">
          <span className="launch-line">{launchLine}</span>
        </div>
        <div className="button-row">
          <button
            type="button"
            className="button"
            onClick={doDryRun}
            disabled={!settings || busy !== ''}
          >
            {busy === 'dry-run' ? 'Rehearsing' : 'Dry run'}
          </button>

          <label className="arm" data-armed={armed}>
            <input
              type="checkbox"
              checked={armed}
              disabled={!plan?.start_enabled || Boolean(activeJob)}
              onChange={(event) => setArmed(event.target.checked)}
            />
            {armed ? 'Armed' : 'Arm GPU run'}
          </label>

          <button
            type="button"
            className="button"
            data-role="primary"
            onClick={doStart}
            disabled={!canStart}
            // Explains the disabled state, which is otherwise invisible.
            title={
              !plan?.start_enabled
                ? 'The server has not cleared this configuration to start.'
                : !armed
                  ? 'Arm the run first.'
                  : 'Start training.'
            }
          >
            {busy === 'start' ? 'Starting' : 'Start training'}
          </button>

          <button
            type="button"
            className="button"
            data-role="danger"
            onClick={doStop}
            disabled={!activeJob || busy !== ''}
          >
            {busy === 'stop' ? 'Stopping' : 'Stop'}
          </button>
        </div>
      </div>
    </div>
  )
}
