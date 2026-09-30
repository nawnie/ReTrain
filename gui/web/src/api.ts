/**
 * Typed client for the local ReTrain API.
 *
 * Design rules this file follows, because they are the ones that went wrong
 * before:
 *
 *  - Every call takes an AbortSignal. A pane that unmounts or a poll that is
 *    superseded must be able to cancel, or a late response overwrites fresh
 *    state with stale state.
 *  - Every call either returns a typed value or throws ApiError. Nothing
 *    returns a promise that a caller can silently drop; nothing resolves to
 *    undefined on failure.
 *  - Error text is normalised here so the UI never renders a raw HTML error
 *    page into a status line.
 */

import type {
  ActionResult,
  BlockedResult,
  Bootstrap,
  GpuSample,
  LiveLayers,
  LiveRun,
  LiveRunSummary,
  LiveSamples,
  CheckpointInventory,
  CompareSummary,
  JobReceipt,
  JobStatus,
  LayerMap,
  LocalModel,
  RuntimeStatus,
  StopResult,
  TensorBoardStatus,
  TrainingPlan,
  TrainingSettings,
} from './types'

const BASE = '/api/retrain'

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Strip markup and collapse whitespace so an error is safe to put in a chip. */
export function readableError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  const text = raw.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
  if (!text) return 'The local API did not explain the failure.'
  return text.length > 200 ? `${text.slice(0, 197)}...` : text
}

async function request<T>(
  path: string,
  init: RequestInit,
  signal: AbortSignal | undefined,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, { ...init, signal })
  } catch (cause) {
    // A network-level failure here almost always means the local API is not
    // running, which is a different problem from a 500 and deserves different
    // words on screen.
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError('The local ReTrain API is not reachable.', 0)
  }
  if (!response.ok) {
    const body = await response.text().catch(() => '')
    throw new ApiError(body || `${response.status} ${response.statusText}`, response.status)
  }
  return (await response.json()) as T
}

function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'GET' }, signal)
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    },
    signal,
  )
}

/* -- Catalogue and machine state ------------------------------------------- */

export const fetchBootstrap = (signal?: AbortSignal) => get<Bootstrap>('/bootstrap', signal)
export const fetchRuntime = (signal?: AbortSignal) => get<RuntimeStatus>('/runtime', signal)

/* -- Planning and launching ------------------------------------------------- */

export const fetchPlan = (settings: TrainingSettings, signal?: AbortSignal) =>
  post<TrainingPlan>('/plan', settings, signal)

/** A rehearsal. Writes the same receipt shape without starting training. */
export const dryRun = (settings: TrainingSettings, signal?: AbortSignal) =>
  post<JobReceipt>('/jobs/dry-run', { ...settings, dryRun: true, confirmed: false }, signal)

/** The real thing. `confirmed` is the server-side gate, set from the arm switch. */
export const startJob = (settings: TrainingSettings, confirmed: boolean, signal?: AbortSignal) =>
  post<JobReceipt>('/jobs/start', { ...settings, dryRun: false, confirmed }, signal)

export const fetchJobStatus = (signal?: AbortSignal) => get<JobStatus>('/jobs/status', signal)

export const stopJob = (pid: number | null, signal?: AbortSignal) =>
  post<StopResult>('/jobs/stop', { pid }, signal)

/* -- Runs and artifacts ----------------------------------------------------- */

export const fetchCheckpoints = (signal?: AbortSignal) =>
  get<CheckpointInventory>('/checkpoints', signal)

export const fetchCompare = (signal?: AbortSignal) => get<CompareSummary>('/compare', signal)

export const exportArtifact = (path: string, signal?: AbortSignal) =>
  post<ActionResult>('/artifacts/export', { path }, signal)

export const deleteArtifact = (path: string, signal?: AbortSignal) =>
  post<ActionResult>('/artifacts/delete', { path }, signal)

/* -- Tooling ---------------------------------------------------------------- */

export const fetchTensorBoard = (signal?: AbortSignal) =>
  get<TensorBoardStatus>('/tensorboard/status', signal)

export const startTensorBoard = (signal?: AbortSignal) =>
  post<TensorBoardStatus>('/tensorboard/start', {}, signal)

/**
 * Check the QLoRA engine without installing it.
 *
 * `execute: false` is deliberate and must stay that way: this is called from a
 * read-only status view, and an install is a side effect the operator has to
 * ask for explicitly.
 */
export const checkEngine = (signal?: AbortSignal) =>
  post<{ status: { ready: boolean; missing: string[] } }>(
    '/engines/qlora/install',
    { execute: false },
    signal,
  )

export const savePreset = (settings: TrainingSettings, name: string, signal?: AbortSignal) =>
  post<{ preset: { name: string } }>('/presets/save', { ...settings, name }, signal)

/* -- Model layer map -------------------------------------------------------- */

export const fetchLocalModels = (signal?: AbortSignal) =>
  get<{ models: LocalModel[] }>('/models/local', signal)

/**
 * Layer geometry for one local checkpoint.
 *
 * `sample` controls whether weight statistics are measured. Structure alone is
 * effectively instant at any model size because it comes from the file header;
 * sampling reads a bounded slice of the actual weights and costs well under a
 * second for a 3GB checkpoint, but that is not free, so the caller chooses.
 */
export const fetchLayerMap = (
  target: { folder?: string; modelId?: string },
  sample: boolean,
  signal?: AbortSignal,
) => {
  const query = new URLSearchParams()
  if (target.folder) query.set('folder', target.folder)
  if (target.modelId) query.set('modelId', target.modelId)
  query.set('sample', sample ? 'true' : 'false')
  return get<LayerMap | BlockedResult>(`/models/layers?${query.toString()}`, signal)
}

/* -- Live training view ------------------------------------------------------- */

export const fetchLiveRuns = (signal?: AbortSignal) => get<{ runs: LiveRunSummary[] }>('/live/runs', signal)

export const fetchLiveRun = (id: string, signal?: AbortSignal) =>
  get<LiveRun | BlockedResult>(`/live/run?id=${encodeURIComponent(id)}`, signal)

export const fetchLiveLayers = (id: string, signal?: AbortSignal) =>
  get<LiveLayers>(`/live/layers?id=${encodeURIComponent(id)}`, signal)

export const fetchLiveSamples = (id: string, signal?: AbortSignal) =>
  get<LiveSamples>(`/live/samples?id=${encodeURIComponent(id)}`, signal)

export const fetchLiveSystem = (signal?: AbortSignal) => get<{ samples: GpuSample[] }>('/live/system', signal)
