/**
 * Shapes returned by the local ReTrain API (gui/api/app.py).
 *
 * These mirror the server's own payloads rather than an idealised model, so a
 * mismatch shows up as a type error here instead of as undefined on screen.
 * Fields the backend may legitimately omit are optional; nothing is `any`.
 */

/* -- Catalogue: what the operator can pick from ----------------------------- */

export interface ModelOption {
  id: string
  name: string
  family: string
  sizeB: number
  backend: string
  status: string
}

export interface DatasetOption {
  id: string
  name: string
  source: string
  rows: number
  rejected: number
  status: string
}

/** Every training control the operator can set. Mirrors bootstrap.defaults. */
export interface TrainingSettings {
  modelId: string
  datasetId: string
  method: string
  tuneScope: string
  lastNLayers: number
  beta: number
  numGenerations: number
  rewardKind: string
  contextLength: number
  microBatch: number
  gradAccum: number
  loraRank: number
  precision: string
  optimizer: string
  scheduler: string
  trustRemoteCode: boolean
  gradientCheckpointing: boolean
  flashAttention: boolean
  qlora: boolean
  cpuOffload: boolean
  lowVramMode: boolean
  tensorboard: boolean
  confirmed: boolean
  dryRun: boolean
}

export interface Bootstrap {
  workspaceName: string
  subtitle: string
  version: string
  activeDatasetLabel: string
  models: ModelOption[]
  datasets: DatasetOption[]
  defaults: TrainingSettings
}

/* -- Runtime: the machine this will actually run on ------------------------- */

export interface DependencyStatus {
  package: string
  label: string
  available: boolean
  version?: string | null
}

export interface ResourceMeter {
  label: string
  value: string
  percent: number
  /** The server's own tone hint. Rendered alongside text, never as hue alone. */
  tone: 'mint' | 'amber' | 'neutral'
}

export interface EngineStatus {
  ready: boolean
  missing?: string[]
}

export interface RuntimeStatus {
  state: string
  backend: string
  device: string
  vramTotalGb: number
  vramFreeGb: number
  engine: EngineStatus
  dependencies: DependencyStatus[]
  resources: ResourceMeter[]
}

/* -- Plan: the server's verdict on whether this run fits -------------------- */

export type FitState = 'safe' | 'tight' | 'unsafe'
export type PlanStatus = 'ready' | 'warning' | 'blocked'
export type GateState = 'ready' | 'warning' | 'blocked'

export interface VramLine {
  item: string
  gb: number
  detail: string
}

export interface VramEstimate {
  estimated_gb: number
  limit_gb: number
  headroom_gb: number
  percent: number
  fit_state: FitState
  breakdown: VramLine[]
  warnings: string[]
}

export interface SafetyGate {
  gate: string
  state: GateState
  detail: string
}

export interface PlanValidation {
  status: PlanStatus
  start_enabled: boolean
  estimate: VramEstimate
  dependencies: DependencyStatus[]
  gates: SafetyGate[]
  notes: string[]
}

export interface PlanSummary {
  dataset_version: string
  model: string
  method: string
  fit_state: FitState
  estimated_gb: number
}

export interface TrainingPlan {
  status: PlanStatus
  start_enabled: boolean
  summary: PlanSummary
  validation: PlanValidation
}

/* -- Receipts: what a launched or rehearsed run leaves behind --------------- */

export interface JobReceipt {
  schema_version?: number
  created_at: string
  status: string
  execute_requested: boolean
  summary: Partial<PlanSummary>
  outputs: {
    output_root?: string
    receipt_path?: string
    tensorboard_logdir?: string
  }
  process?: { pid?: number | null } | null
  error?: string | null
}

export interface JobSnapshot {
  pid: number
  status: string
  running: boolean
  return_code: number | null
  created_at: string
  receipt_path: string
  log_path: string
  summary: Partial<PlanSummary>
}

export interface JobStatus {
  status: 'idle' | 'running'
  active_count: number
  jobs: JobSnapshot[]
}

export interface StopResult {
  status: string
  message: string
  pid: number | null
}

/* -- Artifacts and comparison ---------------------------------------------- */

export interface RunArtifact {
  id: string
  label: string
  path: string
  created_at: string
  /** Display-formatted timestamp; the API sends both this and created_at. */
  time?: string
  status: string
  kind: string
  type: string
  size: string
  model: string
  method: string
  fit_state: string
  estimated_gb?: number | null
}

export interface CheckpointInventory {
  count: number
  items: RunArtifact[]
}

export interface CompareMetricRow {
  label: string
  baseline: string
  candidate_a: string
  candidate_b: string
}

export interface CompareSummary {
  runs: { id: string; label: string }[]
  metrics: CompareMetricRow[]
}

/* -- Small action results --------------------------------------------------- */

export interface ActionResult {
  status: string
  message?: string
  path?: string
  url?: string
}

export interface TensorBoardStatus {
  status: string
  url?: string
  message?: string
}

/* -- Model layer map -------------------------------------------------------
   Read from the safetensors headers by gui/api/model_layers.py. Structural
   fields are exact; anything under `stats` is sampled and must be presented
   as such.
   -------------------------------------------------------------------------- */

export interface BlockStats {
  sampled: number
  rms: number
  mean_abs: number
  peak: number
  active: number
  non_finite: number
}

export interface LayerBlock {
  key: string
  label: string
  /** Module role: attn.q, mlp.gate, norm, and so on. */
  kind: string
  params: number
  bytes: number
  tensor_count: number
  /** Whether a LoRA adapter can attach here. Norms and embeddings cannot. */
  adapter_target: boolean
  stats: BlockStats | null
}

export interface ModelLayer {
  index: number
  label: string
  params: number
  bytes: number
  blocks: LayerBlock[]
  stats: BlockStats | null
}

export interface LayerMap {
  schema: string
  name: string
  path: string
  architecture: string
  hidden_size: number | null
  intermediate_size: number | null
  num_attention_heads: number | null
  num_key_value_heads: number | null
  declared_layers: number | null
  shards: string[]
  dtypes: Record<string, number>
  total_params: number
  total_bytes: number
  layer_count: number
  layers: ModelLayer[]
  terminals: LayerBlock[]
  /** True when weight statistics were sampled; false means structure only. */
  sampled: boolean
  sample_plan: { windows: number; window_bytes: number } | null
  adapter_target_roles: string[]
}

export interface LocalModel {
  folder: string
  path: string
  shards: number
  bytes: number
}

/** The API returns this shape instead of a payload when a request is refused. */
export interface BlockedResult {
  status: 'blocked'
  message: string
}

/* -- Live training view (backend/run_telemetry.py) --------------------------- */

/** One training run the console can watch; the list is newest first. */
export interface LiveRunSummary {
  id: string
  title: string
  status: LiveStatus
  progress: number
  epoch: number
  totalEpochs: number
  lossNow: number | null
  started: number | null
  demo?: boolean
}

export type LiveStatus = 'loading' | 'training' | 'done' | 'failed' | 'stopped'

/** The "learning check" a trainer prints after each epoch: does it actually produce the right answers on unseen examples? */
export interface LiveCheck {
  tag: string
  epoch?: number
  n?: number
  step?: number
  eval_loss?: number | null
  eval_perplexity?: number | null
  /** the four scores are PERCENTAGES, 0 to 100 */
  valid_json?: number
  actions_match?: number
  first_button_match?: number
  exact?: number
  seconds?: number
  by_category?: Record<string, { n: number; actions_match: number }>
}

export interface LiveRun {
  id: string
  title: string
  status: LiveStatus
  loadingPercent: number
  epoch: number
  totalEpochs: number
  progress: number
  elapsedSeconds: number
  etaSeconds: number | null
  examplesPerSecond: number | null
  trainExamples: number
  trainable: string
  /** [epoch, loss, grad_norm, learning_rate, unix time] per report */
  loss: [number, number, number | null, number | null, number][]
  lossNow: number | null
  lossBest: number | null
  evals: [number, number][]
  checks: LiveCheck[]
  peakVram: string
  savedTo: string
  error: string
  messages: string[]
  log: string
  pid: number | null
  hasLayers: boolean
  demo?: boolean
  startedAt?: number
}

export interface LiveLayers {
  available: boolean
  reason?: string
  latest?: { step: number; epoch: number; layers: { i: number; attn: number; mlp: number; upd?: number }[] }
  totals?: { i: number; attn: number; mlp: number }[]
  readings?: number
  /** steps x layers grids for the heatmap: gradient size and adapter size */
  matrix?: { steps: number[]; epochs: number[]; layers: number[]; grad: number[][]; upd: number[][] }
}

/** One held-out question and the model's answer at each learning check (baseline, epoch 1, ...). */
export interface LiveSampleItem {
  i: number
  category: string
  prompt: string
  expected: string
  answers: Record<string, { got: string; valid_json: boolean; actions_match: boolean; first_button_match: boolean; exact: boolean }>
}

export interface LiveSamples {
  available: boolean
  tags: string[]
  items: LiveSampleItem[]
}

export interface GpuSample {
  t: number
  usedGb: number
  totalGb: number
  util: number
  tempC: number
  powerW: number
  clockMhz: number
}
