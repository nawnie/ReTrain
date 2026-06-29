import { useCallback, useEffect, useMemo, useState } from "react";

type HealthResponse = {
  ok: boolean;
  tensorboard: {
    embeddedPath: string;
    logdir: string;
    port: number;
    running: boolean;
  };
};

type DatasetOption = {
  id: string;
  label: string;
  path: string;
  source: string;
  runnerReady: boolean;
  kind: string;
  counts: Record<string, number>;
  notes: string[];
};

type ModelOption = {
  id: string;
  label: string;
  path: string;
  family: string;
  sizeB: number;
  runnerKey: string;
  exists: boolean;
  source: string;
};

type Dependency = {
  package: string;
  label: string;
  available: boolean;
};

type TrainingConfig = {
  name: string;
  datasetId: string;
  datasetPath: string;
  modelId: string;
  modelPath: string;
  method: string;
  precision: string;
  contextLength: number;
  microBatchSize: number;
  gradientAccumulationSteps: number;
  loraRank: number;
  learningRate: number;
  maxSteps: number;
  maxTrainRecords: number;
  maxEvalRecords: number;
  maxTestRecords: number;
  maxVramGb: number;
  freeVramGb: number;
  gradientCheckpointing: boolean;
  tensorboard: boolean;
  saveFinal: boolean;
  saveCheckpoints: boolean;
  dryRun: boolean;
  confirmed: boolean;
};

type Gate = {
  gate: string;
  state: "ready" | "warning" | "blocked" | string;
  detail: string;
};

type PlanResponse = {
  status: "ready" | "warning" | "blocked" | string;
  startEnabled: boolean;
  config: TrainingConfig;
  dataset: DatasetOption;
  model: ModelOption;
  estimate: {
    fitState: "safe" | "tight" | "unsafe" | string;
    estimatedGb: number;
    limitGb: number;
    headroomGb: number;
    percent: number;
    warnings: string[];
    breakdown: Array<{ item: string; gb: number; detail: string }>;
  };
  dependencies: Dependency[];
  gates: Gate[];
  command: string;
  outputs: {
    outputRoot: string;
    stateRoot: string;
    tensorboardLogdir: string;
  };
};

type RunRecord = {
  id: string;
  name: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  executeRequested: boolean;
  plan: PlanResponse;
  process: { pid?: number; exitCode?: number } | null;
  paths: {
    stateDir: string;
    outputRoot: string;
    logPath: string;
    receiptPath: string;
  };
  metrics: {
    metricsFileCount: number;
    eventFileCount: number;
    latestMetricsPath: string;
    latest: Record<string, string | number | null>;
  };
  logTail: string;
};

type OverviewResponse = {
  schemaVersion: string;
  repo: {
    root: string;
    mokRoot: string;
    runner: string;
    stateRoot: string;
    outputRoot: string;
  };
  hardware: {
    gpu: string;
    vramTotalMb: number | null;
    vramFreeMb: number | null;
  };
  datasets: DatasetOption[];
  models: ModelOption[];
  dependencies: Dependency[];
  defaultConfig: TrainingConfig;
  runs: RunRecord[];
};

type TensorBoardSummaryResponse = {
  ready: boolean;
  error: string | null;
  runs: string[];
  scalars: Array<{
    run: string;
    tag: string;
    step: number;
    value: number;
    delta: number;
  }>;
};

type TensorBoardStartResponse = {
  ready: boolean;
  error: string | null;
  embeddedPath: string;
  logdir: string;
  port: number;
};

const API_BASE = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

const numberFields: Array<keyof TrainingConfig> = [
  "contextLength",
  "microBatchSize",
  "gradientAccumulationSteps",
  "loraRank",
  "learningRate",
  "maxSteps",
  "maxTrainRecords",
  "maxEvalRecords",
  "maxTestRecords",
  "maxVramGb",
  "freeVramGb",
];

function scrubLocalPaths(value: string) {
  return value
    .replace(/C:\\Users\\[^\\]+\\Desktop\\Rnv1-ReTrain\\/g, "Rnv1-ReTrain\\")
    .replace(/C:\\Users\\[^\\]+\\Desktop\\MoK-Project\\/g, "MoK-Project\\")
    .replace(/C:\\Users\\[^\\]+\\/g, "%USERPROFILE%\\")
    .replace(/F:\\Ai_Models\\hf\\posttrain_candidates\\/g, "HF candidates\\")
    .replace(/F:\\datasets\\/g, "datasets\\")
    .replace(/Rnv1-ReTrain\\\.venv\\Scripts\\python\.exe/g, "python")
    .replace(/Rnv1-ReTrain\\scripts\\run_posttrain_bakeoff\.py/g, "scripts\\run_posttrain_bakeoff.py")
    .replace(/MoK-Project\\training\\posttrain_bakeoff\\data/g, "MoK-Project\\...\\posttrain_bakeoff\\data")
    .replace(/MoK-Project\\training\\core_mok\\splits/g, "MoK-Project\\...\\core_mok\\splits")
    .replace(/Rnv1-ReTrain\\training\\runs\\([^\s"]+)/g, "Rnv1-ReTrain\\...\\$1")
    .replace(/Rnv1-ReTrain\\training\\run_state\\([^\s"]+)/g, "Rnv1-ReTrain\\...\\$1");
}

function compactPath(value?: string) {
  if (!value) return "-";
  const scrubbed = scrubLocalPaths(value);
  const parts = scrubbed.split(/[\\/]+/).filter(Boolean);
  if ((parts[0] === "Rnv1-ReTrain" || parts[0] === "MoK-Project") && parts.includes("training")) {
    return `${parts[0]}\\...\\${parts[parts.length - 1]}`;
  }
  if (parts.length <= 3) return scrubbed;
  return `${parts[0]}\\...\\${parts[parts.length - 2]}\\${parts[parts.length - 1]}`;
}

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [summary, setSummary] = useState<TensorBoardSummaryResponse | null>(null);
  const [tensorboard, setTensorboard] = useState<TensorBoardStartResponse | null>(null);
  const [tensorboardOpen, setTensorboardOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runs = overview?.runs ?? [];
  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? runs[0] ?? null,
    [runs, selectedRunId],
  );
  const activeRuns = runs.filter((run) => run.status === "running").length;
  const scalarCount = summary?.scalars.length ?? 0;
  const tensorboardReady = tensorboard?.ready ?? health?.tensorboard.running ?? false;
  const embeddedPath = `${API_BASE}${tensorboard?.embeddedPath ?? health?.tensorboard.embeddedPath ?? "/tensorboard/"}`;
  const canStartRun = Boolean(config && plan?.startEnabled && !busy);
  const startBlockedReason = plan?.startEnabled === false ? "Resolve readiness gates before starting." : undefined;

  const refresh = useCallback(async () => {
    setError(null);
    const [nextHealth, nextOverview, nextSummary] = await Promise.all([
      readJson<HealthResponse>("/api/health"),
      readJson<OverviewResponse>("/api/training/overview"),
      readJson<TensorBoardSummaryResponse>("/api/tensorboard/summary"),
    ]);
    setHealth(nextHealth);
    setOverview(nextOverview);
    setSummary(nextSummary);
    setConfig((current) => current ?? nextOverview.defaultConfig);
    if (!selectedRunId && nextOverview.runs[0]) {
      setSelectedRunId(nextOverview.runs[0].id);
    }
  }, [selectedRunId]);

  useEffect(() => {
    refresh().catch((caught) => {
      setError(caught instanceof Error ? caught.message : "Unable to load ReTrain.");
    });
  }, [refresh]);

  useEffect(() => {
    if (!config || !overview) return;
    const timer = window.setTimeout(() => {
      readJson<PlanResponse>("/api/training/runs/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })
        .then(setPlan)
        .catch((caught) => {
          setError(caught instanceof Error ? caught.message : "Unable to refresh plan.");
        });
    }, 250);
    return () => window.clearTimeout(timer);
  }, [config, overview]);

  useEffect(() => {
    if (!activeRuns) return;
    const interval = window.setInterval(() => {
      refresh().catch((caught) => {
        setError(caught instanceof Error ? caught.message : "Unable to refresh active run.");
      });
    }, 4000);
    return () => window.clearInterval(interval);
  }, [activeRuns, refresh]);

  const updateConfig = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    setConfig((current) => (current ? { ...current, [key]: value } : current));
  };

  const planRun = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      const nextPlan = await readJson<PlanResponse>("/api/training/runs/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      setPlan(nextPlan);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to plan run.");
    } finally {
      setBusy(false);
    }
  };

  const startRun = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      const run = await readJson<RunRecord>("/api/training/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...config, execute: true }),
      });
      setSelectedRunId(run.id);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start run.");
    } finally {
      setBusy(false);
    }
  };

  const stopRun = async () => {
    if (!selectedRun) return;
    setBusy(true);
    setError(null);
    try {
      await readJson<RunRecord>(`/api/training/runs/${selectedRun.id}/stop`, { method: "POST" });
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to stop run.");
    } finally {
      setBusy(false);
    }
  };

  const resumeRun = async () => {
    if (!selectedRun) return;
    setBusy(true);
    setError(null);
    try {
      const run = await readJson<RunRecord>(`/api/training/runs/${selectedRun.id}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ execute: false }),
      });
      setSelectedRunId(run.id);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to resume run.");
    } finally {
      setBusy(false);
    }
  };

  const startTensorBoard = async () => {
    setBusy(true);
    setError(null);
    try {
      const nextTensorboard = await readJson<TensorBoardStartResponse>("/api/tensorboard/start", { method: "POST" });
      setTensorboard(nextTensorboard);
      await refresh();
      if (!nextTensorboard.ready) {
        setError(nextTensorboard.error ?? "TensorBoard did not report ready.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start TensorBoard.");
    } finally {
      setBusy(false);
    }
  };

  const setDataset = (datasetId: string) => {
    const dataset = overview?.datasets.find((item) => item.id === datasetId);
    setConfig((current) =>
      current
        ? {
            ...current,
            datasetId,
            datasetPath: dataset?.path ?? current.datasetPath,
          }
        : current,
    );
  };

  const setModel = (modelId: string) => {
    const model = overview?.models.find((item) => item.id === modelId);
    setConfig((current) =>
      current
        ? {
            ...current,
            modelId,
            modelPath: model?.exists ? model.path : current.modelPath,
          }
        : current,
    );
  };

  const selectedDataset = overview?.datasets.find((item) => item.id === config?.datasetId);
  const selectedModel = overview?.models.find((item) => item.id === config?.modelId);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Rnv1 ReTrain</p>
          <h1>Training Runs</h1>
        </div>
        <div className="actions">
          <button type="button" onClick={refresh} disabled={busy}>
            Refresh
          </button>
          <button type="button" onClick={startTensorBoard} disabled={busy}>
            Start TensorBoard
          </button>
          <button className="primaryButton" type="button" onClick={() => setTensorboardOpen(true)} disabled={!tensorboardReady}>
            Open TensorBoard
          </button>
        </div>
      </header>

      <section className="metricStrip" aria-label="Training status">
        <Metric label="FastAPI" value={health?.ok ? "Online" : "Checking"} tone={health?.ok ? "good" : "warn"} />
        <Metric label="Active runs" value={activeRuns} tone={activeRuns ? "warn" : "neutral"} />
        <Metric label="Latest status" value={selectedRun?.status ?? "none"} tone={toneForStatus(selectedRun?.status)} />
        <Metric label="Fit" value={plan?.estimate.fitState ?? "unplanned"} tone={toneForFit(plan?.estimate.fitState)} />
        <Metric label="Scalar tags" value={scalarCount} tone={scalarCount ? "good" : "neutral"} />
        <Metric label="GPU" value={overview?.hardware.gpu ?? "unknown"} tone="neutral" wide />
      </section>

      {error ? <p className="error">{error}</p> : null}

      <div className="workspace">
        <aside className="configPanel" aria-label="Run configuration">
          <header className="panelHeader">
            <div>
              <p className="eyebrow">Run setup</p>
              <h2>Config</h2>
            </div>
            <span className={`pill ${plan?.status ?? "neutral"}`}>{plan?.status ?? "new"}</span>
          </header>

          {config ? (
            <div className="controlGrid">
              <label className="field wide">
                <span>Name</span>
                <input value={config.name} onChange={(event) => updateConfig("name", event.target.value)} />
              </label>
              <label className="field wide">
                <span>Dataset</span>
                <select value={config.datasetId} onChange={(event) => setDataset(event.target.value)}>
                  {overview?.datasets.map((dataset) => (
                    <option value={dataset.id} key={dataset.id}>
                      {dataset.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field wide">
                <span>Model</span>
                <select value={config.modelId} onChange={(event) => setModel(event.target.value)}>
                  {overview?.models.map((model) => (
                    <option value={model.id} key={model.id}>
                      {model.exists ? "" : "missing - "}
                      {model.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Method</span>
                <select value={config.method} onChange={(event) => updateConfig("method", event.target.value)}>
                  <option value="qlora">QLoRA</option>
                  <option value="lora">LoRA</option>
                  <option value="sft">SFT</option>
                </select>
              </label>
              <label className="field">
                <span>Precision</span>
                <select value={config.precision} onChange={(event) => updateConfig("precision", event.target.value)}>
                  <option value="4bit">4-bit</option>
                  <option value="8bit">8-bit</option>
                  <option value="bf16">BF16</option>
                  <option value="fp16">FP16</option>
                </select>
              </label>

              {numberFields.map((field) => (
                <label className="field" key={field}>
                  <span>{labelForField(field)}</span>
                  <input
                    type="number"
                    value={Number(config[field])}
                    step={field === "learningRate" ? "0.000001" : "1"}
                    onChange={(event) => updateConfig(field, Number(event.target.value) as never)}
                  />
                </label>
              ))}

              <label className="check">
                <input
                  type="checkbox"
                  checked={config.gradientCheckpointing}
                  onChange={(event) => updateConfig("gradientCheckpointing", event.target.checked)}
                />
                <span>Gradient checkpointing</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={config.tensorboard}
                  onChange={(event) => updateConfig("tensorboard", event.target.checked)}
                />
                <span>TensorBoard</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={config.dryRun}
                  onChange={(event) => updateConfig("dryRun", event.target.checked)}
                />
                <span>Dry run</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={config.confirmed}
                  onChange={(event) => updateConfig("confirmed", event.target.checked)}
                />
                <span>Confirmed</span>
              </label>
            </div>
          ) : (
            <div className="loadingBox">Loading</div>
          )}

          <div className="buttonRow">
            <button type="button" onClick={planRun} disabled={busy || !config}>
              Plan
            </button>
            <button className="primaryButton" type="button" onClick={startRun} disabled={!canStartRun} title={startBlockedReason}>
              {config?.dryRun ? "Run Dry Check" : "Start Confirmed Run"}
            </button>
          </div>
        </aside>

        <section className="mainPanel" aria-label="Run dashboard">
          <div className="split">
            <section className="planPanel">
              <header className="panelHeader">
                <div>
                  <p className="eyebrow">Readiness</p>
                  <h2>{selectedDataset?.label ?? "Dataset"}</h2>
                </div>
                <span className={`pill ${selectedDataset?.runnerReady ? "ready" : "blocked"}`}>
                  {selectedDataset?.runnerReady ? "split-ready" : "needs split"}
                </span>
              </header>

              <div className="fitRow">
                <div>
                  <span>Model</span>
                  <strong>{selectedModel?.label ?? "none"}</strong>
                </div>
                <div>
                  <span>Estimate</span>
                  <strong>{plan ? `${plan.estimate.estimatedGb} GB` : "-"}</strong>
                </div>
                <div>
                  <span>Headroom</span>
                  <strong>{plan ? `${plan.estimate.headroomGb} GB` : "-"}</strong>
                </div>
              </div>

              <div className="vramTrack">
                <span style={{ width: `${Math.min(100, plan?.estimate.percent ?? 0)}%` }} />
              </div>

              <div className="gateList">
                {(plan?.gates ?? []).map((item) => (
                  <div className={`gate ${item.state}`} key={`${item.gate}-${item.detail}`}>
                    <span>{item.gate}</span>
                    <strong>{item.state}</strong>
                    <small>{compactPath(item.detail)}</small>
                  </div>
                ))}
              </div>

              <div className="commandBox">
                <span>Command</span>
                <code>{plan?.command ? scrubLocalPaths(plan.command) : "No plan yet"}</code>
              </div>
            </section>

            <section className="runPanel">
              <header className="panelHeader">
                <div>
                  <p className="eyebrow">Runs</p>
                  <h2>History</h2>
                </div>
                <div className="buttonRow compact">
                  <button type="button" onClick={stopRun} disabled={busy || !selectedRun || selectedRun.status !== "running"}>
                    Stop
                  </button>
                  <button className="primaryButton" type="button" onClick={resumeRun} disabled={busy || !selectedRun}>
                    Resume
                  </button>
                </div>
              </header>
              <div className="runTable" role="table" aria-label="Training runs">
                <div className="runRow heading" role="row">
                  <span>Name</span>
                  <span>Status</span>
                  <span>Method</span>
                  <span>Events</span>
                </div>
                {runs.map((run) => (
                  <button
                    type="button"
                    className={`runRow ${run.id === selectedRun?.id ? "selected" : ""}`}
                    key={run.id}
                    onClick={() => setSelectedRunId(run.id)}
                  >
                    <span>{run.name}</span>
                    <span>{run.status}</span>
                    <span>{run.plan.config.method}</span>
                    <span>{run.metrics.eventFileCount}</span>
                  </button>
                ))}
                {!runs.length ? <div className="emptyLine">No runs yet</div> : null}
              </div>
            </section>
          </div>

          <section className="detailPanel" aria-label="Selected run details">
            <header className="panelHeader">
              <div>
                <p className="eyebrow">Selected run</p>
                <h2>{selectedRun?.name ?? "None"}</h2>
              </div>
              <span className={`pill ${toneForStatus(selectedRun?.status)}`}>{selectedRun?.status ?? "none"}</span>
            </header>
            <div className="detailGrid">
              <Metric label="Train loss" value={metricValue(selectedRun, "trainLoss")} tone="neutral" />
              <Metric label="Eval loss" value={metricValue(selectedRun, "evalLoss")} tone="neutral" />
              <Metric label="Perplexity" value={metricValue(selectedRun, "evalPerplexity")} tone="neutral" />
              <Metric label="Peak VRAM" value={metricValue(selectedRun, "peakAllocatedGb", " GB")} tone="neutral" />
              <Metric label="Metrics files" value={selectedRun?.metrics.metricsFileCount ?? 0} tone="neutral" />
              <Metric label="Events" value={selectedRun?.metrics.eventFileCount ?? 0} tone="neutral" />
            </div>
            <div className="artifactGrid">
              <PathLine label="Output" value={selectedRun?.paths.outputRoot} />
              <PathLine label="Receipt" value={selectedRun?.paths.receiptPath} />
              <PathLine label="Log" value={selectedRun?.paths.logPath} />
              <PathLine label="Metrics" value={selectedRun?.metrics.latestMetricsPath} />
            </div>
            <pre className="logBox">{selectedRun?.logTail ? scrubLocalPaths(selectedRun.logTail) : "No log output yet"}</pre>
          </section>
        </section>
      </div>

      {tensorboardOpen ? (
        <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="TensorBoard popup">
          <section className="tensorboardModal">
            <header>
              <h2>TensorBoard</h2>
              <button type="button" onClick={() => setTensorboardOpen(false)}>
                Close
              </button>
            </header>
            {tensorboardReady ? <iframe title="TensorBoard" src={embeddedPath} /> : <div className="loadingBox">Stopped</div>}
          </section>
        </div>
      ) : null}
    </main>
  );
}

function Metric({
  label,
  value,
  tone,
  wide = false,
}: {
  label: string;
  value: string | number | null | undefined;
  tone: string;
  wide?: boolean;
}) {
  return (
    <div className={`metric ${tone} ${wide ? "wide" : ""}`}>
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function PathLine({ label, value }: { label: string; value?: string }) {
  return (
    <div className="pathLine">
      <span>{label}</span>
      <strong>{compactPath(value)}</strong>
    </div>
  );
}

function labelForField(field: keyof TrainingConfig) {
  const labels: Partial<Record<keyof TrainingConfig, string>> = {
    contextLength: "Context",
    microBatchSize: "Micro batch",
    gradientAccumulationSteps: "Grad accum",
    loraRank: "LoRA rank",
    learningRate: "LR",
    maxSteps: "Max steps",
    maxTrainRecords: "Train rows",
    maxEvalRecords: "Eval rows",
    maxTestRecords: "Test rows",
    maxVramGb: "Max VRAM",
    freeVramGb: "Free VRAM",
  };
  return labels[field] ?? field;
}

function toneForStatus(status?: string) {
  if (!status || status === "none") return "neutral";
  if (["ready", "completed", "dry_run_completed"].includes(status)) return "good";
  if (["warning", "running", "planned", "stopping"].includes(status)) return "warn";
  return "bad";
}

function toneForFit(fit?: string) {
  if (fit === "safe") return "good";
  if (fit === "tight") return "warn";
  if (fit === "unsafe") return "bad";
  return "neutral";
}

function metricValue(run: RunRecord | null, key: string, suffix = "") {
  const value = run?.metrics.latest[key];
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    return `${Math.abs(value) > 100 ? value.toFixed(1) : value.toPrecision(4)}${suffix}`;
  }
  return `${value}${suffix}`;
}
