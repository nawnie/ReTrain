import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import retrainIconUrl from "./assets/retrain-icon.jpg";

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
  targetType: string;
  sizeB: number;
  runnerKey: string;
  hfId: string;
  exists: boolean;
  source: string;
  notes: string[];
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
  targetType: string;
  datasetFormat: string;
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
  const isWarRoomRoute =
    typeof window !== "undefined" && new URLSearchParams(window.location.search).get("view") === "warroom";
  const [viewMode, setViewMode] = useState<"dashboard" | "warroom">(() => (isWarRoomRoute ? "warroom" : "dashboard"));
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

  const showDashboard = () => {
    setViewMode("dashboard");
    if (isWarRoomRoute) {
      const url = new URL(window.location.href);
      url.searchParams.delete("view");
      window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
    }
  };

  const showWarRoom = () => {
    setViewMode("warroom");
  };

  const openWarRoomPopout = () => {
    const url = new URL(window.location.href);
    url.searchParams.set("view", "warroom");
    const popup = window.open(url.toString(), "retrain-war-room", "width=1440,height=920,left=80,top=80");
    if (!popup) {
      setError("The browser blocked the War Room pop-out.");
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
            targetType: model?.targetType ?? current.targetType,
          }
        : current,
    );
  };

  const selectedDataset = overview?.datasets.find((item) => item.id === config?.datasetId);
  const selectedModel = overview?.models.find((item) => item.id === config?.modelId);
  const readyGates = plan?.gates.filter((item) => item.state === "ready").length ?? 0;
  const blockedGates = plan?.gates.filter((item) => item.state === "blocked").length ?? 0;
  const warningGates = plan?.gates.filter((item) => item.state === "warning").length ?? 0;
  const totalGates = plan?.gates.length ?? 0;
  const readyPercent = totalGates ? Math.round((readyGates / totalGates) * 100) : 0;
  const fitPercent = Math.min(100, Math.max(0, plan?.estimate.percent ?? 0));
  const dependencyReadyCount = overview?.dependencies.filter((item) => item.available).length ?? 0;
  const latestRunTime = selectedRun ? new Date(selectedRun.updatedAt).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : "No runs yet";
  const nextActionTitle = plan?.startEnabled
    ? config?.dryRun
      ? "Ready for a dry check"
      : "Ready for confirmed training"
    : blockedGates
      ? "Resolve blocked gates"
      : warningGates
        ? "Review warnings before launch"
        : "Plan the run";
  const nextActionCopy = plan?.startEnabled
    ? "The selected model, dataset, target, and VRAM budget are in a launchable state."
    : "The dashboard will keep planning as you adjust model, dataset, target type, and budget controls.";

  if (viewMode === "warroom") {
    return <LlmWarRoom onExit={showDashboard} onPopOut={openWarRoomPopout} isPopout={isWarRoomRoute} />;
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brandLockup">
          <img src={retrainIconUrl} alt="" className="appIcon" />
          <div>
            <p className="eyebrow">Rnv1 ReTrain</p>
            <h1>Lab Console</h1>
            <p className="lede">Local training workbench for small chat models, prompt helpers, T5 variants, and text encoders.</p>
          </div>
        </div>
        <nav className="modeTabs" aria-label="Dashboard views">
          <button type="button" className="modeTab active" onClick={showDashboard}>
            Plan
          </button>
          <button type="button" className="modeTab">
            Run
          </button>
          <button type="button" className="modeTab">
            Metrics
          </button>
          <button type="button" className="modeTab">
            Artifacts
          </button>
          <button type="button" className="modeTab monitor" onClick={showWarRoom}>
            War Room
          </button>
          <button type="button" className="modeTab popout" onClick={openWarRoomPopout}>
            Pop out
          </button>
        </nav>
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

      <section className="decisionBand" aria-label="Training decision summary">
        <div className="nextActionPanel">
          <div>
            <p className="eyebrow">Next action</p>
            <h2>{nextActionTitle}</h2>
            <p>{nextActionCopy}</p>
          </div>
          <div className="buttonRow">
            <button type="button" onClick={planRun} disabled={busy || !config}>
              Plan
            </button>
            <button className="primaryButton" type="button" onClick={startRun} disabled={!canStartRun} title={startBlockedReason}>
              {config?.dryRun ? "Run Dry Check" : "Start Confirmed Run"}
            </button>
          </div>
        </div>
        <div className="readinessPanel">
          <div className="readinessDial" style={{ "--ready": `${readyPercent}%` } as CSSProperties}>
            <strong>{readyPercent}%</strong>
            <span>ready</span>
          </div>
          <div className="readinessStack">
            <span>{readyGates} gates ready</span>
            <span>{warningGates} warnings</span>
            <span>{blockedGates} blocked</span>
          </div>
        </div>
        <div className="fitPanel">
          <div>
            <span>VRAM fit</span>
            <strong>{plan?.estimate.fitState ?? "unplanned"}</strong>
          </div>
          <div className="vramTrack large">
            <span style={{ width: `${fitPercent}%` }} />
          </div>
          <small>{plan ? `${plan.estimate.estimatedGb} GB estimated / ${plan.estimate.limitGb} GB limit` : "Plan a run to estimate VRAM."}</small>
        </div>
      </section>

      <section className="metricStrip" aria-label="Training status">
        <Metric label="App" value={health?.ok ? "Online" : "Checking"} tone={health?.ok ? "good" : "warn"} />
        <Metric label="Active runs" value={activeRuns} tone={activeRuns ? "warn" : "neutral"} />
        <Metric label="Latest status" value={selectedRun?.status ?? "none"} tone={toneForStatus(selectedRun?.status)} />
        <Metric label="Fit" value={plan?.estimate.fitState ?? "unplanned"} tone={toneForFit(plan?.estimate.fitState)} />
        <Metric label="Dependencies" value={overview ? `${dependencyReadyCount}/${overview.dependencies.length}` : "-"} tone="neutral" />
        <Metric label="Scalar tags" value={scalarCount} tone={scalarCount ? "good" : "neutral"} />
        <Metric label="Latest update" value={latestRunTime} tone="neutral" wide />
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
                <span>Target</span>
                <select value={config.targetType} onChange={(event) => updateConfig("targetType", event.target.value)}>
                  <option value="causal_lm">Chat LLM</option>
                  <option value="prompt_helper_lm">Prompt helper LM</option>
                  <option value="seq2seq_t5">T5 / Seq2Seq</option>
                  <option value="text_encoder">Text encoder</option>
                  <option value="clip_text">CLIP text</option>
                  <option value="blip_text">BLIP text</option>
                </select>
              </label>
              <label className="field">
                <span>Data format</span>
                <select value={config.datasetFormat} onChange={(event) => updateConfig("datasetFormat", event.target.value)}>
                  <option value="auto">Auto</option>
                  <option value="messages">Messages</option>
                  <option value="prompt_completion">Prompt/completion</option>
                  <option value="text">Text</option>
                  <option value="text_pairs">Text pairs</option>
                  <option value="caption_pairs">Caption pairs</option>
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

              <details className="advancedControls">
                <summary>
                  <span>Budget and limits</span>
                  <strong>Context, VRAM, rows</strong>
                </summary>
                <div className="controlGrid nested">
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
                </div>
              </details>

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

          <div className="buttonRow panelActions">
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
                  <span>Target</span>
                  <strong>{targetLabel(config?.targetType ?? selectedModel?.targetType)}</strong>
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

              <details className="commandBox">
                <summary>
                  <span>Command preview</span>
                  <strong>Advanced</strong>
                </summary>
                <code>{plan?.command ? scrubLocalPaths(plan.command) : "No plan yet"}</code>
              </details>
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
            <details className="logDrawer">
              <summary>
                <span>Log tail</span>
                <strong>Advanced</strong>
              </summary>
              <pre className="logBox">{selectedRun?.logTail ? scrubLocalPaths(selectedRun.logTail) : "No log output yet"}</pre>
            </details>
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

function targetLabel(targetType?: string) {
  const labels: Record<string, string> = {
    causal_lm: "Chat LLM",
    prompt_helper_lm: "Prompt helper",
    seq2seq_t5: "T5 / Seq2Seq",
    text_encoder: "Text encoder",
    clip_text: "CLIP text",
    blip_text: "BLIP text",
  };
  return labels[targetType ?? ""] ?? targetType ?? "-";
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

function LlmWarRoom({
  onExit,
  onPopOut,
  isPopout = false,
}: {
  onExit: () => void;
  onPopOut: () => void;
  isPopout?: boolean;
}) {
  const heatmapCells = useMemo(
    () =>
      Array.from({ length: 128 }, (_, index) => {
        const classes = ["bg-[#00ff66]", "bg-[#00ff66]/60", "bg-[#00ff66]/20", "bg-[#0088ff]", "bg-[#142820]"];
        return classes[(index * 7 + Math.floor(index / 9)) % classes.length];
      }),
    [],
  );

  return (
    <main className="min-h-screen w-screen bg-[#05070a] text-[#00ff66] font-mono text-xs select-none p-3 flex flex-col tracking-wider overflow-x-hidden">
      <header className="border border-[#142820] bg-[#07130e] p-3 flex flex-col xl:flex-row justify-between xl:items-center mb-3 shadow-[0_0_15px_rgba(0,255,102,0.03)] gap-3">
        <div className="flex flex-wrap items-center gap-3 min-w-0">
          <span className="text-white font-black text-sm tracking-widest animate-pulse whitespace-nowrap">
            HYPERION // CORE_TRN
          </span>
          <span className="text-[#0088ff] border border-[#004488] px-2 py-0.5 bg-[#001122] whitespace-nowrap">
            RUN_ID: L3_8B_BASE_04
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-3 xl:gap-6 text-[#558866]">
          <div>
            EPOCH: <strong>3 / 10</strong>
          </div>
          <div>
            TOKENS: <strong>842.11 B</strong>
          </div>
          <div>
            CLUSTER LOAD: <strong className="text-[#ff3366] font-bold">91.4%</strong>
          </div>
          <div className="text-white bg-[#142820] px-2 py-0.5 border border-[#224433] animate-pulse">LIVE_FEED</div>
          {!isPopout ? (
            <button
              type="button"
              className="min-h-0 border border-[#004488] bg-[#001122] px-3 py-1 text-[#0088ff] hover:bg-[#061b30]"
              onClick={onPopOut}
            >
              Pop Out
            </button>
          ) : null}
          <button
            type="button"
            className="min-h-0 border border-[#224433] bg-[#030705] px-3 py-1 text-[#00ff66] hover:bg-[#07130e]"
            onClick={onExit}
          >
            Lab Console
          </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 xl:grid-cols-12 gap-3 min-h-0">
        <div className="xl:col-span-4 flex flex-col gap-3 min-w-0">
          <TailwindSection title="SYSTEM_TELEMETRY">
            <div className="grid grid-cols-2 gap-2 text-slate-400">
              <WarStat label="THROUGHPUT" value="142,510 t/s" color="text-[#00ff66]" />
              <WarStat label="LEARNING RATE" value="1.842e-5" />
              <WarStat label="GRADIENT NORM" value="0.2144" />
              <WarStat label="TENSOR PARALLEL" value="8-WAY" color="text-[#0088ff]" />
            </div>
            <div className="mt-4 pt-3 border-t border-[#142820]">
              <div className="flex justify-between mb-1 text-[#558866]">
                <span>DATASET PROGRESS // FineWeb-EDU</span>
                <span>84.21%</span>
              </div>
              <div className="w-full bg-[#0a1410] border border-[#142820] h-3 p-0.5">
                <div
                  className="bg-gradient-to-r from-[#00ff66] to-[#0088ff] h-full shadow-[0_0_8px_rgba(0,255,102,0.5)]"
                  style={{ width: "84.21%" }}
                />
              </div>
            </div>
          </TailwindSection>

          <TailwindSection title="H100_INTERCONNECT_TOPOLOGY (8-NODE CLUSTER)">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <WarGpuNode id="00" load={98} temp={74} state="CRIT" />
              <WarGpuNode id="01" load={96} temp={72} state="OK" />
              <WarGpuNode id="02" load={95} temp={71} state="OK" />
              <WarGpuNode id="03" load={95} temp={75} state="OK" />
              <WarGpuNode id="04" load={97} temp={73} state="OK" />
              <WarGpuNode id="05" load={96} temp={70} state="OK" />
              <WarGpuNode id="06" load={14} temp={42} state="WARN" />
              <WarGpuNode id="07" load={94} temp={72} state="OK" />
            </div>
            <div className="mt-2 text-xs text-[#558866] flex flex-wrap justify-between gap-3">
              <span>InfiniBand Quantum-2 Rate: 400Gb/s</span>
              <span className="text-[#ff3366] animate-pulse">LINK_06 DEGRADED</span>
            </div>
          </TailwindSection>
        </div>

        <div className="xl:col-span-8 flex flex-col gap-3 min-w-0">
          <section className="min-h-[360px] xl:flex-1 border border-[#142820] bg-[#030705] p-3 flex flex-col relative">
            <div className="static xl:absolute xl:top-2 xl:right-3 text-[#558866] text-[10px] mb-2 xl:mb-0">
              VISUALIZER // LOSS_STABILITY
            </div>
            <h2 className="text-white border-b border-[#142820] pb-1 mb-2 font-bold flex items-center text-xs">
              <span className="inline-block w-2 h-2 bg-[#00ff66] mr-2" />
              REAL-TIME LOSS MATRIX TRACKER
            </h2>
            <div className="flex-1 flex flex-col justify-end gap-1 font-mono text-[10px] text-[#00ff66]/70">
              <p>2.40 | *</p>
              <p>2.00 |   *   *</p>
              <p>1.60 |         *</p>
              <p>1.30 |           *   *   .</p>
              <p>1.10 |                   *   .   .</p>
              <p>0.95 |                           * * * _ _ _ _</p>
              <div className="border-t border-[#142820] pt-1 grid grid-cols-4 gap-2 text-[#558866]">
                <span>0k STEPS</span>
                <span>150k STEPS</span>
                <span>300k STEPS</span>
                <span>450k STEPS</span>
              </div>
            </div>
          </section>

          <section className="min-h-44 border border-[#142820] bg-[#030705] p-3 flex flex-col">
            <h2 className="text-white border-b border-[#142820] pb-1 mb-2 font-bold text-xs">
              ATTENTION HEAD ACTIVATION WEIGHTS (LIVE SAMPLING)
            </h2>
            <div className="grid grid-cols-32 gap-1 flex-1 items-center">
              {heatmapCells.map((cell, index) => (
                <div className={`h-3 w-full ${cell} rounded-sm`} key={index} />
              ))}
            </div>
          </section>

          <section className="h-40 border border-[#142820] bg-[#020403] p-3 font-mono text-[11px] text-[#00ff66] overflow-y-auto space-y-1">
            <p className="text-[#558866]">[00:14:02] INITIALIZING BACKWARD PASS // ALL-REDUCE GRADIENT COLLECTIVE</p>
            <p>[00:14:03] STEP 412,901 | LOSS: 0.9412 | PERPLEXITY: 2.563 | GRAD_NORM: 0.211</p>
            <p>[00:14:04] STEP 412,902 | LOSS: 0.9398 | PERPLEXITY: 2.559 | GRAD_NORM: 0.214</p>
            <p className="text-[#ff3366]">[00:14:05] NCCL WARN: Node-06 interconnect speed dropped below 200Gb/s; retrying lane...</p>
            <p className="text-[#0088ff]">[00:14:06] CHECKPOINT TRIGGER: serializing tensor state to master node target block...</p>
          </section>
        </div>
      </div>
    </main>
  );
}

function TailwindSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border border-[#142820] bg-[#030705] p-3 flex flex-col">
      <div className="text-white font-bold tracking-widest border-b border-[#142820] pb-1.5 mb-3 flex justify-between items-center">
        <span>// {title}</span>
        <span className="text-[10px] text-[#558866]">SYS_REMOTE</span>
      </div>
      {children}
    </section>
  );
}

function WarStat({ label, value, color = "text-white" }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-[#07130e] border border-[#142820] p-2 flex flex-col justify-between">
      <span className="text-[9px] text-[#558866] block mb-1">{label}</span>
      <strong className={`text-sm font-black font-mono ${color}`}>{value}</strong>
    </div>
  );
}

function WarGpuNode({ id, load, temp, state }: { id: string; load: number; temp: number; state: "OK" | "WARN" | "CRIT" }) {
  const statusColor =
    state === "OK" ? "border-[#00ff66] text-[#00ff66]" : state === "WARN" ? "border-[#ffaa00] text-[#ffaa00]" : "border-[#ff3366] text-[#ff3366]";
  const bgColor = state === "OK" ? "bg-[#00ff66]/5" : state === "WARN" ? "bg-[#ffaa00]/5" : "bg-[#ff3366]/5";

  return (
    <div className={`border p-1.5 ${statusColor} ${bgColor} flex flex-col justify-between h-14`}>
      <div className="flex justify-between text-[9px]">
        <span>H100_{id}</span>
        <strong className="font-bold">{state}</strong>
      </div>
      <div className="text-right">
        <span className="text-white font-bold text-sm">{load}%</span>
        <span className="text-[9px] block text-[#558866]">{temp} C</span>
      </div>
    </div>
  );
}
