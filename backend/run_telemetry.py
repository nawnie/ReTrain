"""Turn a training run's log into live, chart-ready numbers for the ReTrain console (and the terminal monitor).

WHY THIS EXISTS
  Trainers print a line like  {'loss': '0.21', 'grad_norm': '1.4', 'learning_rate': '0.0002', 'epoch': '0.4'}
  every few steps. That is fine for the trainer and poor for a human. This module follows the log while it is
  still being written and keeps a small picture of the run: the loss curve, the held-out "learning check" after
  each epoch, speed, time left, and how the run ended. It only READS files; it never touches the training process.

WHO WRITES WHAT
  run log           the trainer's own output (dataset line, loss lines, 'VALIDATION {...}' lines, final lines)
  layers.jsonl      optional, one JSON object per line: {"step": 5, "epoch": 0.08, "layers": [{"i": 0, "attn": g, "mlp": g, "upd": u}, ...]}
                    written by trainers that measure per-layer gradient size and adapter change (run_gemma4_qlora.py)
  live_runs/*.json  a tiny registration file so the console can find a run that was started outside the console

Standard library only, so the terminal monitor and the API can both import it.
"""

from __future__ import annotations

import ast
import json
import os
import math
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
LIVE_RUNS_DIR = ROOT_DIR / "training" / "live_runs"

# Library chatter a person watching does not need to see.
NOISE_RE = re.compile(r"triton not found|flop counting|UserWarning|FutureWarning")


# ------------------------------------------------------------------------------------------------------------------
# SECTION 1: the picture of one run, built up from log lines
# ------------------------------------------------------------------------------------------------------------------
class RunState:
    """everything a dashboard needs to know about one run, built from its log lines"""

    def __init__(self, total_epochs: float = 0.0, start_time: float | None = None) -> None:
        self.total_epochs = total_epochs
        self.start_time = start_time or time.time()
        self.train_examples = 0
        self.loading = 0
        self.trainable = ""
        self.reports: list[dict] = []                 # every {'loss': ...} report: loss, grad_norm, learning_rate, epoch, seen_at
        self.evals: list[tuple[float, float]] = []    # (epoch, eval_loss) from the trainer's own evaluation
        self.checks: list[dict] = []                  # per-epoch "learning check" reports: VALIDATION {json}
        self.peak_vram = ""
        self.saved_to = ""
        self.runtime = 0.0
        self.error = ""
        self.raw_tail: list[str] = []

    def feed(self, line: str, now: float | None = None) -> None:
        now = now or time.time()
        line = line.strip()
        if not line:
            return
        m = re.search(r"Loading weights:\s+(\d+)%", line)
        if m:
            self.loading = int(m.group(1))
            return
        if "Traceback" in line:
            self.error = "the trainer crashed - see the log below"
        if line.startswith("VALIDATION "):
            try:
                self.checks.append(json.loads(line[len("VALIDATION "):]))
            except ValueError:
                pass
            return
        if line.startswith("dataset:"):
            t = re.search(r"'train':\s*(\d+)", line)
            self.train_examples = int(t.group(1)) if t else self.train_examples
        elif line.startswith("trainable params"):
            self.trainable = line
        elif line.startswith("{") and line.endswith("}"):
            try:
                d = {k: float(v) for k, v in ast.literal_eval(line).items()}
            except (ValueError, SyntaxError, TypeError):
                d = {}
            if "loss" in d and "epoch" in d:
                d["seen_at"] = now
                self.reports.append(d)
            elif "eval_loss" in d:
                self.evals.append((d.get("epoch", self.reports[-1]["epoch"] if self.reports else 0.0), d["eval_loss"]))
            elif "train_runtime" in d:
                self.runtime = d["train_runtime"]
        elif line.startswith("peak VRAM"):
            self.peak_vram = line.split(":", 1)[1].strip()
        elif line.startswith("adapter saved to"):
            self.saved_to = line.split("to", 1)[1].strip()
        if not line.startswith("{") and not NOISE_RE.search(line):
            self.raw_tail = (self.raw_tail + [line])[-12:]

    # -- derived numbers ------------------------------------------------------------------------------------------
    @property
    def epoch(self) -> float:
        return self.reports[-1]["epoch"] if self.reports else 0.0

    @property
    def finished(self) -> bool:
        return bool(self.saved_to)

    def epochs_per_second(self, now: float) -> float:
        """recent speed measured between reports we watched arrive; falls back to the whole run so far"""
        seen = [r for r in self.reports if r["seen_at"] > self.start_time + 1]
        recent = seen[-8:]
        if len(recent) >= 2 and recent[-1]["seen_at"] - recent[0]["seen_at"] > 5:
            return (recent[-1]["epoch"] - recent[0]["epoch"]) / (recent[-1]["seen_at"] - recent[0]["seen_at"])
        busy = now - self.start_time - 45          # ~45 s is spent loading the model before the first step
        return self.epoch / busy if self.epoch > 0 and busy > 5 else 0.0

    def snapshot(self, now: float | None = None, alive: bool | None = None) -> dict[str, Any]:
        """the whole picture as plain JSON for the API"""
        now = now or time.time()
        eps = self.epochs_per_second(now)
        total = self.total_epochs
        if self.finished:
            status = "done"
        elif self.error:
            status = "failed"
        elif alive is False:
            status = "stopped"                     # the process ended without saving an adapter
        elif not self.reports:
            status = "loading"
        else:
            status = "training"
        points = [[r["epoch"], r["loss"], r.get("grad_norm"), r.get("learning_rate"), r["seen_at"]] for r in self.reports]
        if len(points) > 600:                      # keep the payload small on long runs
            step = len(points) / 600
            points = [points[int(i * step)] for i in range(600)] + [points[-1]]
        losses = [r["loss"] for r in self.reports]
        return {
            "status": status,
            "loadingPercent": self.loading,
            "epoch": self.epoch,
            "totalEpochs": total,
            "progress": 1.0 if self.finished else (min(1.0, self.epoch / total) if total else 0.0),
            "elapsedSeconds": self.runtime if self.finished and self.runtime else max(0.0, now - self.start_time),
            "etaSeconds": ((total - self.epoch) / eps) if eps and total and not self.finished else None,
            "examplesPerSecond": eps * self.train_examples if eps and self.train_examples else None,
            "trainExamples": self.train_examples,
            "trainable": self.trainable,
            "loss": points,                        # [epoch, loss, grad_norm, learning_rate]
            "lossNow": losses[-1] if losses else None,
            "lossBest": min(losses) if losses else None,
            "evals": [[e, v] for e, v in self.evals],
            "checks": self.checks,
            "peakVram": self.peak_vram,
            "savedTo": self.saved_to,
            "error": self.error,
            "messages": self.raw_tail[-8:],
        }


def read_new(path: str, offset: int) -> tuple[list[str], int]:
    """complete new lines since `offset` (carriage-return progress bars count as lines); returns the new offset"""
    if not os.path.exists(path):
        return [], offset
    if os.path.getsize(path) < offset:             # the file was replaced by a newer run
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    cut = max(text.rfind("\n"), text.rfind("\r"))
    if cut < 0:
        return [], offset
    return re.split(r"[\r\n]+", text[:cut + 1]), offset + len(text[:cut + 1].encode("utf-8"))


def process_alive(pid: int | None) -> bool | None:
    """True/False for a known pid on Windows; None when we cannot tell"""
    if not pid or os.name != "nt":
        return None
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, int(pid))         # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259                                      # STILL_ACTIVE
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------------------
# SECTION 2: find runs, and keep one incremental reader per run so a poll costs only the newly written bytes
# ------------------------------------------------------------------------------------------------------------------
def register_run(run_id: str, *, title: str, log: str, epochs: float, started: float | None = None, pid: int | None = None,
                 layers: str = "", err: str = "", extra: dict[str, Any] | None = None) -> Path:
    """write the small file that makes a run visible in the console (any trainer can call this)"""
    LIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_RUNS_DIR / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', run_id)}.json"
    record = {"id": run_id, "title": title, "log": log, "err": err, "layers": layers, "epochs": epochs,
              "started": started or time.time(), "pid": pid, **(extra or {})}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def list_registered() -> list[dict[str, Any]]:
    runs = []
    for p in sorted(LIVE_RUNS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return runs


class Tracker:
    """keeps a RunState and a file offset per run; safe to call from several request threads"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[RunState, dict[str, int], str]] = {}

    def state(self, rec: dict[str, Any]) -> RunState:
        with self._lock:
            key = rec["id"]
            cached = self._runs.get(key)
            if cached is None or cached[2] != rec.get("log"):
                st = RunState(float(rec.get("epochs") or 0), start_time=float(rec.get("started") or time.time()))
                cached = (st, {rec.get("log", ""): 0, rec.get("err", ""): 0}, rec.get("log", ""))
                self._runs[key] = cached
            st, offsets, _ = cached
            for path in [p for p in offsets if p]:
                before = len(st.reports)
                previous = st.reports[-1]["seen_at"] if st.reports else st.start_time + 45      # ~45 s is spent loading the model
                lines, offsets[path] = read_new(path, offsets[path])
                for ln in lines:
                    st.feed(ln)
                fresh = st.reports[before:]
                if len(fresh) > 1 and os.path.exists(path):
                    # several reports appeared between two looks, so their true times are unknown: space them evenly up to the
                    # moment the log was last written (approximate, and only for readings we did not see arrive)
                    end = max(previous + len(fresh), os.path.getmtime(path))
                    for k, r in enumerate(fresh):
                        r["seen_at"] = previous + (end - previous) * (k + 1) / len(fresh)
            return st

    def snapshot(self, rec: dict[str, Any]) -> dict[str, Any]:
        st = self.state(rec)
        snap = st.snapshot(alive=process_alive(rec.get("pid")))
        snap.update({"id": rec["id"], "title": rec.get("title", rec["id"]), "log": rec.get("log", ""), "pid": rec.get("pid"),
                     "hasLayers": bool(rec.get("layers")) and os.path.exists(rec["layers"]), "demo": bool(rec.get("demo")),
                     "startedAt": rec.get("started")})
        return snap


TRACKER = Tracker()


def layer_stats(rec: dict[str, Any], max_history: int = 400) -> dict[str, Any]:
    """per-layer training signal: the latest reading, plus a running total per layer since the start"""
    path = rec.get("layers") or ""
    if not path or not os.path.exists(path):
        return {"available": False, "reason": "This run did not record per-layer statistics."}
    latest: dict[str, Any] | None = None
    totals: dict[int, dict[str, float]] = {}
    history: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            latest = row
            for layer in row.get("layers", []):
                t = totals.setdefault(int(layer["i"]), {"attn": 0.0, "mlp": 0.0})
                t["attn"] += float(layer.get("attn", 0.0)); t["mlp"] += float(layer.get("mlp", 0.0))
            history.append({"step": row.get("step"), "epoch": row.get("epoch"),
                            "total": sum(float(l.get("attn", 0)) + float(l.get("mlp", 0)) for l in row.get("layers", []))})
    if latest is None:
        return {"available": False, "reason": "No layer readings written yet."}
    step = max(1, len(history) // max_history)
    return {"available": True, "latest": latest, "totals": [{"i": i, **v} for i, v in sorted(totals.items())],
            "history": history[::step], "readings": len(history), "matrix": layer_matrix(path)}


def layer_matrix(path: str, max_rows: int = 240) -> dict[str, Any]:
    """steps x layers grids (gradient size, adapter size) for a heatmap of WHERE the learning happens, and WHEN"""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if not rows:
        return {"steps": [], "epochs": [], "layers": [], "grad": [], "upd": []}
    stride = max(1, math.ceil(len(rows) / max_rows))
    rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
    layers = sorted({int(l["i"]) for r in rows for l in r.get("layers", [])})

    def grid(key: str) -> list[list[float]]:
        out = []
        for r in rows:
            by = {int(l["i"]): l for l in r.get("layers", [])}
            line = []
            for i in layers:
                if i not in by:
                    line.append(0.0)
                elif key == "grad":
                    line.append(round(float(by[i].get("attn", 0)) + float(by[i].get("mlp", 0)), 5))
                else:
                    line.append(round(float(by[i].get("upd", 0)), 5))
            out.append(line)
        return out

    return {"steps": [r.get("step") for r in rows], "epochs": [r.get("epoch") for r in rows], "layers": layers, "grad": grid("grad"), "upd": grid("upd")}


def validation_samples(rec: dict[str, Any]) -> dict[str, Any]:
    """the held-out questions with what the model answered at every learning check, so answers can be compared side by side"""
    layers = rec.get("layers") or ""
    folder = Path(layers).parent / "validation" if layers else None
    if not folder or not folder.exists():
        return {"available": False, "tags": [], "items": []}

    def order(p: Path) -> tuple[int, float, str]:
        name = p.stem
        m = re.search(r"(\d+)", name)
        return (0 if name.startswith("baseline") else 2 if name.startswith("final") else 1, float(m.group(1)) if m else 0.0, name)

    tags: list[str] = []
    items: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.jsonl"), key=order):
        tag = path.stem.replace("-", " ")
        tags.append(tag)
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            while len(items) <= n:
                items.append({"i": len(items), "category": "", "prompt": "", "expected": "", "answers": {}})
            it = items[n]
            it["category"] = row.get("category") or it["category"]
            it["prompt"] = row.get("prompt") or it["prompt"]
            it["expected"] = row.get("expected") or it["expected"]
            it["answers"][tag] = {k: row.get(k) for k in ("got", "valid_json", "actions_match", "first_button_match", "exact")}
    return {"available": True, "tags": tags, "items": items}


# ------------------------------------------------------------------------------------------------------------------
# SECTION 3: GPU history. A background thread keeps the last ~30 minutes so the console can draw a timeline even when it was
# opened late. It only calls nvidia-smi; it never touches the training process.
# ------------------------------------------------------------------------------------------------------------------
class GpuSampler:
    def __init__(self, seconds: float = 2.0, keep: int = 900) -> None:
        self.seconds, self.samples = seconds, deque(maxlen=keep)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        exe = shutil.which("nvidia-smi")
        while exe:
            try:
                out = subprocess.run([exe, "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                used, total, util, temp, power, clock = [float(x) for x in out.split(",")]
                self.samples.append({"t": time.time(), "usedGb": round(used / 1024, 2), "totalGb": round(total / 1024, 2), "util": util,
                                     "tempC": temp, "powerW": power, "clockMhz": clock})
            except Exception:
                pass
            time.sleep(self.seconds)


GPU = GpuSampler()
