#!/usr/bin/env python3
"""training_monitor.py - a readable live view of a training run for a human watching a terminal window.

WHAT IT DOES, IN PLAIN ENGLISH
  Follows a training log file while the trainer is still writing it, and redraws a small dashboard once a second:
  a progress bar, the loss now and its trend, how fast training is going, roughly how long is left, and how much
  GPU memory is in use. It only READS the log and calls nvidia-smi; it never touches the training process, so
  closing this window does not stop training.

USE
  python scripts/training_monitor.py --log <runner.log> [--err <error log>] [--title "Gemma 4 12B QLoRA"] [--epochs 2]
  python scripts/training_monitor.py --log <runner.log> --once        # draw one frame and exit (used by the tests)

WHAT IT UNDERSTANDS IN THE LOG (lines the ReTrain runners print)
  dataset: {... 'train': 501 ...}         -> how many examples per epoch (needed for speed)
  Loading weights: 47%                    -> model loading progress
  {'loss': '0.21', 'grad_norm': ..., 'learning_rate': ..., 'epoch': '0.4'}   -> one training report
  {'eval_loss': '0.09', ...}              -> a held-out evaluation
  peak VRAM allocated: 9.57 GiB / adapter saved to <path> / Traceback         -> the ending
Only the standard library is used, so it runs in any Python.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------
# SECTION 1: colours and drawing helpers (plain ANSI escape codes; Windows Terminal and modern conhost understand them)
# ------------------------------------------------------------------------------------------------------------------
RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
CYAN, GREEN, YELLOW, RED, GREY, WHITE = "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[90m", "\x1b[97m"
SPARK = "▁▂▃▄▅▆▇█"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def enable_terminal_features() -> None:
    """turn on colour handling and UTF-8 output on Windows consoles; harmless elsewhere"""
    if os.name == "nt":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            kernel.GetConsoleMode(handle, ctypes.byref(mode))
            kernel.SetConsoleMode(handle, mode.value | 0x0004)      # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def bar(fraction: float, width: int, colour: str = GREEN) -> str:
    """a solid progress bar, e.g. ████████░░░░"""
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return f"{colour}{'█' * filled}{GREY}{'░' * (width - filled)}{RESET}"


def sparkline(values: list[float], width: int) -> str:
    """a tiny trend chart of the most recent values (low = short bar)"""
    values = values[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / span * 7.999))] for v in values)


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


# ------------------------------------------------------------------------------------------------------------------
# SECTION 2: reading the log - the shared parser lives in backend/run_telemetry.py (the ReTrain console uses it too)
# ------------------------------------------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.run_telemetry import RunState, read_new   # noqa: E402


def gpu_stats() -> dict | None:
    """memory, load and temperature from nvidia-smi; None when it is not available"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout.strip().splitlines()[0]
        used, total, util, temp, power = [x.strip() for x in out.split(",")]
        return {"used": float(used) / 1024, "total": float(total) / 1024, "util": float(util), "temp": temp, "power": power}
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------------------
# SECTION 3: drawing one frame
# ------------------------------------------------------------------------------------------------------------------
def render(state: RunState, title: str, width: int, now: float, gpu: dict | None) -> str:
    width = max(64, min(width, 110))
    inner = width - 4
    line = f"{GREY}{'─' * (width - 2)}{RESET}"
    total = state.total_epochs
    frac = state.epoch / total if total else 0.0
    if state.finished:
        frac = 1.0
    out = []
    # header: what is running, current time
    out.append(f" {BOLD}{CYAN}{title}{RESET}{' ' * max(1, inner - len(title) - 8)}{GREY}{time.strftime('%H:%M:%S')}{RESET}")
    out.append(" " + line)

    # status line
    if state.error:
        status = f"{RED}{BOLD}✖ FAILED{RESET}  {state.error}"
    elif state.finished:
        status = f"{GREEN}{BOLD}✔ DONE{RESET}  adapter saved"
    elif not state.reports:
        status = f"{YELLOW}Loading the model… {state.loading}%{RESET}" if state.loading < 100 else f"{YELLOW}Starting the first step…{RESET}"
    else:
        status = f"{GREEN}● Training{RESET}"
    out.append(f" {BOLD}Status{RESET}    {status}")

    # progress
    if total:
        label = f"{frac * 100:5.1f}%   epoch {min(state.epoch, total):.2f} / {total:g}"
    else:
        label = f"epoch {state.epoch:.2f}"
    out.append(f" {BOLD}Progress{RESET}  {bar(frac, max(20, inner - len(label) - 12))}  {label}")

    # speed and time
    eps = state.epochs_per_second(now)
    rate = f"{eps * state.train_examples:.2f} examples/s" if eps and state.train_examples else "measuring…"
    left = f"~{clock((total - state.epoch) / eps)}" if eps and total and not state.finished else ("—" if state.finished else "estimating…")
    elapsed = state.runtime if state.finished and state.runtime else now - state.start_time
    out.append(f" {BOLD}Time{RESET}      elapsed {WHITE}{clock(elapsed)}{RESET}   left {WHITE}{left}{RESET}   speed {WHITE}{rate}{RESET}")

    # loss
    if state.reports:
        losses = [r["loss"] for r in state.reports]
        best, cur = min(losses), losses[-1]
        colour = GREEN if cur <= best * 1.25 else YELLOW
        trend = sparkline(losses, max(12, inner - 46))
        out.append(f" {BOLD}Loss{RESET}      {colour}{cur:.4f}{RESET}   best {WHITE}{best:.4f}{RESET}   {CYAN}{trend}{RESET}")
        last = state.reports[-1]
        out.append(f" {BOLD}Optimiser{RESET} grad-norm {WHITE}{last.get('grad_norm', 0):.2f}{RESET}   learning-rate {WHITE}{last.get('learning_rate', 0):.2e}{RESET}")
    if state.evals:
        e = state.evals[-1][1]
        out.append(f" {BOLD}Held-out{RESET}  eval loss {WHITE}{e:.4f}{RESET}  (perplexity {math.exp(e) if e < 20 else float('inf'):.2f})   checks so far: {len(state.evals)}")

    if state.checks:
        out.append(f" {BOLD}Learning{RESET}  {DIM}{'check':<11}{'valid JSON':>11}{'action ok':>11}{'exact':>8}{'eval loss':>11}{RESET}")
        for c in state.checks[-4:]:
            # run_gemma4_qlora.py already reports these as percentages (0-100), and eval_loss may be null when there is no held-out set
            eval_loss = c.get("eval_loss")
            out.append(f"           {WHITE}{str(c.get('tag', '?')):<11}{RESET}{c.get('valid_json', 0):>10.0f}%{c.get('actions_match', 0):>10.0f}%"
                       f"{c.get('exact', 0):>7.0f}%{(eval_loss if eval_loss is not None else float('nan')):>11.4f}")

    # GPU
    if gpu:
        gfrac = gpu["used"] / gpu["total"] if gpu["total"] else 0
        gcol = RED if gfrac > 0.97 else YELLOW if gfrac > 0.85 else GREEN
        out.append(f" {BOLD}GPU{RESET}       {bar(gfrac, 22, gcol)}  {gpu['used']:.1f} / {gpu['total']:.1f} GB   load {gpu['util']:.0f}%   {gpu['temp']}°C   {gpu['power']} W")
    if state.peak_vram:
        out.append(f" {BOLD}Peak VRAM{RESET} {state.peak_vram}")
    if state.saved_to:
        out.append(f" {BOLD}Saved to{RESET}  {GREEN}{state.saved_to}{RESET}")

    # recent non-report lines (warnings, saves) so surprises are visible
    out.append(" " + line)
    out.append(f" {DIM}Recent messages{RESET}")
    for msg in state.raw_tail[-4:] or ["(nothing yet)"]:
        out.append(f" {GREY}{msg[:inner]}{RESET}")
    out.append(f" {DIM}Closing this window does NOT stop training.{RESET}")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------
# SECTION 4: follow the log and keep redrawing
# ------------------------------------------------------------------------------------------------------------------
def parse_started(text: str) -> float:
    """accept unix seconds ("1790729815") or a time of day today ("20:56:55"); empty/invalid -> 0"""
    text = text.strip()
    if not text:
        return 0.0
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not m:
        return 0.0
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, int(m[1]), int(m[2]), int(m[3] or 0), 0, 0, -1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Readable live view of a training log")
    ap.add_argument("--log", required=True, help="the trainer's output log (followed live)")
    ap.add_argument("--err", default="", help="optional error log; a Traceback there marks the run as failed")
    ap.add_argument("--title", default="ReTrain training")
    ap.add_argument("--epochs", type=float, default=0.0, help="total epochs, for the progress bar")
    ap.add_argument("--started", default="", help="when the run began: unix seconds or HH:MM:SS today (default: the log file's creation time, "
                                                  "which is wrong if an older log was overwritten)")
    ap.add_argument("--once", action="store_true", help="draw a single frame and exit")
    a = ap.parse_args()

    enable_terminal_features()
    born = parse_started(a.started) or (os.path.getctime(a.log) if os.path.exists(a.log) else time.time())
    state = RunState(a.epochs, start_time=born)
    offsets = {a.log: 0, a.err: 0} if a.err else {a.log: 0}
    first_pass = True
    if not a.once:
        sys.stdout.write("\x1b[?25l\x1b[2J")                 # hide the cursor, clear the screen once
    try:
        while True:
            for path in list(offsets):
                lines, offsets[path] = read_new(path, offsets[path])
                for ln in lines:
                    # lines that existed before we started are old news: give them the log's start time so speed is not distorted
                    state.feed(ln, now=born if first_pass else None)
            first_pass = False
            frame = render(state, a.title, shutil.get_terminal_size((100, 30)).columns, time.time(), None if a.once else gpu_stats())
            if a.once:
                print(frame)
                return 0
            sys.stdout.write("\x1b[H" + "\n".join(l + "\x1b[K" for l in frame.split("\n")) + "\x1b[J")
            sys.stdout.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        if not a.once:
            sys.stdout.write("\x1b[?25h\n")


if __name__ == "__main__":
    sys.exit(main())
