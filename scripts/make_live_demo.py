#!/usr/bin/env python3
"""make_live_demo.py - write a SIMULATED finished training run so the console's Live pane can be developed and demonstrated
without using the GPU.

NOTHING HERE IS A REAL RESULT. The loss curve, layer activity and model answers are generated from a fixed random seed
to look like a plausible two-epoch run. The run is registered with "demo": true, and the console shows a DEMO badge on it,
so it can never be mistaken for a measurement. Questions and reference answers are taken from a real chat dataset when one
is given, so the answer cards look like the real thing.

  python scripts/make_live_demo.py [--dataset <train.jsonl>]

Output: training/live_demo/<id>/{runner.log, layers.jsonl, validation/*.jsonl} and a registration in training/live_runs/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import run_telemetry  # noqa: E402

BUTTONS = ["A", "B", "START", "UP", "DOWN", "LEFT", "RIGHT"]


def load_questions(path: Path | None, n: int, rnd: random.Random) -> list[dict]:
    """held-out questions: real ones from a dataset when available, otherwise a few made-up ones"""
    rows = []
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row["messages"][-1]["role"] == "assistant":
                    rows.append(row)
            except (ValueError, KeyError):
                continue
    if not rows:
        rows = [{"category": "title", "messages": [{"role": "user", "content": "On screen: the title screen with PRESS START."},
                                                     {"role": "assistant", "content": json.dumps({"think": "press start", "say": "", "actions": [{"buttons": ["START"], "frames": 8}]})}]}
                for _ in range(n)]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("category", "all"), []).append(r)
    picked, i = [], 0
    while len(picked) < n:
        for group in by.values():
            if len(picked) < n:
                picked.append(group[(i * 7 + len(picked)) % len(group)])
        i += 1
    return picked


def wrong_answer(expected: str, rnd: random.Random) -> str:
    """a plausible mistake: same shape, different buttons"""
    try:
        d = json.loads(expected)
        d["actions"] = [{"buttons": [rnd.choice(BUTTONS)], "frames": rnd.choice([8, 16, 32])}]
        return json.dumps(d, separators=(",", ":"))
    except ValueError:
        return expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None, help="Optional JSONL dataset; otherwise use synthetic demo questions.")
    a = ap.parse_args()
    rnd = random.Random(20260929)
    run_id = "demo-gemma4-12b"
    root = ROOT / "training" / "live_demo" / run_id
    (root / "validation").mkdir(parents=True, exist_ok=True)
    steps_per_epoch, epochs, every = 63, 2, 5
    total_steps = steps_per_epoch * epochs
    n_layers = 48

    # ---- the training log: dataset line, loss reports, learning checks, ending ---------------------------------------
    questions = load_questions(a.dataset, 24, rnd)
    cats = sorted({q.get("category", "all") for q in questions})
    log = ["dataset: {'examples': 556, 'too_long': 0, 'images_ignored': 0, 'train': 501, 'eval': 55}",
           "trainable params: 65,568,768 || all params: 12,025,298,944 || trainable%: 0.5453"]
    skill = {"baseline": 0.16, "epoch 1": 0.74, "epoch 2": 0.90}          # how often the simulated model answers right at each check
    losses = {"baseline": 1.9, "epoch 1": 0.16, "epoch 2": 0.07}

    def check(tag: str, epoch: float, step: int) -> None:
        rows, scored = [], []
        for q in questions:
            expected = q["messages"][-1]["content"]
            good = rnd.random() < skill[tag] + (0.05 if q.get("category") in ("title", "waiting") else 0)
            got = expected if good else wrong_answer(expected, rnd)
            if tag == "baseline" and rnd.random() < 0.35:
                got = "I would press the button to continue. " + got[:20]      # an untrained model often ignores the format
            valid = got.startswith("{") and got.endswith("}")
            rows.append({"category": q.get("category"), "prompt": q["messages"][-2]["content"], "expected": expected, "got": got,
                         "valid_json": valid, "actions_match": good and valid, "first_button_match": good and valid, "exact": good and valid})
            scored.append(rows[-1])
        (root / "validation" / (tag.replace(" ", "-") + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        pct = lambda k: round(100.0 * sum(1 for r in scored if r[k]) / len(scored), 1)
        by = {c: {"n": sum(1 for r in scored if r["category"] == c), "actions_match": round(100.0 * sum(1 for r in scored if r["category"] == c and r["actions_match"]) / max(1, sum(1 for r in scored if r["category"] == c)), 1)} for c in cats}
        log.append("VALIDATION " + json.dumps({"tag": tag, "epoch": epoch, "step": step, "eval_loss": losses[tag], "eval_perplexity": round(math.exp(losses[tag]), 3),
                                              "valid_json": pct("valid_json"), "actions_match": pct("actions_match"), "first_button_match": pct("first_button_match"),
                                              "exact": pct("exact"), "n": len(scored), "seconds": 41.2, "by_category": by}))

    layer_lines = []
    check("baseline", 0.0, 0)
    for step in range(every, total_steps + 1, every):
        epoch = round(step * 8 / 501, 5)
        loss = 2.05 * math.exp(-step / 9.0) + 0.075 + rnd.gauss(0, 0.02) * (1 + 2 * math.exp(-step / 20))
        gnorm = 0.7 + 3.0 * math.exp(-step / 14.0) + rnd.gauss(0, 0.15)
        warm = min(1.0, step / (0.03 * total_steps))
        lr = 2e-4 * warm * 0.5 * (1 + math.cos(math.pi * step / total_steps))
        log.append(str({"loss": f"{max(loss, 0.03):.4g}", "grad_norm": f"{max(gnorm, 0.2):.4g}", "learning_rate": f"{lr:.4g}", "epoch": f"{epoch:.4g}"}))
        # per-layer activity: the middle-to-late layers learn most, the full-attention layers (every 6th) stand out, and everything fades as the loss falls
        layers = []
        for i in range(n_layers):
            base = 0.25 + 0.75 * math.exp(-((i - 30) / 12.0) ** 2) + (0.35 if i % 6 == 5 else 0.0)
            g = base * gnorm * (0.85 + 0.3 * rnd.random()) / 6
            layers.append({"i": i, "attn": round(g * 0.42, 5), "mlp": round(g * 0.58, 5), "upd": round(base * math.sqrt(step) * 0.06 * (0.9 + 0.2 * rnd.random()), 5)})
        layer_lines.append(json.dumps({"step": step, "epoch": epoch, "layers": layers}))
        if step == steps_per_epoch - (steps_per_epoch % every):
            check("epoch 1", 1.0, step)
    check("epoch 2", 2.0, total_steps)
    log += ["{'train_runtime': '1183.4', 'train_samples_per_second': '0.847', 'train_steps_per_second': '0.106', 'train_loss': '0.412', 'epoch': '2'}",
            "peak VRAM allocated: 9.61 GiB", f"adapter saved to {root / 'final'}"]
    (root / "runner.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    (root / "layers.jsonl").write_text("\n".join(layer_lines) + "\n", encoding="utf-8")
    run_telemetry.register_run(run_id, title="DEMO - simulated Gemma 4 12B run (not real data)", log=str(root / "runner.log"), epochs=epochs,
                               started=time.time() - 1500, pid=None, layers=str(root / "layers.jsonl"), extra={"demo": True})
    print("demo run written to", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
