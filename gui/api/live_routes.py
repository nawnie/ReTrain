"""API routes for watching a training run live in the ReTrain console.

  GET /api/retrain/live/runs                      every run the console can watch (newest first), each with a short summary
  GET /api/retrain/live/run?id=<run id>           the full picture: progress, loss curve, learning checks, speed, messages
  GET /api/retrain/live/layers?id=<run id>        per-layer training signal, when the trainer recorded it

The heavy lifting (reading the log, keeping a running picture) is in backend/run_telemetry.py. These routes only
read files; nothing here starts, stops or changes a training run.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from backend import run_telemetry as telemetry


def _find(run_id: str) -> dict[str, Any] | None:
    """the registered run with this id (or the newest one when no id is given)"""
    runs = telemetry.list_registered()
    if not run_id:
        return runs[0] if runs else None
    return next((r for r in runs if r.get("id") == run_id), None)


def register(app: FastAPI) -> None:
    telemetry.GPU.start()              # keep GPU history from the moment the API is up
    @app.get("/api/retrain/live/runs")
    def live_runs() -> dict[str, Any]:
        out = []
        for rec in telemetry.list_registered():
            snap = telemetry.TRACKER.snapshot(rec)
            out.append({"id": rec["id"], "title": rec.get("title", rec["id"]), "status": snap["status"], "progress": snap["progress"],
                        "epoch": snap["epoch"], "totalEpochs": snap["totalEpochs"], "lossNow": snap["lossNow"], "started": rec.get("started"), "demo": bool(rec.get("demo"))})
        return {"runs": out}

    @app.get("/api/retrain/live/run")
    def live_run(id: str = "") -> dict[str, Any]:
        rec = _find(id)
        if rec is None:
            return {"status": "blocked", "message": "No live runs are registered yet."}
        return telemetry.TRACKER.snapshot(rec)

    @app.get("/api/retrain/live/layers")
    def live_layers(id: str = "") -> dict[str, Any]:
        rec = _find(id)
        if rec is None:
            return {"available": False, "reason": "No live runs are registered yet."}
        return telemetry.layer_stats(rec)

    @app.get("/api/retrain/live/samples")
    def live_samples(id: str = "") -> dict[str, Any]:
        rec = _find(id)
        if rec is None:
            return {"available": False, "tags": [], "items": []}
        return telemetry.validation_samples(rec)

    @app.get("/api/retrain/live/system")
    def live_system() -> dict[str, Any]:
        return {"samples": list(telemetry.GPU.samples)}
