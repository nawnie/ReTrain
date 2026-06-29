from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import training_runs


ROOT_DIR: Final = Path(__file__).resolve().parents[1]
FRONTEND_DIST: Final = ROOT_DIR / "frontend" / "dist"
DEFAULT_LOGDIR: Final = training_runs.RUN_OUTPUT_ROOT
CODEX_APP_DATASET_DIR: Final = ROOT_DIR / "datasets" / "codex_app_environment"
CODEX_APP_MANIFEST: Final = CODEX_APP_DATASET_DIR / "manifest.json"
TENSORBOARD_HOST: Final = "127.0.0.1"
TENSORBOARD_PORT: Final = int(
    os.environ.get("RNV1_TENSORBOARD_PORT", os.environ.get("MOK_TENSORBOARD_PORT", "6006"))
)
TENSORBOARD_BASE: Final = f"http://{TENSORBOARD_HOST}:{TENSORBOARD_PORT}"

app = FastAPI(title="Rnv1 ReTrain Dashboard", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_tensorboard_process: subprocess.Popen[bytes] | None = None
_tensorboard_lock = asyncio.Lock()


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _tensorboard_command(logdir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        TENSORBOARD_HOST,
        "--port",
        str(TENSORBOARD_PORT),
        "--path_prefix",
        "/tensorboard",
    ]


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dataset_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return CODEX_APP_DATASET_DIR / candidate


async def _ensure_tensorboard() -> tuple[bool, str | None]:
    global _tensorboard_process

    if _is_port_open(TENSORBOARD_HOST, TENSORBOARD_PORT):
        return True, None

    async with _tensorboard_lock:
        if _is_port_open(TENSORBOARD_HOST, TENSORBOARD_PORT):
            return True, None

        DEFAULT_LOGDIR.mkdir(parents=True, exist_ok=True)
        command = _tensorboard_command(DEFAULT_LOGDIR)
        try:
            _tensorboard_process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as exc:
            return False, f"Failed to start TensorBoard: {exc}"

        for _ in range(40):
            if _tensorboard_process.poll() is not None:
                return False, "TensorBoard exited during startup."
            if _is_port_open(TENSORBOARD_HOST, TENSORBOARD_PORT):
                return True, None
            await asyncio.sleep(0.25)

    return False, "TensorBoard did not become ready within 10 seconds."


@app.on_event("shutdown")
def shutdown_tensorboard() -> None:
    if _tensorboard_process and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "rnv1-retrain-dashboard",
        "tensorboard": {
            "embeddedPath": "/tensorboard/",
            "logdir": str(DEFAULT_LOGDIR),
            "port": TENSORBOARD_PORT,
            "running": _is_port_open(TENSORBOARD_HOST, TENSORBOARD_PORT),
        },
    }


@app.get("/api/training/codex-app")
async def codex_app_training_pack() -> dict[str, object]:
    issues: list[str] = []
    manifest: dict[str, Any] | None = None

    if not CODEX_APP_MANIFEST.exists():
        issues.append("manifest.json is missing")
    else:
        try:
            manifest = _read_json_file(CODEX_APP_MANIFEST)
        except json.JSONDecodeError as exc:
            issues.append(f"manifest.json is not valid JSON: {exc}")

    files = manifest.get("files", {}) if manifest else {}
    sft_path = _dataset_path(files.get("sft") if isinstance(files, dict) else None)
    eval_path = _dataset_path(files.get("eval") if isinstance(files, dict) else None)
    snapshot_path = _dataset_path(
        files.get("skill_inventory_snapshot") if isinstance(files, dict) else None
    )

    if manifest and sft_path is None:
        issues.append("manifest files.sft is missing or unsafe")
    if manifest and eval_path is None:
        issues.append("manifest files.eval is missing or unsafe")

    snapshot_total: int | None = None
    if snapshot_path and snapshot_path.exists():
        try:
            snapshot = _read_json_file(snapshot_path)
            total = snapshot.get("total_skills")
            snapshot_total = total if isinstance(total, int) else None
        except json.JSONDecodeError:
            issues.append("skill inventory snapshot is not valid JSON")

    sources_dir = CODEX_APP_DATASET_DIR / "sources"
    return {
        "ready": bool(manifest) and not issues,
        "error": None if not issues else "; ".join(issues),
        "datasetRoot": str(CODEX_APP_DATASET_DIR),
        "manifest": manifest,
        "counts": {
            "sftRows": _count_jsonl(sft_path) if sft_path else 0,
            "evalRows": _count_jsonl(eval_path) if eval_path else 0,
            "skillSnapshotSkills": snapshot_total,
            "sourceFiles": len(list(sources_dir.glob("*"))) if sources_dir.exists() else 0,
        },
        "issues": issues,
    }


@app.get("/api/training/overview")
async def training_overview() -> dict[str, object]:
    return training_runs.training_overview()


@app.post("/api/training/runs/plan")
async def plan_training_run(payload: dict[str, Any] | None = None) -> dict[str, object]:
    return training_runs.plan_training_run(payload)


@app.get("/api/training/runs")
async def list_training_runs() -> dict[str, object]:
    return {"runs": training_runs.list_runs()}


@app.post("/api/training/runs")
async def create_training_run(payload: dict[str, Any] | None = None) -> dict[str, object]:
    payload = payload or {}
    execute = bool(payload.pop("execute", False))
    return training_runs.create_run(payload, execute=execute)


@app.get("/api/training/runs/{run_id}")
async def get_training_run(run_id: str) -> dict[str, object]:
    try:
        return training_runs.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}") from exc


@app.post("/api/training/runs/{run_id}/stop")
async def stop_training_run(run_id: str) -> dict[str, object]:
    try:
        return training_runs.stop_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}") from exc


@app.post("/api/training/runs/{run_id}/resume")
async def resume_training_run(run_id: str, payload: dict[str, Any] | None = None) -> dict[str, object]:
    try:
        return training_runs.resume_run(run_id, execute=bool((payload or {}).get("execute", False)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}") from exc


@app.post("/api/tensorboard/start")
async def start_tensorboard() -> dict[str, object]:
    ready, error = await _ensure_tensorboard()
    return {
        "ready": ready,
        "error": error,
        "embeddedPath": "/tensorboard/",
        "logdir": str(DEFAULT_LOGDIR),
        "port": TENSORBOARD_PORT,
    }


@app.get("/api/tensorboard/summary")
async def tensorboard_summary() -> dict[str, object]:
    ready, error = await _ensure_tensorboard()
    if not ready:
        return {"ready": False, "error": error, "runs": [], "scalars": []}

    async with httpx.AsyncClient(timeout=30.0) as client:
        tags_response = await client.get(f"{TENSORBOARD_BASE}/tensorboard/data/plugin/scalars/tags")

        if tags_response.status_code == 404:
            return {"ready": True, "error": None, "runs": [], "scalars": []}
        tags_response.raise_for_status()
        tags_by_run = tags_response.json()

        scalars: list[dict[str, object]] = []
        for run, tags in tags_by_run.items():
            tag_names = list(tags.keys()) if isinstance(tags, dict) else list(tags)
            for tag in tag_names[:8]:
                scalar_response = await client.get(
                    f"{TENSORBOARD_BASE}/tensorboard/data/plugin/scalars/scalars",
                    params={"run": run, "tag": tag, "format": "json"},
                )
                if scalar_response.status_code != 200:
                    continue
                points = scalar_response.json()
                if not points:
                    continue
                latest = points[-1]
                first = points[0]
                scalars.append(
                    {
                        "run": run,
                        "tag": tag,
                        "step": latest[1],
                        "value": latest[2],
                        "delta": latest[2] - first[2],
                    }
                )

    return {
        "ready": True,
        "error": None,
        "runs": sorted(tags_by_run.keys()),
        "scalars": scalars,
    }


@app.api_route("/tensorboard/{path:path}", methods=["GET", "POST"])
async def proxy_tensorboard(path: str, request: Request) -> Response:
    ready, error = await _ensure_tensorboard()
    if not ready:
        raise HTTPException(status_code=503, detail=error or "TensorBoard is not ready.")

    upstream_path = f"/tensorboard/{path}"
    upstream_url = httpx.URL(f"{TENSORBOARD_BASE}{upstream_path}").copy_with(
        query=request.url.query.encode("utf-8")
    )
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        upstream = await client.request(
            request.method,
            upstream_url,
            headers=headers,
            content=await request.body(),
        )

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = {
        key: value for key, value in upstream.headers.items() if key.lower() not in excluded_headers
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}")
async def serve_spa(path: str) -> FileResponse:
    index_html = FRONTEND_DIST / "index.html"
    if not index_html.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run `npm install` and `npm run build` in frontend/.",
        )
    return FileResponse(index_html)
