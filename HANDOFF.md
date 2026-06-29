# ReTrain Handoff

## Move

- Shawn asked to move this work out of `C:\Users\Shawn\Desktop\MoK-Project`.
- Windows cannot create a folder named `Rnv1:ReTrain` because `:` is invalid in
  filenames.
- The intended new folder is `C:\Users\Shawn\Desktop\Rnv1-ReTrain`.

## Current files

- `backend/main.py` defines a FastAPI app.
- `backend/training_runs.py` owns the ReTrain training-run planner, local model
  discovery, dataset discovery, VRAM estimates, run receipts, and start/stop
  state.
- `scripts/run_posttrain_bakeoff.py` was copied from Project MoK and patched as
  a ReTrain-owned local SFT/LoRA/QLoRA smoke runner. Keep future trainer changes
  here unless the task is specifically about MoK internals.
- `frontend/` contains a Vite React app for the ReTrain dashboard.
- `requirements.txt` lists FastAPI, Uvicorn, httpx, and TensorBoard.
  `setuptools<81` is pinned because TensorBoard 2.20 still imports
  `pkg_resources`.
- `datasets/codex_app_environment/` contains the first ReTrain-owned corpus for
  teaching Codex app environment behavior: skills, plugins, MCP tools, app
  connectors, local tools, memory, privacy, and verification habits.
- `training/EveryDream2trainer` remains the upstream trainer folder and should
  not be edited unless the task specifically calls for trainer internals.

## Dashboard intent

- The first screen is now a training-run dashboard, not a raw TensorBoard iframe.
- The dashboard shows run config, dataset/model selectors, readiness gates, VRAM
  estimate, run history, selected-run artifacts, log tail, and scalar summary.
- The backend starts TensorBoard on `127.0.0.1:6006` with the path prefix
  `/tensorboard`.
- TensorBoard now points at `training/runs`, the ReTrain run-output root.
- `/api/tensorboard/summary` reads TensorBoard scalar data and returns compact
  run/tag/value records for the React dashboard.
- The React app has a `Pop Up TensorBoard` button that opens the full embedded
  TensorBoard UI in a modal.
- `/api/training/codex-app` reports the Codex environment training-pack
  manifest and row counts for the dashboard.

## Training-run MVP

- ReTrain is its own repo and training workbench. Project MoK remains about MoK;
  ReTrain may train MoK models and datasets as external inputs without becoming
  part of MoK.
- `GET /api/training/overview` reports hardware, discovered datasets, local
  model candidates, dependencies, default config, and run history.
- `POST /api/training/runs/plan` returns a command preview, readiness gates, and
  VRAM estimate.
- `POST /api/training/runs` creates a run. With `execute: true` and `dryRun:
  true`, it runs the copied runner in dry-run mode and writes a receipt/log under
  `training/run_state`. Real training requires confirmation and `dryRun: false`.
- `GET /api/training/runs`, `GET /api/training/runs/{run_id}`,
  `POST /api/training/runs/{run_id}/stop`, and
  `POST /api/training/runs/{run_id}/resume` are wired.
- The default runner-ready dataset is Project MoK's external posttrain bakeoff
  pack when it exists:
  `C:\Users\Shawn\Desktop\MoK-Project\training\posttrain_bakeoff\data`.
- The default local starter model is
  `F:\Ai_Models\hf\posttrain_candidates\Qwen--Qwen2.5-Coder-1.5B` when present.
- Runtime state and heavy artifacts are ignored by git:
  `training/run_state/`, `training/runs/`, `models/`, and model weight suffixes.

## Codex app training intent

- Train behavior before memorization. The model should inspect the active tool
  list, read triggered skills, validate MCP schemas, protect user files, avoid
  secrets in corpora, and prove work with receipts.
- Refresh `datasets/codex_app_environment/sources/skill_inventory_snapshot.json`
  after skills or plugins change.
- Validate the pack with:

```powershell
python datasets\codex_app_environment\scripts\build_skills_snapshot.py
python datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
```

## Verification completed on 2026-06-28

Use `py -3.10` for Python setup on this machine. In the current Codex shell,
plain `python` points at the Hermes agent venv and has no `pip`.

Completed checks from `C:\Users\Shawn\Desktop\Rnv1-ReTrain`:

```powershell
py -3.10 -m pip install -r requirements.txt
python datasets\codex_app_environment\scripts\build_skills_snapshot.py
python datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
py -3.10 -m compileall backend datasets\codex_app_environment\scripts
cd frontend; npm install; npm run build
```

Endpoint receipts with the FastAPI app running at `http://127.0.0.1:8000`:

- `http://127.0.0.1:8000/api/health` returns JSON with `"ok": true`.
- `http://127.0.0.1:8000/api/training/codex-app` returns `ready: true`,
  `16` SFT rows, `10` eval rows, and `198` snapshot skills.
- `http://127.0.0.1:8000/` loads the React dashboard after `npm run build`.
- `POST http://127.0.0.1:8000/api/tensorboard/start` returns `ready: true`.
- `http://127.0.0.1:8000/tensorboard/` returns HTTP 200 through the proxy.

Manual browser check still useful: press `Pop Up TensorBoard` and confirm the
modal shows the raw TensorBoard UI.

## Known state before the move

- Before the move, the active Python had `fastapi`, `uvicorn`, and `httpx`, but
  not `tensorboard`.
- Build and smoke verification were first completed after the standalone
  ReTrain move.

## Verification completed on 2026-06-29

`git init` was run in `C:\Users\Shawn\Desktop\Rnv1-ReTrain` because `.git` was
an empty directory and `git rev-parse` previously failed.

Completed checks:

```powershell
py -3.10 -m compileall backend scripts
py -3.10 datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
cd frontend; npm run build
py -3.10 scripts\run_posttrain_bakeoff.py --dry-run --model qwen2.5-coder-1.5b --data-dir C:\Users\Shawn\Desktop\MoK-Project\training\posttrain_bakeoff\data --output-root training\runs\smoke
```

Endpoint receipts with FastAPI running at `http://127.0.0.1:8000`:

- `/api/health` returns `ok: true` and TensorBoard logdir
  `C:\Users\Shawn\Desktop\Rnv1-ReTrain\training\runs`.
- `/api/training/overview` returns `4` datasets, `7` models, and the local GPU
  `NVIDIA GeForce RTX 4070 Ti SUPER` with `16376 MB` VRAM.
- `POST /api/training/runs/plan` returns default dataset
  `mok-posttrain-bakeoff`, default model `qwen2.5-coder-1.5b`, safe VRAM
  estimate around `3.4 GB`, and `startEnabled: true`.
- `POST /api/training/runs` with `execute: true` and `dryRun: true` writes a
  `dry_run_completed` receipt.
- `/tensorboard/` returns HTTP 200 through the proxy.

Browser QA:

- `http://127.0.0.1:8000/` loads the React dashboard.
- The first screen auto-plans and shows readiness gates plus the safe VRAM
  estimate.
- `Plan` and `Run Dry Check` work from the UI and update run history.
- Desktop and mobile screenshots showed no blocking overlap or console errors.
