# Rnv1 ReTrain

Local Windows-first training workbench for LLM/chat fine-tuning, LoRA, and QLoRA experiments. ReTrain is standalone: it can use MoK datasets as external inputs, but it is not wired into MoK or AIWF Studio.

## MVP Surface

- FastAPI backend serving the React dashboard at `http://127.0.0.1:8000`
- User-friendly training-run dashboard with dataset/model selection, readiness gates, VRAM estimate, run history, artifacts, logs, and scalar summaries
- Safe dry-run path that writes receipts without loading model weights
- TensorBoard served from the same app and opened through the dashboard popup

## Setup

Use Python 3.10 on this machine. The plain `python` command may point at a different agent environment.

```powershell
cd C:\Users\Shawn\Desktop\Rnv1-ReTrain
py -3.10 -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

## Launch

```powershell
.\scripts\start_retrain.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

The startup script expects `frontend\dist\index.html` to exist. If it is missing, run `npm run build` in `frontend\`.

## Smoke Checks

```powershell
py -3.10 -m compileall backend scripts
py -3.10 datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
cd frontend
npm run build
cd ..
```

With the app running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/training/overview
Invoke-RestMethod http://127.0.0.1:8000/api/training/runs/plan -Method Post -ContentType 'application/json' -Body '{}'
Invoke-RestMethod http://127.0.0.1:8000/api/training/runs -Method Post -ContentType 'application/json' -Body '{"execute":true,"dryRun":true}'
Invoke-RestMethod http://127.0.0.1:8000/api/tensorboard/start -Method Post
Invoke-WebRequest http://127.0.0.1:8000/tensorboard/ -UseBasicParsing
```

Browser check:

- Dashboard loads at `/`
- `Plan` refreshes readiness
- `Run Dry Check` writes a `dry_run_completed` receipt
- `Start TensorBoard` reports ready
- `Open TensorBoard` opens the embedded TensorBoard modal

## Guardrails

- Do not download large models or run real VRAM-heavy training unless Shawn explicitly asks.
- Real training requires `confirmed: true` and `dryRun: false`.
- Keep runtime state and weights out of git: `training\run_state`, `training\runs`, `models`, logs, and weight files are ignored.
- Keep modality expansion out of this MVP. Image, video, audio, cloud, auth, and hosted release work are not part of this baseline.
