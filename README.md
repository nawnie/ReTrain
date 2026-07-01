# Rnv1 ReTrain

Local Windows-first training workbench for LLM/chat fine-tuning, LoRA, and QLoRA experiments. ReTrain is standalone: it can use MoK datasets as external inputs, but it is not wired into MoK or AIWF Studio.

## MVP Surface

- FastAPI backend serving the React dashboard at `http://127.0.0.1:8000`
- User-friendly training-run dashboard with dataset/model selection, readiness gates, VRAM estimate, run history, artifacts, logs, and scalar summaries
- Safe dry-run path that writes receipts without loading model weights
- TensorBoard served from the same app and opened through the dashboard popup

## Training Mode Roadmap

- **Additive Adapter Growth:** planned frozen-core training mode where a stable 1.5B-class base model stays frozen while ReTrain trains LoRA, QLoRA, DoRA, adapter, or expert weights as modular add-ons.
- **Target package shape:** `1.5B frozen base + trainable adapter/expert weights = expanded 1.6B-2B total model package`.
- **Growth stages:** start with 16M, 64M, and 128M adapters before attempting 250M+ expert packs or experimental dense growth.
- **Capability packs:** code, tool-use, routing, Atlas/card reasoning, style/personality, and safety/alignment adapters.

See [`docs/training_modes/additive_adapter_growth.md`](docs/training_modes/additive_adapter_growth.md) for the detailed plan.

## Install

Use the project installer. It creates `\.venv`, installs the web app and CUDA
training stack, installs frontend dependencies, builds the dashboard, validates
CUDA visibility, and runs smoke checks.

```powershell
cd C:\Users\Shawn\Desktop\Rnv1-ReTrain
.\scripts\install_retrain.ps1
```

## Launch

```powershell
.\scripts\start_retrain.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

The startup script expects `frontend\dist\index.html` and `.\.venv` to exist.
If either is missing, run `.\scripts\install_retrain.ps1`. It will not silently
fall back to another project venv or global Python.

## Smoke Checks

```powershell
py -3.10 -m compileall backend scripts
.\.venv\Scripts\python.exe datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
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
