# ReTrain

ReTrain is an approachable, Windows-first local training workbench. It helps a
person inspect local datasets and model folders, plan a bounded run, check
readiness and estimated VRAM, run a dry check, and keep receipts, logs, and
TensorBoard summaries together. It is not merely a generic dashboard.

ReTrain is standalone. It can use an explicitly configured MoK dataset as an
external input, but it does not merge MoK or AIWF Studio into this repository.

## What runs today

The current execution surface is deliberately narrower than the product
roadmap:

- Causal and prompt-helper decoder LMs: Transformers `Trainer` full SFT, LoRA,
  and QLoRA through `scripts/run_posttrain_bakeoff.py`.
- Seq2Seq/T5 targets: full fine-tuning through
  `scripts/run_text_target_training.py`.
- Generic masked-LM text encoders: full fine-tuning through the same text-side
  runner.
- Planning only: CLIP text, BLIP text, and listed component encoders remain
  blocked until a matching paired-data runner exists.

DoRA, adapter stacking/routing, expert packs, and dense model growth are not
implemented execution modes yet. They are documented research directions, not
current capabilities.

## Additive adapter growth

ReTrain's near-term research direction is a frozen-core base model with modular
LoRA or QLoRA capability packs. The working engine can save LoRA/QLoRA output;
the future adapter/expert packaging, routing, stacking, and DoRA path still
need implementation and measured evaluation.

See [the additive-adapter notes](docs/training_modes/additive_adapter_growth.md).

## Repository map

See [docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md) for the code, data,
runtime-state, and documentation boundaries. Release changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Install

From the repository root:

```powershell
.\scripts\install_retrain.ps1
```

The installer creates a project-owned `.venv`, installs the app and training
dependencies, builds the frontend, validates CUDA visibility by default, and
runs static/dataset checks. It does not download model weights.

Optional local configuration is via environment variables, never hard-coded
workstation paths:

| Variable | Purpose |
| --- | --- |
| `RETRAIN_MODEL_CANDIDATES_ROOT` | Folder containing already-downloaded local base models. Defaults to `models/candidates`. |
| `RETRAIN_MOK_ROOT` | Optional external MoK checkout used only for dataset discovery. |
| `RETRAIN_TORCH_WHEELHOUSE` | Optional local PyTorch wheelhouse. |
| `RETRAIN_TRAINING_WHEELHOUSE` | Optional local training-dependency wheelhouse. |
| `RETRAIN_SMOKE_MODEL_PATH` + `RETRAIN_SMOKE_DATA_DIR` | Optional pair enabling the installer’s local-model dry run. |

Use `-Offline` only when one or both configured wheelhouses contain everything
required by the requirements files.

## Launch

```powershell
.\scripts\start_retrain.ps1
```

Then open `http://127.0.0.1:8000`.

## Safe smoke checks

```powershell
.\.venv\Scripts\python.exe -m compileall backend scripts
.\.venv\Scripts\python.exe datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
Push-Location frontend; npm run build; Pop-Location
```

With the app running, planning and `dryRun: true` execution create receipts
without loading model weights. Real training requires `confirmed: true` and
`dryRun: false`.

## Guardrails

- Do not download large models or run VRAM-heavy training unless explicitly requested.
- Do not commit weights, private datasets, run state, logs, environment files, or generated frontend output.
- TensorBoard is loopback-only through the local app; this repository does not configure hosted service, auth, or cloud release flows.
- Treat planned training modes as planned until they have a matching runner and measured evidence.
