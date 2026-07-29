# ReTrain repository map

| Area | Purpose | Public boundary |
| --- | --- | --- |
| `backend/` | FastAPI routes, local planning, discovery, run receipts, and TensorBoard proxying. | Source only; runtime state stays ignored. |
| `frontend/` | Vite/React local dashboard. | Source and lockfile only; `node_modules/` and `dist/` stay ignored. |
| `scripts/` | Installer, loopback launcher, causal-LM runner, and text-target runner. | No model weights or personal paths. |
| `datasets/codex_app_environment/` | ReTrain-owned, validated example corpus for tool-use behavior. | Review every added row; do not add secrets, account data, or private records. |
| `docs/` | Product boundaries, training-mode notes, release review, and repository navigation. | Claims must distinguish implemented behavior from planned work. |
| `training/` | Generated run state, receipts, TensorBoard logs, and output artifacts. | Ignored; never publish outputs without an explicit review. |
| `models/` | Optional local base-model discovery root. | Ignored; model weights remain local. |

## External inputs

- `RETRAIN_MOK_ROOT` may point to an external MoK checkout for dataset
  discovery. It is optional and remains outside this repository.
- `RETRAIN_MODEL_CANDIDATES_ROOT` may point to a local model cache. ReTrain
  never downloads weights automatically.
