# Rnv1-ReTrain Skill Routing

Standing rule: on every non-trivial task, evaluate implicit skill triggers before editing files or running long commands.

## Read First

Read `HANDOFF.md` before project work. It tracks the current ReTrain move, dashboard state, and verification steps. Keep this workspace standalone: do not wire it back into AIWF Studio or MoK unless Shawn asks for that integration.

## Project Shape

- `backend/` is a FastAPI app for the ReTrain dashboard.
- `frontend/` is a Vite React dashboard.
- `datasets/` stores ReTrain training corpora that are owned by this workspace.
- `training/EveryDream2trainer/` is an upstream trainer checkout. Do not edit it unless the task is explicitly about trainer internals.

## Routing Table

| Scenario | Use when | Skills | Expected output |
|---|---|---|---|
| Codex app environment training | The task mentions Codex, MCP, skills, plugins, tools, apps, connectors, or tool-use datasets. | `codex-tooling`, `avoid-ai-writing` for generated docs | Dataset or guidance under `datasets/`, plus validation. |
| Local model training runs | The task changes trainer launch, LoRA/QLoRA/full-SFT settings, model or dataset selection, VRAM estimates, run receipts, or training artifacts. | `local-ai-dev`; consider quantization or serving skills only when the task reaches those systems | Backend API changes plus dry-run or receipt validation; avoid loading models unless Shawn explicitly asks. |
| Project skill routing | A task asks which skills apply, the routing map is stale, or a new recurring lane appears. | `projectskill-list` | Updated `PROJECT_SKILLS.md` and `docs/projectskill-list.use-cases.json`. |
| Dashboard or frontend work | The task changes React views, dashboard layout, UI state, or browser rendering. | `build-web-apps:frontend-testing-debugging`; consider `build-web-apps:frontend-app-builder` for larger rebuilds | `npm run build`, and browser/smoke verification when layout risk is meaningful. |
| FastAPI backend work | The task changes API routes, TensorBoard proxying, app startup, or service status. | No extra skill by default | Python compile checks and targeted endpoint smoke tests. |
| Training data quality | The task adds or audits structured training corpora, evals, manifests, or receipts. | Consider `data-analytics:analyze-data-quality` for larger dataset audits | JSONL validation, count checks, and a short data-quality note. |
| OpenAI or Codex API docs | The task needs current OpenAI API, Agents SDK, Responses API, or Codex product details. | `openai-docs`; consider `openai-developers:agents-sdk` for Agents SDK builds | Official-doc-grounded implementation or notes. |

## Guardrails

- Keep AIWF Studio focused on inference and generation workflows.
- Keep MoK out of this workspace unless Shawn asks for MoK integration. ReTrain may train against MoK datasets as external inputs.
- Add ReTrain-owned data under `datasets/`, not inside `training/EveryDream2trainer/`.
- Keep training state and model weights out of git: `training/run_state/`, `training/runs/`, `models/`, and weight files.
- Prefer local verification before claims about installed tools, skills, and app state.
- Use `apply_patch` for manual edits and keep changes scoped to the task.
