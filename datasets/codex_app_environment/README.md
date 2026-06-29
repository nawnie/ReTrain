# Codex App Environment Training Pack

This pack teaches a local model how to behave when it is loaded into Codex with project files, skills, plugins, MCP tools, app connectors, browser tools, and local execution tools available.

The goal is behavior, not memorization. The model should learn to inspect the active tool list, read project instructions, load relevant skills before using them, validate schemas, keep file edits scoped, and prove work with local checks.

## Files

- `sources/codex_app_environment.md`: plain-language operating guide for Codex app behavior.
- `sources/codex_tool_taxonomy.json`: structured definitions for skills, plugins, MCP tools, apps, connectors, and local tools.
- `sources/skill_inventory_snapshot.json`: generated snapshot of installed local skills.
- `sft/codex_app_tool_use_sft.jsonl`: supervised examples for tool-routing behavior.
- `evals/codex_app_tool_use_eval.jsonl`: held-out eval prompts and scoring rules.
- `scripts/build_skills_snapshot.py`: refreshes the local skill inventory snapshot.
- `scripts/validate_codex_app_dataset.py`: validates JSON, JSONL, ids, counts, and manifest references.

## Refresh

Run this from the repo root after Codex skills or plugins change:

```powershell
python datasets\codex_app_environment\scripts\build_skills_snapshot.py
python datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
```

Keep private secrets, API keys, and personal message contents out of this pack. Train the model on routing habits and safe tool use, not on hidden chain-of-thought or credentials.
