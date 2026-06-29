from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = DATASET_ROOT / "manifest.json"


class ValidationError(Exception):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return loaded


def _resolve_dataset_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"Unsafe dataset path in manifest: {relative_path}")
    return DATASET_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number} invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _require_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValidationError(f"{label} row missing string id")
        if row_id in seen:
            raise ValidationError(f"{label} duplicate id: {row_id}")
        seen.add(row_id)


def _validate_sft(rows: list[dict[str, Any]]) -> None:
    _require_unique_ids(rows, "sft")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValidationError(f"{row['id']} must include at least two messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValidationError(f"{row['id']} contains a non-object message")
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValidationError(f"{row['id']} has invalid message role")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValidationError(f"{row['id']} has empty message content")


def _validate_eval(rows: list[dict[str, Any]]) -> None:
    _require_unique_ids(rows, "eval")
    for row in rows:
        for key in ("prompt", "ideal_behavior", "must", "must_not"):
            if key not in row:
                raise ValidationError(f"{row['id']} missing {key}")
        if not isinstance(row["must"], list) or not isinstance(row["must_not"], list):
            raise ValidationError(f"{row['id']} must and must_not must be lists")


def validate() -> dict[str, int]:
    manifest = _load_json(MANIFEST)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValidationError("manifest files must be an object")

    for key in ("source_guide", "tool_taxonomy", "sft", "eval"):
        if key not in files:
            raise ValidationError(f"manifest missing files.{key}")
        path = _resolve_dataset_path(files[key])
        if not path.exists():
            raise ValidationError(f"manifest references missing file: {path}")

    sft_rows = _load_jsonl(_resolve_dataset_path(files["sft"]))
    eval_rows = _load_jsonl(_resolve_dataset_path(files["eval"]))
    _validate_sft(sft_rows)
    _validate_eval(eval_rows)

    taxonomy = _load_json(_resolve_dataset_path(files["tool_taxonomy"]))
    if not isinstance(taxonomy.get("categories"), list) or not taxonomy["categories"]:
        raise ValidationError("tool taxonomy must contain categories")

    expected_counts = manifest.get("counts", {})
    if isinstance(expected_counts, dict):
        if expected_counts.get("sft_rows") != len(sft_rows):
            raise ValidationError("manifest counts.sft_rows does not match SFT row count")
        if expected_counts.get("eval_rows") != len(eval_rows):
            raise ValidationError("manifest counts.eval_rows does not match eval row count")

    snapshot_path = _resolve_dataset_path(
        files.get("skill_inventory_snapshot", "sources/skill_inventory_snapshot.json")
    )
    snapshot_skills = 0
    if snapshot_path.exists():
        snapshot = _load_json(snapshot_path)
        total = snapshot.get("total_skills")
        snapshot_skills = total if isinstance(total, int) else 0

    return {
        "sft_rows": len(sft_rows),
        "eval_rows": len(eval_rows),
        "snapshot_skills": snapshot_skills,
    }


def main() -> int:
    try:
        counts = validate()
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        "Validation passed: "
        f"{counts['sft_rows']} SFT rows, "
        f"{counts['eval_rows']} eval rows, "
        f"{counts['snapshot_skills']} snapshot skills"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
