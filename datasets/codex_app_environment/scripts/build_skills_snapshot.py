from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "codex_app_environment" / "sources" / "skill_inventory_snapshot.json"


def _default_roots() -> list[Path]:
    codex_home = Path.home() / ".codex"
    return [
        codex_home / "skills",
        codex_home / "plugins" / "cache",
    ]


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text or text[0].strip() != "---":
        return {}

    parsed: dict[str, Any] = {}
    for line in text[1:]:
        if line.strip() == "---":
            break
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def _skill_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob("SKILL.md"))
    return sorted(set(files), key=lambda item: str(item).lower())


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.home())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_snapshot(roots: list[Path]) -> dict[str, Any]:
    skills: list[dict[str, Any]] = []
    for skill_file in _skill_files(roots):
        frontmatter = _parse_frontmatter(skill_file)
        name = frontmatter.get("name") or skill_file.parent.name
        source_root = next((root for root in roots if root in skill_file.parents), skill_file.parent)
        skills.append(
            {
                "name": name,
                "description": frontmatter.get("description", ""),
                "path": _safe_relative(skill_file),
                "source_root": _safe_relative(source_root),
            }
        )

    return {
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "roots": [_safe_relative(root) for root in roots],
        "total_skills": len(skills),
        "skills": sorted(skills, key=lambda item: str(item["name"]).lower()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local Codex skill inventory snapshot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    args = parser.parse_args()

    roots = args.roots if args.roots else _default_roots()
    snapshot = build_snapshot([root.expanduser().resolve() for root in roots])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {snapshot['total_skills']} skills to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
