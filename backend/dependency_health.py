from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from typing import Any, Iterable


@lru_cache(maxsize=1)
def bitsandbytes_native_status() -> dict[str, Any]:
    """Verify that BitsAndBytes has a loadable CUDA backend, not only metadata."""
    probe = (
        "import json, bitsandbytes as bnb; "
        "lib=getattr(getattr(bnb, 'cextension', None), 'lib', None); "
        "ready=bool(lib and getattr(lib, 'compiled_with_cuda', False)); "
        "print(json.dumps({'ready': ready, 'type': type(lib).__name__ if lib else 'missing'}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "detail": f"native probe failed: {exc}"}

    payload: dict[str, Any] = {}
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    ready = result.returncode == 0 and bool(payload.get("ready"))
    if ready:
        return {"ready": True, "detail": "CUDA native library loaded"}
    error_line = next(
        (line.strip() for line in result.stderr.splitlines() if "CUDA" in line and "not found" in line),
        "CUDA native library did not load",
    )
    return {"ready": False, "detail": error_line}


def inspect_packages(packages: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    status: list[dict[str, Any]] = []
    for package, label in packages:
        installed = importlib.util.find_spec(package) is not None
        item: dict[str, Any] = {
            "package": package,
            "label": label,
            "installed": installed,
            "available": installed,
        }
        if package == "bitsandbytes" and installed:
            native = bitsandbytes_native_status()
            item["available"] = native["ready"]
            item["detail"] = native["detail"]
        status.append(item)
    return status


def clear_dependency_probe_cache() -> None:
    bitsandbytes_native_status.cache_clear()
