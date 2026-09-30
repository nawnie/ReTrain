"""Read a local model's layer geometry and weight statistics without loading it.

Why this can be fast
--------------------
A safetensors file begins with an 8-byte little-endian header length followed by
that many bytes of JSON describing every tensor: its dtype, its shape, and the
byte range holding its data. So the entire structure of a model -- how many
layers, how wide each projection is, how many parameters sit in each block --
is readable from the first few tens of kilobytes of a 3GB file. No torch, no
GPU, no model load.

Weight statistics are a different matter: those need the actual bytes. Reading
every weight of a 1.5B model to answer "how dense is layer 14" would move
gigabytes for a number that a sample answers just as well, so this module reads
a bounded set of evenly spaced windows from each tensor and reports the result
as sampled. Sampled numbers are labelled as such everywhere they surface; they
are a guide for choosing where to train, not a measurement to quote.

What "density" means here
-------------------------
The word is doing several jobs, so this module reports each one separately
rather than inventing a single score:

  params   how many parameters the block holds        (structural, exact)
  bytes    what it occupies on disk                   (structural, exact)
  rms      root-mean-square weight magnitude          (sampled)
  active   fraction of weights above 1% of the block's
           own peak magnitude -- how much of the block
           carries signal rather than near-zero noise  (sampled)

A block can be large and quiet (many parameters, low RMS) or small and loud.
Those are different training targets, which is exactly why one number would
hide the choice being made.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# safetensors primitives
# ---------------------------------------------------------------------------

# Bytes per element. Anything absent here is still counted structurally by
# shape, but its statistics are skipped rather than guessed at.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
# Only these can be decoded into floats for the sampled statistics.
SAMPLEABLE_DTYPES = {"F16", "BF16", "F32"}

# The largest header this will accept. Real headers are tens of KB; a value far
# beyond that means the file is not what it claims to be.
MAX_HEADER_BYTES = 64 * 1024 * 1024


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Return the tensor index from a safetensors file's JSON header."""
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise ValueError(f"{path.name} is too short to be a safetensors file")
        length = struct.unpack("<Q", raw_length)[0]
        if not 0 < length <= MAX_HEADER_BYTES:
            raise ValueError(f"{path.name} declares an implausible header length")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{path.name} header is not a JSON object")
    # The data section starts immediately after the header, and every tensor's
    # data_offsets are relative to that point.
    header["__data_start__"] = 8 + length
    return header


def shard_files(model_dir: Path) -> list[Path]:
    """Every safetensors shard for a model, single-file or sharded.

    A sharded checkpoint ships model.safetensors.index.json naming its parts;
    the index is preferred over a glob so shards that belong to a different
    checkpoint sitting in the same folder are not mixed in.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted({str(name) for name in index.get("weight_map", {}).values()})
            shards = [model_dir / name for name in names]
            if shards and all(shard.is_file() for shard in shards):
                return shards
        except (OSError, json.JSONDecodeError):
            pass  # fall through to the glob
    return sorted(model_dir.glob("*.safetensors"))


# ---------------------------------------------------------------------------
# Naming: turning tensor names into a layer graph
#
# Architectures disagree about what to call things. Qwen and Llama use
# `model.layers.N.`, GPT-2 descendants use `transformer.h.N.`, and others use
# `blocks.N.`. All three are matched rather than assuming one family.
# ---------------------------------------------------------------------------

LAYER_INDEX_RE = re.compile(r"\.(?:layers|h|blocks|layer)\.(\d+)\.")

# Module roles, in the order they appear in a forward pass. Order matters: it is
# what makes the visualisation read as a pipeline rather than a bar chart.
ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("norm", re.compile(r"(layernorm|layer_norm|\bnorm\b|ln_\d|rmsnorm)", re.I)),
    ("attn.q", re.compile(r"(q_proj|query|\bwq\b|attn\.q)", re.I)),
    ("attn.k", re.compile(r"(k_proj|\bkey\b|\bwk\b|attn\.k)", re.I)),
    ("attn.v", re.compile(r"(v_proj|value|\bwv\b|attn\.v)", re.I)),
    ("attn.o", re.compile(r"(o_proj|out_proj|\bwo\b|attn\.proj|c_proj)", re.I)),
    ("attn", re.compile(r"(self_attn|attention|attn)", re.I)),
    ("mlp.gate", re.compile(r"(gate_proj|w1\b|\bgate\b)", re.I)),
    ("mlp.up", re.compile(r"(up_proj|w3\b|c_fc|fc_in)", re.I)),
    ("mlp.down", re.compile(r"(down_proj|w2\b|fc_out)", re.I)),
    ("mlp", re.compile(r"(mlp|feed_forward|ffn|moe)", re.I)),
    ("conv", re.compile(r"(conv|shortconv)", re.I)),
)

# Blocks outside the repeating stack. These bookend the model and are shown as
# caps rather than as layers, because they are not interchangeable with one.
TERMINAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("embedding", re.compile(r"(embed_tokens|wte|word_embeddings|tok_embeddings|embedding)", re.I)),
    ("head", re.compile(r"(lm_head|output\.weight|score\.weight)", re.I)),
    ("final_norm", re.compile(r"(model\.norm|final_layernorm|ln_f|\bnorm\.weight)", re.I)),
)


def classify_role(tensor_name: str) -> str:
    """Best-effort module role for one tensor name."""
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(tensor_name):
            return role
    return "other"


def classify_terminal(tensor_name: str) -> str | None:
    for role, pattern in TERMINAL_PATTERNS:
        if pattern.search(tensor_name):
            return role
    return None


# LoRA adapters attach to linear projections. Norms and embeddings are not
# adapter targets, so the UI must not offer them as one.
ADAPTER_TARGET_ROLES = {"attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"}


# ---------------------------------------------------------------------------
# Sampled weight statistics
# ---------------------------------------------------------------------------

def _decode(buffer: bytes, dtype: str):
    """Decode raw bytes into a float array, or None if numpy is unavailable.

    BF16 is the upper half of an FP32, so it is widened by shifting rather than
    converted; numpy has no native bfloat16.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    if dtype == "F32":
        return np.frombuffer(buffer, dtype="<f4")
    if dtype == "F16":
        return np.frombuffer(buffer, dtype="<f2").astype("f4")
    if dtype == "BF16":
        raw = np.frombuffer(buffer, dtype="<u2").astype("<u4")
        return (raw << 16).view("<f4")
    return None


def sample_tensor_stats(
    handle,
    data_start: int,
    offsets: tuple[int, int],
    dtype: str,
    *,
    windows: int,
    window_bytes: int,
) -> dict[str, float] | None:
    """Read a bounded, evenly spaced sample of one tensor and describe it.

    Evenly spaced rather than contiguous: weight matrices are laid out row-major,
    so a single leading chunk would only ever describe the first few output
    channels. Spreading the windows across the whole tensor samples every part
    of the matrix.
    """
    if dtype not in SAMPLEABLE_DTYPES:
        return None
    element_size = DTYPE_BYTES[dtype]
    start, end = offsets
    total = end - start
    if total <= 0:
        return None

    # Align the window to whole elements, and never ask for more than exists.
    window = max(element_size, (window_bytes // element_size) * element_size)
    window = min(window, total)
    count = max(1, min(windows, total // window))
    stride = (total - window) // count if count > 1 else 0

    chunks = []
    for index in range(count):
        offset = data_start + start + index * stride
        handle.seek(offset)
        raw = handle.read(window)
        if len(raw) < element_size:
            continue
        # Trim to a whole number of elements before decoding.
        raw = raw[: (len(raw) // element_size) * element_size]
        decoded = _decode(raw, dtype)
        if decoded is None:
            return None
        chunks.append(decoded)

    if not chunks:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    values = np.concatenate(chunks)
    # Non-finite values would poison every statistic below; a checkpoint with
    # them is worth knowing about, so they are counted rather than dropped
    # silently.
    finite_mask = np.isfinite(values)
    non_finite = int(values.size - int(finite_mask.sum()))
    values = values[finite_mask]
    if values.size == 0:
        return {"sampled": 0, "non_finite": non_finite}

    magnitude = np.abs(values)
    peak = float(magnitude.max())
    # "Active" is measured against the block's own peak, not a global constant,
    # so blocks at different scales stay comparable.
    threshold = peak * 0.01
    active = float((magnitude > threshold).mean()) if peak > 0 else 0.0

    return {
        "sampled": int(values.size),
        "rms": float(np.sqrt(np.mean(values.astype("f8") ** 2))),
        "mean_abs": float(magnitude.mean()),
        "peak": peak,
        "active": active,
        "non_finite": non_finite,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _element_count(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


def _blank_block(key: str, label: str, kind: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "params": 0,
        "bytes": 0,
        "tensors": [],
        "adapter_target": kind in ADAPTER_TARGET_ROLES,
        "stats": None,
    }


def _merge_stats(blocks: list[dict[str, Any]]) -> dict[str, float] | None:
    """Combine child statistics into a parent, weighted by how much was sampled."""
    scored = [block for block in blocks if block.get("stats") and block["stats"].get("sampled")]
    if not scored:
        return None
    total = sum(block["stats"]["sampled"] for block in scored)
    if total <= 0:
        return None
    weighted = lambda key: sum(  # noqa: E731 - a local shorthand, used three times
        block["stats"].get(key, 0.0) * block["stats"]["sampled"] for block in scored
    ) / total
    return {
        "sampled": total,
        "rms": weighted("rms"),
        "mean_abs": weighted("mean_abs"),
        "peak": max(block["stats"].get("peak", 0.0) for block in scored),
        "active": weighted("active"),
        "non_finite": sum(block["stats"].get("non_finite", 0) for block in scored),
    }


def inspect_model(
    model_dir: Path,
    *,
    sample: bool = True,
    windows: int = 8,
    window_bytes: int = 32768,
    include_tensors: bool = False,
) -> dict[str, Any]:
    """Describe a local checkpoint's layer stack.

    Structure is exact and comes from the headers. Statistics are sampled, and
    are omitted entirely when `sample` is false or numpy is unavailable, rather
    than being filled in with a placeholder that would read as a measurement.

    Per-tensor records are dropped unless `include_tensors` is set: a 28-layer
    model carries a few hundred of them and they quadruple the payload, while
    the visualisation only ever reads block-level totals.
    """
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    shards = shard_files(model_dir)
    if not shards:
        raise FileNotFoundError(f"No .safetensors weights in {model_dir}")

    config: dict[str, Any] = {}
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}

    layers: dict[int, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    dtype_counts: dict[str, int] = {}
    total_params = 0
    total_bytes = 0
    sampled_any = False

    for shard in shards:
        header = read_safetensors_header(shard)
        data_start = header.pop("__data_start__")
        header.pop("__metadata__", None)

        # Open the shard once and sample every tensor in it, so the whole model
        # costs one file handle per shard rather than one per tensor.
        handle = shard.open("rb") if sample else None
        try:
            for name, spec in header.items():
                if not isinstance(spec, dict) or "shape" not in spec:
                    continue
                dtype = str(spec.get("dtype", "?"))
                shape = [int(value) for value in spec.get("shape", [])]
                params = _element_count(shape)
                offsets = spec.get("data_offsets") or [0, 0]
                size = int(offsets[1]) - int(offsets[0])

                total_params += params
                total_bytes += size
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

                stats = None
                if handle is not None and params > 0:
                    stats = sample_tensor_stats(
                        handle,
                        data_start,
                        (int(offsets[0]), int(offsets[1])),
                        dtype,
                        windows=windows,
                        window_bytes=window_bytes,
                    )
                    if stats:
                        sampled_any = True

                tensor_record = {
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "params": params,
                    "bytes": size,
                    "stats": stats,
                }

                match = LAYER_INDEX_RE.search(name)
                if match:
                    index = int(match.group(1))
                    layer = layers.setdefault(
                        index,
                        {"index": index, "params": 0, "bytes": 0, "blocks": {}},
                    )
                    role = classify_role(name)
                    block = layer["blocks"].setdefault(role, _blank_block(role, role, role))
                    block["params"] += params
                    block["bytes"] += size
                    block["tensors"].append(tensor_record)
                    layer["params"] += params
                    layer["bytes"] += size
                else:
                    role = classify_terminal(name) or "other"
                    terminal = terminals.setdefault(role, _blank_block(role, role, role))
                    terminal["params"] += params
                    terminal["bytes"] += size
                    terminal["tensors"].append(tensor_record)
        finally:
            if handle is not None:
                handle.close()

    # Roll per-tensor statistics up into blocks, then blocks into layers.
    layer_list: list[dict[str, Any]] = []
    for index in sorted(layers):
        layer = layers[index]
        blocks = []
        for role, block in layer["blocks"].items():
            block["stats"] = _merge_stats(
                [{"stats": tensor["stats"]} for tensor in block["tensors"]]
            )
            block["label"] = role
            block["tensor_count"] = len(block["tensors"])
            if not include_tensors:
                block.pop("tensors")
            blocks.append(block)
        # Forward-pass order, so the stack reads as a pipeline.
        role_order = [role for role, _ in ROLE_PATTERNS] + ["other"]
        blocks.sort(key=lambda item: role_order.index(item["kind"]) if item["kind"] in role_order else 99)
        layer_list.append({
            "index": index,
            "label": f"Layer {index}",
            "params": layer["params"],
            "bytes": layer["bytes"],
            "blocks": blocks,
            "stats": _merge_stats(blocks),
        })

    terminal_list = []
    for role in ("embedding", "final_norm", "head", "other"):
        if role not in terminals:
            continue
        terminal = terminals[role]
        terminal["stats"] = _merge_stats(
            [{"stats": tensor["stats"]} for tensor in terminal["tensors"]]
        )
        terminal["label"] = role.replace("_", " ")
        terminal["tensor_count"] = len(terminal["tensors"])
        if not include_tensors:
            terminal.pop("tensors")
        terminal_list.append(terminal)

    return {
        "schema": "retrain.model-layers.v1",
        "name": model_dir.name,
        "path": str(model_dir),
        "architecture": (config.get("architectures") or ["unknown"])[0],
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "declared_layers": config.get("num_hidden_layers"),
        "shards": [shard.name for shard in shards],
        "dtypes": dtype_counts,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "layer_count": len(layer_list),
        "layers": layer_list,
        "terminals": terminal_list,
        # Stated plainly so the client never presents sampled figures as exact.
        "sampled": bool(sample and sampled_any),
        "sample_plan": {"windows": windows, "window_bytes": window_bytes} if sample else None,
        "adapter_target_roles": sorted(ADAPTER_TARGET_ROLES),
    }


def list_local_models(model_root: Path) -> list[dict[str, Any]]:
    """Every folder under the model root that actually holds safetensors weights."""
    model_root = model_root.resolve()
    if not model_root.is_dir():
        return []
    rows = []
    for child in sorted(model_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        shards = shard_files(child)
        if not shards:
            continue
        rows.append({
            "folder": child.name,
            "path": str(child),
            "shards": len(shards),
            "bytes": sum(shard.stat().st_size for shard in shards),
        })
    return rows


def resolve_model_dir(model_root: Path, folder: str) -> Path:
    """Resolve a client-supplied folder name, refusing anything outside the root.

    The client sends a folder name, never a path. Resolving and then checking
    containment means a name carrying `..` or an absolute path cannot escape,
    which matters because this module opens and reads whatever it is given.
    """
    model_root = model_root.resolve()
    candidate = (model_root / folder).resolve()
    if candidate != model_root and model_root not in candidate.parents:
        raise ValueError("Model folder must be inside the configured model root.")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")
    return candidate
