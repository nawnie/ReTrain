from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


ROOT_DIR: Final = Path(__file__).resolve().parents[1]
MOK_ROOT: Final = Path(r"C:\Users\Shawn\Desktop\MoK-Project")
RUNNER_SCRIPT: Final = ROOT_DIR / "scripts" / "run_posttrain_bakeoff.py"
TEXT_TARGET_RUNNER_SCRIPT: Final = ROOT_DIR / "scripts" / "run_text_target_training.py"
RUN_STATE_ROOT: Final = ROOT_DIR / "training" / "run_state"
RUN_OUTPUT_ROOT: Final = ROOT_DIR / "training" / "runs"
RETRAIN_MODEL_ROOT: Final = ROOT_DIR / "models"
HF_CANDIDATE_ROOT: Final = Path(r"F:\Ai_Models\hf\posttrain_candidates")
ANSI_RE: Final = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

METHOD_TO_RUNNER: Final = {
    "sft": "full_sft",
    "full_sft": "full_sft",
    "lora": "lora",
    "qlora": "qlora",
}
RUNNER_READY_TARGETS: Final = {"causal_lm", "prompt_helper_lm", "seq2seq_t5", "text_encoder"}
SUPPORTED_TARGET_TYPES: Final = {
    "causal_lm",
    "prompt_helper_lm",
    "seq2seq_t5",
    "text_encoder",
    "clip_text",
    "blip_text",
}
DATASET_FORMATS: Final = {"auto", "messages", "prompt_completion", "text", "text_pairs", "caption_pairs"}

MODEL_PRESETS: Final = {
    "smollm2-135m-prompt-helper": {
        "label": "SmolLM2 135M prompt helper",
        "size_b": 0.135,
        "family": "smollm2",
        "target_type": "prompt_helper_lm",
        "runner_key": "",
        "hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "path": HF_CANDIDATE_ROOT / "HuggingFaceTB--SmolLM2-135M-Instruct",
        "notes": ["Small prompt-helper candidate for rewrite, routing, and prompt expansion jobs."],
    },
    "pythia-160m-prompt-helper": {
        "label": "Pythia 160M prompt helper",
        "size_b": 0.16,
        "family": "pythia",
        "target_type": "prompt_helper_lm",
        "runner_key": "",
        "hf_id": "EleutherAI/pythia-160m",
        "path": HF_CANDIDATE_ROOT / "EleutherAI--pythia-160m",
        "notes": ["Tiny local LM candidate when a plain decoder model is enough."],
    },
    "google-t5-efficient-tiny": {
        "label": "Google T5 Efficient Tiny",
        "size_b": 0.01558,
        "family": "t5-efficient",
        "target_type": "seq2seq_t5",
        "runner_key": "",
        "hf_id": "google/t5-efficient-tiny",
        "path": HF_CANDIDATE_ROOT / "google--t5-efficient-tiny",
        "notes": ["Tiny T5 encoder-decoder candidate for text-to-text prompt helpers."],
    },
    "google-t5-efficient-mini": {
        "label": "Google T5 Efficient Mini",
        "size_b": 0.031,
        "family": "t5-efficient",
        "target_type": "seq2seq_t5",
        "runner_key": "",
        "hf_id": "google/t5-efficient-mini",
        "path": HF_CANDIDATE_ROOT / "google--t5-efficient-mini",
        "notes": ["Small T5-style text-to-text target; keep it separate from diffusion encoder swaps."],
    },
    "google-t5-efficient-small": {
        "label": "Google T5 Efficient Small",
        "size_b": 0.06,
        "family": "t5-efficient",
        "target_type": "seq2seq_t5",
        "runner_key": "",
        "hf_id": "google/t5-efficient-small",
        "path": HF_CANDIDATE_ROOT / "google--t5-efficient-small",
        "notes": ["Light T5-style target for prompt helper and text transform tasks."],
    },
    "openai-clip-vit-large-patch14-text": {
        "label": "CLIP-L text encoder",
        "size_b": 0.123,
        "family": "clip",
        "target_type": "clip_text",
        "runner_key": "",
        "hf_id": "openai/clip-vit-large-patch14",
        "path": HF_CANDIDATE_ROOT / "openai--clip-vit-large-patch14",
        "notes": ["AIWF uses CLIP-L with Flux/SD-style text conditioning; train only the text side unless a paired-caption job is added."],
    },
    "salesforce-blip-image-captioning-base-text": {
        "label": "BLIP caption text side",
        "size_b": 0.247,
        "family": "blip",
        "target_type": "blip_text",
        "runner_key": "",
        "hf_id": "Salesforce/blip-image-captioning-base",
        "path": HF_CANDIDATE_ROOT / "Salesforce--blip-image-captioning-base",
        "notes": ["BLIP/VLM text-side candidate for caption and prompt-helper work; not image generator training."],
    },
    "wan-umt5-text-encoder": {
        "label": "Wan UMT5 text encoder",
        "size_b": 4.7,
        "family": "umt5",
        "target_type": "text_encoder",
        "runner_key": "",
        "hf_id": "",
        "path": Path(r"F:\AIWF_Studio\models\wan\Wan2.2-TI2V-5B-Diffusers\text_encoder"),
        "notes": ["AIWF Wan uses UMT5-specific text encoders; do not substitute generic Flux or SD3 T5 files."],
        "execution_blocked": True,
    },
    "ltx-gemma-text-encoder": {
        "label": "LTX Gemma text encoder",
        "size_b": 12.0,
        "family": "gemma-text-encoder",
        "target_type": "text_encoder",
        "runner_key": "",
        "hf_id": "google/gemma-3-12b-it-qat-q4_0-unquantized",
        "path": Path(r"F:\AIWF_Studio\models\ltx\text_encoder\gemma-3-12b-it-qat-q4_0-unquantized"),
        "notes": ["AIWF LTX needs Gemma hidden states; this is cataloged for compatibility, not a light default."],
        "execution_blocked": True,
    },
    "qwen2.5-coder-1.5b": {
        "label": "Qwen2.5-Coder 1.5B",
        "size_b": 1.5,
        "family": "qwen-coder",
        "target_type": "causal_lm",
        "runner_key": "qwen2.5-coder-1.5b",
        "hf_id": "",
        "path": HF_CANDIDATE_ROOT / "Qwen--Qwen2.5-Coder-1.5B",
        "notes": ["Current ReTrain runner-ready starter for local SFT/LoRA/QLoRA."],
    },
    "smollm3-3b-instruct": {
        "label": "SmolLM3 3B",
        "size_b": 3.0,
        "family": "smollm3",
        "target_type": "causal_lm",
        "runner_key": "smollm3-3b-instruct",
        "hf_id": "",
        "path": HF_CANDIDATE_ROOT / "HuggingFaceTB--SmolLM3-3B",
        "notes": ["Larger chat/instruct candidate; keep bounded on 16GB-class GPUs."],
    },
    "gemma4-e2b": {
        "label": "Gemma 4 E2B",
        "size_b": 2.0,
        "family": "gemma4",
        "target_type": "causal_lm",
        "runner_key": "gemma4-e2b",
        "hf_id": "",
        "path": HF_CANDIDATE_ROOT / "google--gemma-4-E2B",
        "notes": ["Small Gemma-family chat candidate when present locally."],
    },
}

_processes: dict[str, subprocess.Popen[Any]] = {}


@dataclass(frozen=True)
class DatasetCandidate:
    id: str
    label: str
    path: Path
    source: str
    runner_ready: bool
    kind: str
    counts: dict[str, int]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "path": str(self.path),
            "source": self.source,
            "runnerReady": self.runner_ready,
            "kind": self.kind,
            "counts": self.counts,
            "notes": self.notes,
        }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_runtime_dirs() -> None:
    RUN_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RETRAIN_MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def safe_slug(value: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned[:64] or fallback


def dataset_counts(path: Path) -> dict[str, int]:
    return {
        "train": jsonl_count(path / "train.jsonl"),
        "validation": jsonl_count(path / "validation.jsonl"),
        "test": jsonl_count(path / "test.jsonl"),
        "evalPrompts": jsonl_count(path / "eval_prompts.jsonl"),
    }


def runner_ready(path: Path) -> bool:
    return (path / "train.jsonl").exists() and (path / "validation.jsonl").exists()


def discover_datasets() -> list[dict[str, Any]]:
    datasets: list[DatasetCandidate] = []

    retrain_smoke = ROOT_DIR / "training" / "posttrain_smoke" / "data"
    datasets.append(
        DatasetCandidate(
            id="retrain-posttrain-smoke",
            label="ReTrain post-training smoke pack",
            path=retrain_smoke,
            source="ReTrain",
            runner_ready=runner_ready(retrain_smoke),
            kind="chat_sft_splits",
            counts=dataset_counts(retrain_smoke),
            notes=["Preferred local ReTrain split pack once generated."],
        )
    )

    codex_root = ROOT_DIR / "datasets" / "codex_app_environment"
    codex_manifest = codex_root / "manifest.json"
    codex_counts = {"sft": 0, "eval": 0}
    if codex_manifest.exists():
        try:
            manifest = read_json(codex_manifest)
            files = manifest.get("files", {})
            codex_counts = {
                "sft": jsonl_count(codex_root / str(files.get("sft", ""))),
                "eval": jsonl_count(codex_root / str(files.get("eval", ""))),
            }
        except Exception:
            codex_counts = {"sft": 0, "eval": 0}
    datasets.append(
        DatasetCandidate(
            id="codex-app-environment",
            label="Codex app environment corpus",
            path=codex_root,
            source="ReTrain",
            runner_ready=False,
            kind="source_corpus",
            counts=codex_counts,
            notes=["Validated corpus; needs export to train/validation/test splits before direct runner use."],
        )
    )

    mok_smoke = MOK_ROOT / "training" / "posttrain_bakeoff" / "data"
    if mok_smoke.exists():
        datasets.append(
            DatasetCandidate(
                id="mok-posttrain-bakeoff",
                label="MoK post-training bakeoff pack",
                path=mok_smoke,
                source="MoK-Project",
                runner_ready=runner_ready(mok_smoke),
                kind="chat_sft_splits",
                counts=dataset_counts(mok_smoke),
                notes=["External MoK pack; ReTrain can train against it without becoming part of MoK."],
            )
        )

    mok_core = MOK_ROOT / "training" / "core_mok" / "splits"
    if mok_core.exists():
        datasets.append(
            DatasetCandidate(
                id="mok-core-splits",
                label="MoK core train/validation/test splits",
                path=mok_core,
                source="MoK-Project",
                runner_ready=runner_ready(mok_core),
                kind="chat_sft_splits",
                counts=dataset_counts(mok_core),
                notes=["External MoK core splits; generation eval is skipped unless eval_prompts.jsonl is added."],
            )
        )

    return [dataset.to_dict() for dataset in datasets]


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").exists() or (path / "tokenizer.json").exists():
        return True
    return any(path.glob("*.safetensors"))


def infer_model_size_b(name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[be]?\b", name, flags=re.IGNORECASE)
    if match and "b" in name[match.end() : match.end() + 2].lower():
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    if "e2b" in name.lower():
        return 2.0
    return 1.5


def infer_target_type(name: str) -> str:
    lower = name.lower()
    if "t5" in lower or "umt5" in lower:
        return "seq2seq_t5" if "efficient" in lower or "google--t5" in lower else "text_encoder"
    if "clip" in lower:
        return "clip_text"
    if "blip" in lower:
        return "blip_text"
    if "smollm2" in lower or "pythia" in lower or "tiny" in lower:
        return "prompt_helper_lm"
    return "causal_lm"


def discover_models() -> list[dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model_id, preset in MODEL_PRESETS.items():
        path = Path(preset["path"])
        models[model_id] = {
            "id": model_id,
            "label": preset["label"],
            "path": str(path),
            "family": preset["family"],
            "targetType": preset.get("target_type", "causal_lm"),
            "sizeB": preset["size_b"],
            "runnerKey": preset["runner_key"],
            "hfId": preset.get("hf_id", ""),
            "exists": path.exists(),
            "source": "known-local",
            "notes": preset.get("notes", []),
            "executionBlocked": bool(preset.get("execution_blocked", False)),
        }

    for root in (RETRAIN_MODEL_ROOT, HF_CANDIDATE_ROOT):
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not _looks_like_model_dir(child):
                continue
            model_id = safe_slug(child.name.lower(), "local-model")
            models.setdefault(
                model_id,
                {
                    "id": model_id,
                    "label": child.name,
                    "path": str(child),
                    "family": "local",
                    "targetType": infer_target_type(child.name),
                    "sizeB": infer_model_size_b(child.name),
                    "runnerKey": "",
                    "hfId": "",
                    "exists": True,
                    "source": str(root),
                    "notes": ["Discovered local model folder."],
                    "executionBlocked": False,
                },
            )
    return sorted(models.values(), key=lambda item: (not item["exists"], item["label"].lower()))


def dependency_status() -> list[dict[str, Any]]:
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("tensorboard", "TensorBoard"),
    ]
    return [
        {"package": package, "label": label, "available": importlib.util.find_spec(package) is not None}
        for package, label in packages
    ]


def hardware_summary() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except Exception:
        return {"gpu": "unknown", "vramTotalMb": None, "vramFreeMb": None}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return {"gpu": "unknown", "vramTotalMb": None, "vramFreeMb": None}
    return {
        "gpu": parts[0],
        "vramTotalMb": int(float(parts[1])),
        "vramFreeMb": int(float(parts[2])),
    }


def default_config() -> dict[str, Any]:
    datasets = discover_datasets()
    models = discover_models()
    default_dataset = next((item for item in datasets if item["runnerReady"]), datasets[0])
    default_model = next(
        (item for item in models if item["id"] == "qwen2.5-coder-1.5b" and item["exists"]),
        next((item for item in models if item["exists"]), models[0]),
    )
    return {
        "name": "Starter local training run",
        "datasetId": default_dataset["id"],
        "datasetPath": default_dataset["path"],
        "modelId": default_model["id"],
        "modelPath": default_model["path"] if default_model["exists"] else "",
        "targetType": default_model.get("targetType", "causal_lm"),
        "datasetFormat": "auto",
        "method": "qlora",
        "precision": "4bit",
        "contextLength": 2048,
        "microBatchSize": 1,
        "gradientAccumulationSteps": 8,
        "loraRank": 16,
        "learningRate": 2e-5,
        "maxSteps": 1,
        "maxTrainRecords": 8,
        "maxEvalRecords": 4,
        "maxTestRecords": 2,
        "maxVramGb": 16.0,
        "freeVramGb": 14.0,
        "gradientCheckpointing": True,
        "tensorboard": True,
        "saveFinal": False,
        "saveCheckpoints": False,
        "dryRun": True,
        "confirmed": False,
    }


def normalize_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = {**default_config(), **(payload or {})}
    normalized = {
        "name": str(raw.get("name") or "Training run"),
        "datasetId": str(raw.get("datasetId") or ""),
        "datasetPath": str(raw.get("datasetPath") or ""),
        "modelId": str(raw.get("modelId") or ""),
        "modelPath": str(raw.get("modelPath") or ""),
        "targetType": str(raw.get("targetType") or "causal_lm").lower(),
        "datasetFormat": str(raw.get("datasetFormat") or "auto").lower(),
        "method": str(raw.get("method") or "qlora").lower(),
        "precision": str(raw.get("precision") or "4bit").lower(),
        "contextLength": _int(raw.get("contextLength"), 2048, 512, 32768),
        "microBatchSize": _int(raw.get("microBatchSize"), 1, 1, 64),
        "gradientAccumulationSteps": _int(raw.get("gradientAccumulationSteps"), 8, 1, 256),
        "loraRank": _int(raw.get("loraRank"), 16, 1, 256),
        "learningRate": _float(raw.get("learningRate"), 2e-5, 1e-7, 1e-2),
        "maxSteps": _int(raw.get("maxSteps"), 1, 0, 1000000),
        "maxTrainRecords": _int(raw.get("maxTrainRecords"), 8, 1, 1000000),
        "maxEvalRecords": _int(raw.get("maxEvalRecords"), 4, 1, 1000000),
        "maxTestRecords": _int(raw.get("maxTestRecords"), 2, 0, 1000000),
        "maxVramGb": _float(raw.get("maxVramGb"), 16.0, 1.0, 256.0),
        "freeVramGb": _float(raw.get("freeVramGb"), 14.0, 0.1, 256.0),
        "gradientCheckpointing": _bool(raw.get("gradientCheckpointing"), True),
        "tensorboard": _bool(raw.get("tensorboard"), True),
        "saveFinal": _bool(raw.get("saveFinal"), False),
        "saveCheckpoints": _bool(raw.get("saveCheckpoints"), False),
        "dryRun": _bool(raw.get("dryRun"), True),
        "confirmed": _bool(raw.get("confirmed"), False),
    }
    if normalized["method"] not in METHOD_TO_RUNNER:
        normalized["method"] = "qlora"
    if normalized["precision"] not in {"4bit", "8bit", "bf16", "fp16"}:
        normalized["precision"] = "4bit"
    if normalized["targetType"] not in SUPPORTED_TARGET_TYPES:
        normalized["targetType"] = "causal_lm"
    if normalized["datasetFormat"] not in DATASET_FORMATS:
        normalized["datasetFormat"] = "auto"
    return normalized


def _int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_dataset(config: dict[str, Any]) -> dict[str, Any]:
    datasets = discover_datasets()
    if config["datasetId"]:
        match = next((item for item in datasets if item["id"] == config["datasetId"]), None)
        if match:
            config["datasetPath"] = match["path"]
            return match
    path = Path(config["datasetPath"]).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return DatasetCandidate(
        id="custom",
        label=path.name or "Custom dataset",
        path=path,
        source="custom",
        runner_ready=runner_ready(path),
        kind="chat_sft_splits",
        counts=dataset_counts(path),
        notes=["Custom dataset path from request."],
    ).to_dict()


def resolve_model(config: dict[str, Any]) -> dict[str, Any]:
    models = discover_models()
    if config["modelId"]:
        match = next((item for item in models if item["id"] == config["modelId"]), None)
        if match:
            if not config["modelPath"] and match.get("exists"):
                config["modelPath"] = str(match["path"])
            return match
    path = Path(config["modelPath"]).expanduser()
    if config["modelPath"] and not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return {
        "id": "custom",
        "label": path.name if config["modelPath"] else "Custom model",
        "path": str(path) if config["modelPath"] else "",
        "family": "custom",
        "targetType": config.get("targetType") or infer_target_type(path.name if config["modelPath"] else ""),
        "sizeB": infer_model_size_b(path.name if config["modelPath"] else ""),
        "runnerKey": "",
        "hfId": "",
        "exists": path.exists() if config["modelPath"] else False,
        "source": "custom",
        "notes": ["Custom model path from request."],
        "executionBlocked": False,
    }


def estimate_vram(config: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    size_b = float(model.get("sizeB") or 1.5)
    target_type = str(config.get("targetType") or model.get("targetType") or "causal_lm")
    precision_factor = {"4bit": 0.58, "8bit": 1.05, "bf16": 2.05, "fp16": 2.05}[config["precision"]]
    base_weights = size_b * precision_factor
    adapter = max(0.12, size_b * config["loraRank"] * 0.012) if config["method"] != "sft" else 0.0
    optimizer_base = size_b * (1.8 if config["method"] in {"sft", "full_sft"} else 0.22)
    optimizer = optimizer_base
    checkpoint_factor = 0.58 if config["gradientCheckpointing"] else 1.0
    qlora_factor = 0.74 if config["method"] == "qlora" or config["precision"] == "4bit" else 1.0
    activations = (
        (config["contextLength"] / 1024)
        * config["microBatchSize"]
        * max(0.22, size_b * 0.2)
        * checkpoint_factor
        * qlora_factor
    )
    dataloader = 0.45
    safety = 1.2 if config["maxVramGb"] <= 16 else 1.6
    if target_type in {"seq2seq_t5", "text_encoder", "clip_text", "blip_text"}:
        activations *= 0.72
        adapter *= 0.82
        safety = max(0.75, safety - 0.35)
    estimated = base_weights + adapter + optimizer + activations + dataloader + safety
    limit = min(config["maxVramGb"], config["freeVramGb"] + 1.2)
    headroom = limit - estimated
    fit_state = "safe" if headroom >= 2 else "tight" if headroom >= 0 else "unsafe"
    warnings: list[str] = []
    if fit_state == "unsafe":
        warnings.append("Estimated VRAM exceeds the current safety limit.")
    if config["precision"] in {"bf16", "fp16"} and size_b >= 7 and config["maxVramGb"] <= 16:
        warnings.append("Full precision 7B training is not a safe 16GB default.")
    if not config["gradientCheckpointing"] and config["maxVramGb"] <= 24:
        warnings.append("Gradient checkpointing is off; activation memory will be higher.")
    if target_type not in RUNNER_READY_TARGETS:
        warnings.append("This target is cataloged for lightweight planning; the current runner cannot execute it yet.")
    return {
        "fitState": fit_state,
        "estimatedGb": round(estimated, 2),
        "limitGb": round(limit, 2),
        "headroomGb": round(headroom, 2),
        "percent": int(max(1, min(100, round((estimated / max(1.0, limit)) * 100)))),
        "warnings": warnings,
        "breakdown": [
            {"item": "Base weights", "gb": round(base_weights, 2), "detail": config["precision"]},
            {"item": "Adapter", "gb": round(adapter, 2), "detail": f"rank {config['loraRank']}"},
            {"item": "Optimizer", "gb": round(optimizer, 2), "detail": config["method"]},
            {
                "item": "Activations",
                "gb": round(activations, 2),
                "detail": f"{config['contextLength']} ctx x {config['microBatchSize']} batch",
            },
            {"item": "Dataloader", "gb": round(dataloader, 2), "detail": "reserved"},
            {"item": "Safety margin", "gb": round(safety, 2), "detail": f"{target_type} local profile"},
        ],
    }


def gate(name: str, ok: bool, detail: str, *, fail_state: str = "blocked") -> dict[str, str]:
    return {"gate": name, "state": "ready" if ok else fail_state, "detail": detail}


def target_detail(target_type: str) -> str:
    return {
        "causal_lm": "Runner-ready Causal LM chat/SFT target.",
        "prompt_helper_lm": "Runner-ready small decoder LM target for prompt helper jobs.",
        "seq2seq_t5": "Runner-ready lightweight Seq2Seq/T5 target.",
        "text_encoder": "Runner-ready masked-LM text encoder target; AIWF component encoders may still be catalog-only.",
        "clip_text": "Needs a CLIP text-side contrastive runner before execution.",
        "blip_text": "Needs a BLIP/VLM text-side runner before execution.",
    }.get(target_type, "Unknown training target.")


def build_runner_argv(config: dict[str, Any], dataset: dict[str, Any], model: dict[str, Any], output_root: Path) -> list[str]:
    target_type = str(config.get("targetType") or model.get("targetType") or "causal_lm")
    if target_type in {"seq2seq_t5", "text_encoder"}:
        model_path = Path(config["modelPath"] or model.get("path") or "").expanduser()
        return [
            sys.executable,
            str(TEXT_TARGET_RUNNER_SCRIPT),
            "--target-type",
            target_type,
            "--data-dir",
            str(Path(dataset["path"])),
            "--output-root",
            str(output_root),
            "--model-path",
            str(model_path),
            "--model-name",
            safe_slug(model_path.name or str(model.get("id") or "text-target"), "text-target"),
            "--dataset-format",
            config["datasetFormat"],
            "--max-seq-length",
            str(config["contextLength"]),
            "--batch-size",
            str(config["microBatchSize"]),
            "--gradient-accumulation-steps",
            str(config["gradientAccumulationSteps"]),
            "--learning-rate",
            str(config["learningRate"]),
            "--precision",
            config["precision"] if config["precision"] in {"bf16", "fp16"} else "bf16",
            "--max-steps",
            str(config["maxSteps"] or 1),
            "--max-train-records",
            str(config["maxTrainRecords"]),
            "--max-eval-records",
            str(config["maxEvalRecords"]),
            *(["--no-gradient-checkpointing"] if not config["gradientCheckpointing"] else []),
            *(["--no-save-final"] if not config["saveFinal"] else []),
            *(["--no-save-checkpoints"] if not config["saveCheckpoints"] else []),
            *(["--dry-run"] if config["dryRun"] else []),
        ]

    runner_method = METHOD_TO_RUNNER[config["method"]]
    precision = config["precision"] if config["precision"] in {"bf16", "fp16"} else "bf16"
    argv = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--data-dir",
        str(Path(dataset["path"])),
        "--output-root",
        str(output_root),
        "--method",
        runner_method,
        "--max-seq-length",
        str(config["contextLength"]),
        "--batch-size",
        str(config["microBatchSize"]),
        "--gradient-accumulation-steps",
        str(config["gradientAccumulationSteps"]),
        "--learning-rate",
        str(config["learningRate"]),
        "--precision",
        precision,
        "--lora-r",
        str(config["loraRank"]),
        "--max-steps",
        str(config["maxSteps"] or 1),
        "--max-train-records",
        str(config["maxTrainRecords"]),
        "--max-eval-records",
        str(config["maxEvalRecords"]),
        "--max-test-records",
        str(config["maxTestRecords"]),
    ]
    runner_key = str(model.get("runnerKey") or "")
    if config["modelPath"]:
        model_path = Path(config["modelPath"]).expanduser()
        argv.extend(["--model-path", str(model_path), "--model-name", safe_slug(model_path.name, model["id"])])
    elif runner_key:
        argv.extend(["--model", runner_key])
    else:
        argv.extend(["--model", "qwen2.5-coder-1.5b"])
    if not config["gradientCheckpointing"]:
        argv.append("--no-gradient-checkpointing")
    if not config["saveFinal"]:
        argv.append("--no-save-final")
    if not config["saveCheckpoints"]:
        argv.append("--no-save-checkpoints")
    if not dataset["counts"].get("evalPrompts"):
        argv.append("--skip-generation")
    if config["dryRun"]:
        argv.append("--dry-run")
    return argv


def plan_training_run(payload: dict[str, Any] | None = None, *, run_id: str | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = normalize_config(payload)
    dataset = resolve_dataset(config)
    model = resolve_model(config)
    if config["modelId"] and model.get("id") != "custom":
        config["targetType"] = str(model.get("targetType") or config["targetType"])
    estimate = estimate_vram(config, model)
    deps = dependency_status()
    output_root = RUN_OUTPUT_ROOT / (run_id or "planned")
    argv = build_runner_argv(config, dataset, model, output_root)

    gates = [
        gate(
            "Runner script",
            (TEXT_TARGET_RUNNER_SCRIPT if config["targetType"] in {"seq2seq_t5", "text_encoder"} else RUNNER_SCRIPT).exists(),
            str(TEXT_TARGET_RUNNER_SCRIPT if config["targetType"] in {"seq2seq_t5", "text_encoder"} else RUNNER_SCRIPT),
        ),
        gate("Dataset splits", bool(dataset["runnerReady"]), dataset["path"]),
        gate("Output folder", output_root.parent.exists(), str(output_root.parent)),
        gate(
            "Training target",
            config["targetType"] in RUNNER_READY_TARGETS and not bool(model.get("executionBlocked")),
            target_detail(config["targetType"]),
            fail_state="blocked",
        ),
        gate("VRAM estimate", estimate["fitState"] != "unsafe", estimate["fitState"]),
        gate("Confirmation", config["dryRun"] or config["confirmed"], "required before write-heavy training"),
    ]
    if model.get("executionBlocked"):
        gates.append(gate("Model execution", False, "AIWF component encoder is catalog-only for this pass."))
    if config["modelPath"]:
        gates.append(
            gate(
                "Base model path",
                Path(config["modelPath"]).expanduser().exists(),
                config["modelPath"],
                fail_state="warning" if config["dryRun"] else "blocked",
            )
        )
    else:
        gates.append(gate("Base model selection", bool(model.get("runnerKey")), model["label"]))
    if config["method"] in {"lora", "qlora"}:
        for package in ("peft", "accelerate"):
            dep = next((item for item in deps if item["package"] == package), None)
            gates.append(
                gate(
                    dep["label"] if dep else package,
                    bool(dep and dep["available"]),
                    "adapter training dependency",
                    fail_state="warning" if config["dryRun"] else "blocked",
                )
            )
    if config["method"] == "qlora" or config["precision"] in {"4bit", "8bit"}:
        dep = next((item for item in deps if item["package"] == "bitsandbytes"), None)
        gates.append(
            gate(
                "BitsAndBytes",
                bool(dep and dep["available"]),
                "4-bit/8-bit adapter dependency",
                fail_state="warning" if config["dryRun"] else "blocked",
            )
        )
    if config["tensorboard"]:
        dep = next((item for item in deps if item["package"] == "tensorboard"), None)
        gates.append(gate("TensorBoard", bool(dep and dep["available"]), str(output_root)))

    blocked = any(item["state"] == "blocked" for item in gates)
    warning = any(item["state"] == "warning" for item in gates) or estimate["fitState"] == "tight"
    status = "blocked" if blocked else "warning" if warning else "ready"

    return {
        "status": status,
        "startEnabled": not blocked,
        "config": config,
        "dataset": dataset,
        "model": model,
        "estimate": estimate,
        "dependencies": deps,
        "gates": gates,
        "argv": argv,
        "command": powershell_command(argv),
        "outputs": {
            "outputRoot": str(output_root),
            "stateRoot": str(RUN_STATE_ROOT / (run_id or "planned")),
            "tensorboardLogdir": str(output_root),
        },
        "notes": [
            "Dry runs validate paths and write receipts without loading model weights.",
            "Real training requires confirmation and uses the copied ReTrain runner.",
            "Prompt-helper decoder LMs, T5/Seq2Seq, and generic masked-LM text encoders are runner-ready.",
            "CLIP, BLIP, and large AIWF component encoders are cataloged until a matching paired-data runner is added.",
            "Video and image model training are intentionally outside this ReTrain pass.",
            "MoK datasets are treated as external inputs; ReTrain remains the trainer repo.",
        ],
    }


def powershell_command(argv: list[str]) -> str:
    def quote(part: str) -> str:
        text = str(part)
        if not text:
            return "''"
        if any(char.isspace() for char in text) or any(char in text for char in ("'", '"', "$", "`")):
            return "'" + text.replace("'", "''") + "'"
        return text

    return " ".join(quote(part) for part in argv)


def create_run(payload: dict[str, Any] | None = None, *, execute: bool = False) -> dict[str, Any]:
    ensure_runtime_dirs()
    config = normalize_config(payload)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}-{safe_slug(config['name'])}"
    plan = plan_training_run(config, run_id=run_id)
    state_dir = RUN_STATE_ROOT / run_id
    output_root = RUN_OUTPUT_ROOT / run_id
    state_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "runner.log"
    receipt_path = state_dir / "run.json"

    status = "planned"
    process: dict[str, Any] | None = None
    if execute and not plan["startEnabled"]:
        status = "blocked"
    elif execute and config["dryRun"]:
        result = subprocess.run(
            plan["argv"],
            cwd=str(ROOT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
        status = "dry_run_completed" if result.returncode == 0 else "dry_run_failed"
        process = {"exitCode": result.returncode}
    elif execute:
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                plan["argv"],
                cwd=str(ROOT_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        finally:
            log_handle.close()
        _processes[run_id] = proc
        status = "running"
        process = {"pid": proc.pid}

    record = {
        "schemaVersion": "retrain-training-run-v1",
        "id": run_id,
        "name": config["name"],
        "status": status,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "executeRequested": execute,
        "plan": plan,
        "process": process,
        "paths": {
            "stateDir": str(state_dir),
            "outputRoot": str(output_root),
            "logPath": str(log_path),
            "receiptPath": str(receipt_path),
        },
    }
    write_json(receipt_path, record)
    return refresh_run(record)


def list_runs() -> list[dict[str, Any]]:
    ensure_runtime_dirs()
    runs = []
    for path in sorted(RUN_STATE_ROOT.glob("*/run.json"), reverse=True):
        try:
            runs.append(refresh_run(read_json(path)))
        except Exception:
            continue
    return runs


def get_run(run_id: str) -> dict[str, Any]:
    path = RUN_STATE_ROOT / safe_slug(run_id) / "run.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return refresh_run(read_json(path))


def refresh_run(record: dict[str, Any]) -> dict[str, Any]:
    run_id = record["id"]
    proc = _processes.get(run_id)
    if proc and record.get("status") == "running":
        exit_code = proc.poll()
        if exit_code is not None:
            record["status"] = "completed" if exit_code == 0 else "failed"
            record["process"] = {"pid": proc.pid, "exitCode": exit_code}
            record["updatedAt"] = now_iso()
            write_json(RUN_STATE_ROOT / run_id / "run.json", record)
            _processes.pop(run_id, None)
    record["metrics"] = collect_run_metrics(Path(record["paths"]["outputRoot"]))
    record["logTail"] = read_log_tail(Path(record["paths"]["logPath"]))
    return record


def collect_run_metrics(output_root: Path) -> dict[str, Any]:
    metrics_files = sorted(output_root.rglob("metrics.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    event_files = sorted(output_root.rglob("events.out.tfevents*"))
    latest: dict[str, Any] | None = None
    if metrics_files:
        try:
            latest = read_json(metrics_files[0])
        except Exception:
            latest = None
    return {
        "metricsFileCount": len(metrics_files),
        "eventFileCount": len(event_files),
        "latestMetricsPath": str(metrics_files[0]) if metrics_files else "",
        "latest": compact_metrics(latest or {}),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    train = metrics.get("train_metrics") if isinstance(metrics.get("train_metrics"), dict) else {}
    eval_metrics = metrics.get("eval_metrics") if isinstance(metrics.get("eval_metrics"), dict) else {}
    generation = metrics.get("generation_metrics") if isinstance(metrics.get("generation_metrics"), dict) else {}
    cuda = metrics.get("cuda") if isinstance(metrics.get("cuda"), dict) else {}
    return {
        "model": metrics.get("model"),
        "method": metrics.get("method"),
        "trainRows": metrics.get("train_rows"),
        "validationRows": metrics.get("validation_rows"),
        "maxSteps": metrics.get("max_steps"),
        "runtimeSeconds": metrics.get("train_runtime_seconds"),
        "trainLoss": train.get("train_loss") or train.get("loss"),
        "evalLoss": eval_metrics.get("eval_loss"),
        "evalPerplexity": metrics.get("eval_perplexity"),
        "generationHitRate": generation.get("required_tag_hit_rate") if generation else None,
        "peakAllocatedGb": cuda.get("peak_allocated_gb") if cuda else None,
    }


def read_log_tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned = ANSI_RE.sub("", text).replace("\r", "\n")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines)[-limit:]


def stop_run(run_id: str) -> dict[str, Any]:
    record = get_run(run_id)
    proc = _processes.get(run_id)
    if proc and proc.poll() is None:
        proc.terminate()
        record["status"] = "stopping"
        record["updatedAt"] = now_iso()
    else:
        pid = (record.get("process") or {}).get("pid")
        if pid and record.get("status") == "running":
            try:
                os.kill(int(pid), signal.SIGTERM)
                record["status"] = "stopping"
            except OSError as exc:
                record["status"] = "stop_failed"
                record["stopError"] = str(exc)
        else:
            record["status"] = "not_running"
    write_json(RUN_STATE_ROOT / run_id / "run.json", record)
    return refresh_run(record)


def resume_run(run_id: str, *, execute: bool = False) -> dict[str, Any]:
    original = get_run(run_id)
    config = dict(original["plan"]["config"])
    config["name"] = f"{original['name']} resume"
    config["dryRun"] = True if not execute else config.get("dryRun", True)
    resumed = create_run(config, execute=execute)
    resumed["resumedFrom"] = run_id
    write_json(RUN_STATE_ROOT / resumed["id"] / "run.json", resumed)
    return resumed


def training_overview() -> dict[str, Any]:
    ensure_runtime_dirs()
    return {
        "schemaVersion": "retrain-training-overview-v1",
        "repo": {
            "root": str(ROOT_DIR),
            "mokRoot": str(MOK_ROOT),
            "runner": str(RUNNER_SCRIPT),
            "stateRoot": str(RUN_STATE_ROOT),
            "outputRoot": str(RUN_OUTPUT_ROOT),
        },
        "hardware": hardware_summary(),
        "datasets": discover_datasets(),
        "models": discover_models(),
        "dependencies": dependency_status(),
        "defaultConfig": default_config(),
        "runs": list_runs(),
    }
