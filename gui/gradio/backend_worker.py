from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse
from urllib.request import urlretrieve

from backend.dependency_health import clear_dependency_probe_cache, inspect_packages


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DATASET_VERSION = "qwen3.5-2b-Research-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "training" / "gui_runs"
DEFAULT_DATASET_PATH = ROOT / "training" / "posttrain_bakeoff" / "data"
DEFAULT_RECEIPT_ROOT = ROOT / "training" / "gui_runs" / "receipts"
DEFAULT_MODEL_ROOT = Path(os.environ.get("RETRAIN_MODEL_ROOT", str(ROOT / "models")))
DEFAULT_ENGINE_ROOT = ROOT / "engines"
DEFAULT_DATASET_EXPORT_ROOT = ROOT / "training" / "gui_runs" / "datasets"

TrainingMethod = Literal["sft", "full_sft", "lora", "qlora", "dpo", "grpo", "reward_model", "kto", "rloo", "ppo"]
PrecisionMode = Literal["bf16", "fp16", "8bit", "4bit"]
FitState = Literal["safe", "tight", "unsafe"]
BackendTarget = Literal["ollama", "llama.cpp", "vllm"]
OptimizerMode = Literal["paged_adamw_8bit", "adamw_8bit", "adamw_torch"]
SchedulerMode = Literal["cosine", "linear", "constant"]
DatasetFormat = Literal["chatml", "alpaca", "sharegpt", "qwen-vl", "preference", "tool-calls"]
TuneScope = Literal["all", "last_n_layers", "lm_head"]
RewardKind = Literal["heuristic", "length", "tag_match"]
ProcessFactory = Callable[..., subprocess.Popen[Any]]

_ACTIVE_JOBS: dict[int, dict[str, Any]] = {}

SUPERVISED_METHODS = {"sft", "full_sft", "lora", "qlora"}
TRL_METHODS = {"dpo", "grpo", "reward_model", "kto", "rloo"}
RUNNER_METHODS = SUPERVISED_METHODS
BLOCKED_METHODS = TRL_METHODS | {"ppo"}


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "lfm2.5-350m": {"label": "LFM2.5 350M", "size_b": 0.35, "family": "lfm2", "path": str(DEFAULT_MODEL_ROOT / "LiquidAI--LFM2.5-350M")},
    "qwen2.5-coder-1.5b": {"label": "Qwen2.5-Coder 1.5B", "size_b": 1.5, "family": "qwen", "path": str(DEFAULT_MODEL_ROOT / "Qwen--Qwen2.5-Coder-1.5B-Instruct")},
    "smollm3-3b": {"label": "SmolLM3 3B", "size_b": 3.0, "family": "smollm", "path": str(DEFAULT_MODEL_ROOT / "HuggingFaceTB--SmolLM3-3B")},
    "qwen2.5-7b": {"label": "Qwen2.5 7B", "size_b": 7.0, "family": "qwen"},
    "qwen-vl-4b": {"label": "Qwen-VL 4B", "size_b": 4.0, "family": "qwen-vl"},
}

MODEL_REPOSITORIES: list[dict[str, str]] = [
    {
        "label": "Qwen2.5-Coder 1.5B Instruct",
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "fit": "starter QLoRA",
    },
    {
        "label": "Qwen2.5 1.5B Instruct",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        "fit": "starter chat",
    },
    {
        "label": "Qwen2.5 3B Instruct",
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
        "fit": "better quality",
    },
    {
        "label": "SmolLM3 3B",
        "repo": "HuggingFaceTB/SmolLM3-3B",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
        "fit": "small general model",
    },
]

ENGINE_PROFILES: dict[str, dict[str, Any]] = {
    "qlora": {
        "label": "QLoRA consumer training",
        "packages": [
            "torch",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "trl",
            "bitsandbytes",
            "sentencepiece",
            "safetensors",
            "huggingface_hub",
            "tensorboard",
        ],
    }
}

MODEL_FILE_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".model",
    ".json",
    ".tokenizer",
}


@dataclass(frozen=True)
class TrainingConfig:
    dataset_version: str = ACTIVE_DATASET_VERSION
    model_key: str = "qwen2.5-coder-1.5b"
    model_path: str = ""
    backend_target: BackendTarget = "ollama"
    dataset_path: str = str(DEFAULT_DATASET_PATH)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    method: TrainingMethod = "qlora"
    tune_scope: TuneScope = "last_n_layers"
    last_n_layers: int = 4
    beta: float = 0.1
    num_generations: int = 2
    reward_kind: RewardKind = "heuristic"
    precision: PrecisionMode = "4bit"
    dataset_format: DatasetFormat = "chatml"
    context_length: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    learning_rate: float = 2e-5
    epochs: float = 1.0
    optimizer: OptimizerMode = "paged_adamw_8bit"
    scheduler: SchedulerMode = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    save_every_steps: int = 100
    eval_every_steps: int = 100
    max_steps: int = 0
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    trust_remote_code: bool = False
    qlora: bool = True
    bitsandbytes: bool = True
    unsloth: bool = False
    cpu_offload: bool = False
    low_vram_mode: bool = True
    sequence_packing: bool = True
    tensorboard: bool = True
    tensorboard_logdir: str = str(DEFAULT_OUTPUT_ROOT / "tensorboard")
    max_vram_gb: float = 16.0
    free_vram_gb: float = 14.8
    confirmed: bool = False
    dry_run: bool = True


@dataclass(frozen=True)
class VramEstimate:
    fit_state: FitState
    estimated_gb: float
    limit_gb: float
    headroom_gb: float
    percent: int
    breakdown: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True)
class ValidationResult:
    status: Literal["ready", "blocked", "warning"]
    start_enabled: bool
    config: dict[str, Any]
    estimate: dict[str, Any]
    dependencies: list[dict[str, Any]]
    gates: list[dict[str, Any]]
    notes: list[str]


def normalize_payload(payload: dict[str, Any] | None) -> TrainingConfig:
    payload = payload or {}
    defaults = asdict(TrainingConfig())
    normalized: dict[str, Any] = {}
    for key, default in defaults.items():
        value = payload.get(key, default)
        if isinstance(default, bool):
            normalized[key] = _to_bool(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            normalized[key] = int(float(value))
        elif isinstance(default, float):
            normalized[key] = float(value)
        else:
            normalized[key] = str(value)

    if normalized["model_key"] not in MODEL_PRESETS:
        normalized["model_key"] = "qwen2.5-coder-1.5b"
    if not normalized["model_path"].strip():
        preset_path = str(MODEL_PRESETS[normalized["model_key"]].get("path", ""))
        if preset_path and Path(preset_path).is_dir():
            normalized["model_path"] = preset_path
    if "dataset_path" not in payload or not str(payload.get("dataset_path", "")).strip():
        normalized["dataset_path"] = str(_latest_default_dataset_path())
    if normalized["backend_target"] not in {"ollama", "llama.cpp", "vllm"}:
        normalized["backend_target"] = "ollama"
    if normalized["method"] not in RUNNER_METHODS | BLOCKED_METHODS:
        normalized["method"] = "qlora"
    if normalized["tune_scope"] not in {"all", "last_n_layers", "lm_head"}:
        normalized["tune_scope"] = "last_n_layers"
    if normalized["reward_kind"] not in {"heuristic", "length", "tag_match"}:
        normalized["reward_kind"] = "heuristic"
    if normalized["precision"] not in {"bf16", "fp16", "8bit", "4bit"}:
        normalized["precision"] = "4bit"
    if normalized["dataset_format"] not in {"chatml", "alpaca", "sharegpt", "qwen-vl", "preference", "tool-calls"}:
        normalized["dataset_format"] = "chatml"
    if normalized["optimizer"] not in {"paged_adamw_8bit", "adamw_8bit", "adamw_torch"}:
        normalized["optimizer"] = "paged_adamw_8bit"
    if normalized["scheduler"] not in {"cosine", "linear", "constant"}:
        normalized["scheduler"] = "cosine"
    normalized["context_length"] = _clamp(normalized["context_length"], 512, 32768)
    normalized["micro_batch_size"] = _clamp(normalized["micro_batch_size"], 1, 64)
    normalized["gradient_accumulation_steps"] = _clamp(normalized["gradient_accumulation_steps"], 1, 256)
    normalized["lora_rank"] = _clamp(normalized["lora_rank"], 1, 256)
    normalized["last_n_layers"] = _clamp(normalized["last_n_layers"], 1, 128)
    normalized["beta"] = max(0.0, min(2.0, normalized["beta"]))
    normalized["num_generations"] = _clamp(normalized["num_generations"], 2, 16)
    normalized["warmup_ratio"] = max(0.0, min(0.5, normalized["warmup_ratio"]))
    normalized["weight_decay"] = max(0.0, min(1.0, normalized["weight_decay"]))
    normalized["save_every_steps"] = _clamp(normalized["save_every_steps"], 1, 100000)
    normalized["eval_every_steps"] = _clamp(normalized["eval_every_steps"], 1, 100000)
    normalized["max_steps"] = _clamp(normalized["max_steps"], 0, 10000000)
    normalized["max_vram_gb"] = max(1.0, normalized["max_vram_gb"])
    normalized["free_vram_gb"] = max(0.1, normalized["free_vram_gb"])
    if not normalized["tensorboard_logdir"].strip():
        normalized["tensorboard_logdir"] = str(Path(normalized["output_root"]) / "tensorboard")
    return TrainingConfig(**normalized)


def _latest_default_dataset_path() -> Path:
    export_root = DEFAULT_DATASET_EXPORT_ROOT
    if export_root.is_dir():
        candidates = [path for path in export_root.iterdir() if path.is_dir() and (path / "train.jsonl").is_file()]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return DEFAULT_DATASET_PATH


def ensure_mvp_folders() -> dict[str, str]:
    folders = {
        "models": DEFAULT_MODEL_ROOT,
        "datasets": ROOT / "datasets",
        "engines": DEFAULT_ENGINE_ROOT,
        "outputs": DEFAULT_OUTPUT_ROOT,
        "receipts": DEFAULT_RECEIPT_ROOT,
        "dataset_exports": DEFAULT_DATASET_EXPORT_ROOT,
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return {name: str(path) for name, path in folders.items()}


def model_repository_rows() -> list[list[str]]:
    return [[item["label"], item["fit"], item["repo"], item["url"]] for item in MODEL_REPOSITORIES]


def model_repository_markdown() -> str:
    lines = ["### Quick Model Links"]
    for item in MODEL_REPOSITORIES:
        lines.append(f"- [{item['label']}]({item['url']}) - {item['fit']}")
    return "\n".join(lines)


def model_inventory(model_root: str | None = None) -> dict[str, Any]:
    root = _resolve_path(model_root or str(DEFAULT_MODEL_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, str]] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            model_files = [path for path in child.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_FILE_SUFFIXES]
            if model_files:
                candidates.append(
                    {
                        "name": child.name,
                        "type": "folder",
                        "path": str(child),
                        "files": str(len(model_files)),
                    }
                )
        elif child.is_file() and child.suffix.lower() in MODEL_FILE_SUFFIXES:
            candidates.append({"name": child.name, "type": child.suffix.lower(), "path": str(child), "files": "1"})
    return {"root": str(root), "count": len(candidates), "models": candidates}


def engine_status(engine_key: str = "qlora") -> dict[str, Any]:
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["qlora"])
    packages = inspect_packages((package, package) for package in profile["packages"])
    missing = [package["package"] for package in packages if not package["available"]]
    marker = DEFAULT_ENGINE_ROOT / engine_key / "installed.json"
    return {
        "engine": engine_key,
        "label": profile["label"],
        "ready": not missing,
        "missing": missing,
        "packages": packages,
        "marker": str(marker),
    }


def install_engine(engine_key: str = "qlora", *, execute: bool = False) -> dict[str, Any]:
    ensure_mvp_folders()
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["qlora"])
    status_before = engine_status(engine_key)
    engine_dir = DEFAULT_ENGINE_ROOT / engine_key
    engine_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "engine": engine_key,
        "label": profile["label"],
        "execute_requested": execute,
        "missing_before": status_before["missing"],
        "status": "ready" if status_before["ready"] else "needs_install",
        "packages": profile["packages"],
    }
    if execute and status_before["missing"]:
        commands = [
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-cuda-cu132.txt")],
            [sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[training]"],
        ]
        command_results = []
        for argv in commands:
            result = subprocess.run(
                argv,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            command_results.append({"exit_code": result.returncode, "output_tail": result.stdout[-4000:]})
            if result.returncode != 0:
                break
        receipt["pip_commands"] = command_results
        receipt["pip_exit_code"] = command_results[-1]["exit_code"]
        receipt["pip_output_tail"] = command_results[-1]["output_tail"]
        receipt["status"] = "installed" if all(item["exit_code"] == 0 for item in command_results) else "failed"
    elif execute:
        receipt["status"] = "already_ready"

    if execute:
        clear_dependency_probe_cache()
    status_after = engine_status(engine_key)
    receipt["missing_after"] = status_after["missing"]
    if status_after["ready"]:
        marker = Path(status_after["marker"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {"receipt": receipt, "status": status_after}


def stage_model_download(download_ref: str, model_root: str | None = None, alias: str = "", *, execute: bool = True) -> dict[str, Any]:
    ensure_mvp_folders()
    root = _resolve_path(model_root or str(DEFAULT_MODEL_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    ref = (download_ref or "").strip()
    if not ref:
        return {"status": "blocked", "message": "Add a model repository link first.", "destination": str(root)}

    parsed = _parse_model_ref(ref)
    destination_name = _safe_name(alias.strip() or parsed["name"])
    destination = root / destination_name
    receipt = {
        "schema_version": 1,
        "download_ref": ref,
        "kind": parsed["kind"],
        "repo_id": parsed.get("repo_id", ""),
        "destination": str(destination),
        "execute_requested": execute,
    }
    if not execute:
        receipt["status"] = "planned"
        return {"status": "planned", "message": "Download is planned.", "destination": str(destination), "receipt": receipt}

    destination.mkdir(parents=True, exist_ok=True)
    if parsed["kind"] == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            receipt["status"] = "missing_huggingface_hub"
            return {
                "status": "blocked",
                "message": "Install the QLoRA engine first so model downloads are available.",
                "destination": str(destination),
                "receipt": receipt,
            }
        snapshot_download(repo_id=parsed["repo_id"], local_dir=str(destination))
        receipt["status"] = "downloaded"
    else:
        file_name = Path(urlparse(ref).path).name or "downloaded-model-file"
        local_file = destination / file_name
        urlretrieve(ref, local_file)
        receipt["status"] = "downloaded"
        receipt["file"] = str(local_file)

    receipt_path = destination / "retrain_download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {"status": receipt["status"], "message": "Model staged in the models folder.", "destination": str(destination), "receipt": receipt}


def _is_full_sft_method(method: str) -> bool:
    return method in {"sft", "full_sft"}


def _is_adapter_method(method: str) -> bool:
    return method in {"lora", "qlora"}


def estimate_vram(payload: dict[str, Any] | TrainingConfig | None) -> dict[str, Any]:
    config = payload if isinstance(payload, TrainingConfig) else normalize_payload(payload)
    model = MODEL_PRESETS[config.model_key]
    size_b = float(model["size_b"])
    full_sft = _is_full_sft_method(config.method)

    precision_factor = {"4bit": 0.58, "8bit": 1.05, "bf16": 2.05, "fp16": 2.05}[config.precision]
    base_weights = size_b * precision_factor
    adapter = max(0.12, size_b * config.lora_rank * 0.012) if _is_adapter_method(config.method) else 0.0
    if full_sft:
        scope_factor = {"all": 1.0, "last_n_layers": 0.42, "lm_head": 0.08}[config.tune_scope]
        optimizer_base = size_b * 1.8 * scope_factor
    elif config.method in {"dpo", "kto"}:
        optimizer_base = size_b * 1.12
    elif config.method == "reward_model":
        optimizer_base = size_b * 1.0
    elif config.method in {"grpo", "rloo"}:
        optimizer_base = size_b * (0.85 + min(config.num_generations, 8) * 0.08)
    elif config.method == "ppo":
        optimizer_base = size_b * 2.4
    else:
        optimizer_base = size_b * 0.22
    optimizer = optimizer_base * 0.24 if config.cpu_offload else optimizer_base
    checkpoint_factor = 0.58 if config.gradient_checkpointing else 1.0
    attention_factor = 0.82 if config.flash_attention else 1.0
    low_vram_factor = 0.84 if config.low_vram_mode else 1.0
    qlora_factor = 0.74 if config.method == "qlora" else 1.0
    rollout_factor = min(2.0, 1.0 + (config.num_generations - 1) * 0.12) if config.method in {"grpo", "rloo"} else 1.0
    activations = (
        (config.context_length / 1024)
        * config.micro_batch_size
        * max(0.22, size_b * 0.2)
        * checkpoint_factor
        * attention_factor
        * low_vram_factor
        * qlora_factor
        * rollout_factor
    )
    dataloader = 0.45
    safety = 1.6 if config.max_vram_gb >= 24 else 1.2
    estimated_gb = base_weights + adapter + optimizer + activations + dataloader + safety
    limit_gb = min(config.max_vram_gb, config.free_vram_gb + 1.2)
    headroom_gb = limit_gb - estimated_gb
    if headroom_gb >= 2:
        fit_state: FitState = "safe"
    elif headroom_gb >= 0:
        fit_state = "tight"
    else:
        fit_state = "unsafe"

    warnings: list[str] = []
    if fit_state == "unsafe":
        warnings.append("Estimated VRAM exceeds the current safety limit.")
    if config.micro_batch_size > 1 and size_b >= 7 and config.max_vram_gb <= 16:
        warnings.append("7B+ models on a 16 GB GPU should usually start at micro batch 1.")
    if not config.gradient_checkpointing and config.max_vram_gb <= 24:
        warnings.append("Gradient checkpointing is off; this raises activation memory.")
    if config.precision in {"bf16", "fp16"} and size_b >= 7 and config.max_vram_gb <= 16:
        warnings.append("Full precision 7B training is not a safe 16 GB default.")
    if full_sft and config.tune_scope == "all" and config.max_vram_gb <= 24:
        warnings.append("Full-model SFT on a consumer GPU is high risk; start with last layers unless you have headroom.")
    if full_sft and config.precision in {"4bit", "8bit"}:
        warnings.append("Full SFT runner will use BF16/FP16 compute; 4-bit/8-bit selections are adapter-oriented.")
    if config.method in {"dpo", "kto"} and config.max_vram_gb <= 16:
        warnings.append(f"{config.method.upper()} keeps policy/reference pressure high; prefer small models or LoRA on 16 GB.")
    if config.method in {"grpo", "rloo"} and config.num_generations > 4 and config.max_vram_gb <= 24:
        warnings.append("Online RL generations raise activation memory; start with 2-4 generations.")
    if config.method == "ppo":
        warnings.append("PPO is blocked in this TRL install; use GRPO or RLOO for online RL.")

    breakdown = [
        {"item": "Base weights", "gb": round(base_weights, 2), "detail": config.precision},
        {
            "item": "Adapter",
            "gb": round(adapter, 2),
            "detail": f"rank {config.lora_rank}" if _is_adapter_method(config.method) else "none",
        },
        {
            "item": "Optimizer",
            "gb": round(optimizer, 2),
            "detail": (
                f"{config.tune_scope}, CPU offload" if config.cpu_offload and full_sft
                else config.tune_scope if full_sft
                else "CPU offload" if config.cpu_offload
                else "GPU"
            ),
        },
        {
            "item": "Activations",
            "gb": round(activations, 2),
            "detail": f"{config.context_length} ctx x {config.micro_batch_size} batch",
        },
        {"item": "Dataloader", "gb": round(dataloader, 2), "detail": "reserved"},
        {"item": "Safety margin", "gb": round(safety, 2), "detail": f"{config.max_vram_gb:.0f} GB profile"},
    ]
    return asdict(
        VramEstimate(
            fit_state=fit_state,
            estimated_gb=round(estimated_gb, 2),
            limit_gb=round(limit_gb, 2),
            headroom_gb=round(headroom_gb, 2),
            percent=int(max(1, min(100, round((estimated_gb / max(1.0, limit_gb)) * 100)))),
            breakdown=breakdown,
            warnings=warnings,
        )
    )


def dependency_status() -> list[dict[str, Any]]:
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("accelerate", "Accelerate"),
        ("tokenizers", "Tokenizers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("tensorboard", "TensorBoard"),
    ]
    return inspect_packages(packages)


def validate_training_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    config = normalize_payload(payload)
    estimate = estimate_vram(config)
    deps = dependency_status()
    dataset_path = Path(config.dataset_path).expanduser()
    output_root = Path(config.output_root).expanduser()

    gates = [
        _gate("Dataset label", bool(config.dataset_version), config.dataset_version),
        _gate(
            "Base model",
            bool(config.model_path and Path(config.model_path).expanduser().exists()),
            config.model_path or "pick or download a local model",
            fail_state="warning" if config.dry_run else "blocked",
        ),
        _gate("Dataset path", dataset_path.exists(), str(dataset_path)),
        _gate("Output parent", output_root.parent.exists(), str(output_root.parent)),
        _gate("VRAM estimate", estimate["fit_state"] != "unsafe", estimate["fit_state"]),
        _gate("Confirmation", config.confirmed or config.dry_run, "required for write-heavy jobs"),
    ]
    if config.method in BLOCKED_METHODS:
        gates.append(
            _gate(
                "Training runner",
                False,
                f"{config.method.upper()} is not implemented by this checkout's training runner",
                fail_state="blocked",
            )
        )
    if config.method in TRL_METHODS:
        trl_dep = next((dep for dep in deps if dep["package"] == "trl"), None)
        datasets_dep = next((dep for dep in deps if dep["package"] == "datasets"), None)
        gates.append(
            _gate(
                "TRL",
                bool(trl_dep and trl_dep["available"]),
                "needed for DPO, GRPO, KTO, RLOO, and reward-model training",
                fail_state="warning" if config.dry_run else "blocked",
            )
        )
        gates.append(
            _gate(
                "Datasets",
                bool(datasets_dep and datasets_dep["available"]),
                "needed to build TRL datasets from JSONL",
                fail_state="warning" if config.dry_run else "blocked",
            )
        )
    full_sft = _is_full_sft_method(config.method)
    if config.method == "qlora":
        bnb = next((dep for dep in deps if dep["package"] == "bitsandbytes"), None)
        gates.append(
            _gate(
                "BitsAndBytes",
                bool(bnb and bnb["available"]),
                "needed for native 4-bit/8-bit QLoRA",
                fail_state="warning" if config.dry_run else "blocked",
            )
        )
    if config.method in {"lora", "qlora"}:
        peft = next((dep for dep in deps if dep["package"] == "peft"), None)
        gates.append(
            _gate(
                "PEFT",
                bool(peft and peft["available"]),
                "needed for adapter training",
                fail_state="warning" if config.dry_run else "blocked",
            )
        )
    if config.tensorboard:
        tensorboard = next((dep for dep in deps if dep["package"] == "tensorboard"), None)
        gates.append(
            _gate(
                "TensorBoard",
                bool(tensorboard and tensorboard["available"]),
                config.tensorboard_logdir,
                fail_state="warning" if config.dry_run else "blocked",
            )
        )

    blocked = any(gate["state"] == "blocked" for gate in gates)
    status: Literal["ready", "blocked", "warning"]
    has_warning = any(gate["state"] == "warning" for gate in gates)
    if blocked:
        status = "blocked"
    elif estimate["fit_state"] == "tight" or has_warning:
        status = "warning"
    else:
        status = "ready"
    start_enabled = status != "blocked"

    notes = [
        "Plan is generated from the controls in this dashboard.",
        "Receipts include the dataset label, safety gates, output folder, and TensorBoard folder.",
        "Keep dry run enabled until the dependency and VRAM gates are green.",
    ]
    return asdict(
        ValidationResult(
            status=status,
            start_enabled=start_enabled,
            config=asdict(config),
            estimate=estimate,
            dependencies=deps,
            gates=gates,
            notes=notes,
        )
    )


def plan_training_job(payload: dict[str, Any] | None) -> dict[str, Any]:
    result = validate_training_config(payload)
    config = result["config"]
    runner_argv = _training_runner_argv(config)
    launch_spec = {
        "entrypoint": "retrain.gui.training_worker",
        "mode": "dry_run" if config["dry_run"] else "train",
        "runner": "posttrain_bakeoff",
        "argv": runner_argv,
        "dataset_version": config["dataset_version"],
        "model_key": config["model_key"],
        "model_path": config["model_path"],
        "backend_target": config["backend_target"],
        "dataset_path": config["dataset_path"],
        "output_root": config["output_root"],
        "method": config["method"],
        "tune_scope": config["tune_scope"],
        "last_n_layers": config["last_n_layers"],
        "beta": config["beta"],
        "num_generations": config["num_generations"],
        "reward_kind": config["reward_kind"],
        "precision": config["precision"],
        "dataset_format": config["dataset_format"],
        "training_args": {
            "context_length": config["context_length"],
            "micro_batch_size": config["micro_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "tune_scope": config["tune_scope"],
            "last_n_layers": config["last_n_layers"],
            "beta": config["beta"],
            "num_generations": config["num_generations"],
            "reward_kind": config["reward_kind"],
            "lora_rank": config["lora_rank"],
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "warmup_ratio": config["warmup_ratio"],
            "weight_decay": config["weight_decay"],
            "save_every_steps": config["save_every_steps"],
            "eval_every_steps": config["eval_every_steps"],
            "max_steps": config["max_steps"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "flash_attention": config["flash_attention"],
            "qlora": config["qlora"],
            "bitsandbytes": config["bitsandbytes"],
            "unsloth": config["unsloth"],
            "cpu_offload": config["cpu_offload"],
            "low_vram_mode": config["low_vram_mode"],
            "sequence_packing": config["sequence_packing"],
            "tensorboard": config["tensorboard"],
            "tensorboard_logdir": config["tensorboard_logdir"],
        },
    }
    return {
        "status": result["status"],
        "start_enabled": result["start_enabled"],
        "summary": {
            "dataset_version": config["dataset_version"],
            "model": MODEL_PRESETS[config["model_key"]]["label"],
            "method": config["method"],
            "fit_state": result["estimate"]["fit_state"],
            "estimated_gb": result["estimate"]["estimated_gb"],
        },
        "internal_launch_spec": launch_spec,
        "validation": result,
    }


def run_training_job(
    payload: dict[str, Any] | None,
    *,
    execute: bool = False,
    process_factory: ProcessFactory = subprocess.Popen,
) -> dict[str, Any]:
    plan = plan_training_job(payload)
    config = plan["validation"]["config"]
    receipt_root = Path(config["output_root"]) / "receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    receipt_path = receipt_root / f"retrain-training-{timestamp}.json"
    log_path = receipt_root / f"retrain-training-{timestamp}.log"
    status = "dry_run_ready" if config["dry_run"] or not execute else "ready_to_start"
    if not plan["start_enabled"]:
        status = "blocked"
    receipt = {
        "schema_version": 1,
        "created_at": timestamp,
        "status": status,
        "execute_requested": execute,
        "summary": plan["summary"],
        "validation": plan["validation"],
        "internal_launch_spec": plan["internal_launch_spec"],
        "outputs": {
            "receipt_path": str(receipt_path),
            "log_path": str(log_path),
            "output_root": config["output_root"],
            "tensorboard_logdir": config["tensorboard_logdir"],
        },
    }
    if execute and not config["dry_run"] and plan["start_enabled"]:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            try:
                process = process_factory(
                    plan["internal_launch_spec"]["argv"],
                    cwd=str(ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                log_handle.close()
            receipt["status"] = "started"
            pid = getattr(process, "pid", None)
            receipt["process"] = {"pid": pid}
            if isinstance(pid, int) and pid > 0:
                _ACTIVE_JOBS[pid] = {
                    "process": process,
                    "receipt_path": receipt_path,
                    "log_path": log_path,
                    "summary": receipt["summary"],
                    "created_at": timestamp,
                }
        except Exception as exc:
            receipt["status"] = "failed_to_start"
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def training_job_status() -> dict[str, Any]:
    jobs = [_job_snapshot(pid, record) for pid, record in sorted(_ACTIVE_JOBS.items(), reverse=True)]
    active_count = sum(1 for job in jobs if job["running"])
    return {
        "status": "running" if active_count else "idle",
        "active_count": active_count,
        "jobs": jobs,
    }


def stop_training_job(pid: int | None = None) -> dict[str, Any]:
    snapshots = training_job_status()["jobs"]
    active_pids = [int(job["pid"]) for job in snapshots if job["running"]]
    if not active_pids:
        return {"status": "idle", "message": "No ReTrain process is running.", "pid": None}
    target_pid = int(pid) if pid is not None else active_pids[0]
    if target_pid not in active_pids or target_pid not in _ACTIVE_JOBS:
        return {"status": "blocked", "message": "The requested PID is not an active ReTrain job.", "pid": target_pid}

    record = _ACTIVE_JOBS[target_pid]
    process = record["process"]
    terminate = getattr(process, "terminate", None)
    if not callable(terminate):
        return {"status": "blocked", "message": "The active process does not expose a stop handle.", "pid": target_pid}

    forced = False
    try:
        terminate()
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                    forced = True
                    wait(timeout=5)
        final_status = "stopped_forced" if forced else "stopped"
        _update_job_receipt(record, final_status, getattr(process, "returncode", None))
        return {
            "status": final_status,
            "message": f"ReTrain job {target_pid} stopped.",
            "pid": target_pid,
            "forced": forced,
        }
    except Exception as exc:
        _update_job_receipt(record, "stop_failed", getattr(process, "returncode", None), error=str(exc))
        return {"status": "stop_failed", "message": str(exc), "pid": target_pid, "forced": forced}


def _job_snapshot(pid: int, record: dict[str, Any]) -> dict[str, Any]:
    process = record["process"]
    poll = getattr(process, "poll", None)
    return_code = poll() if callable(poll) else None
    running = return_code is None
    status = "running" if running else ("completed" if return_code == 0 else "failed")
    if not running:
        _update_job_receipt(record, status, return_code)
    return {
        "pid": pid,
        "status": status,
        "running": running,
        "return_code": return_code,
        "created_at": record["created_at"],
        "receipt_path": str(record["receipt_path"]),
        "log_path": str(record["log_path"]),
        "summary": record["summary"],
    }


def _update_job_receipt(record: dict[str, Any], status: str, return_code: int | None, *, error: str = "") -> None:
    receipt_path = Path(record["receipt_path"])
    if not receipt_path.exists():
        return
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if payload.get("status") == status and payload.get("process", {}).get("return_code") == return_code:
        return
    payload["status"] = status
    process_payload = payload.get("process") if isinstance(payload.get("process"), dict) else {}
    process_payload.update({"pid": getattr(record["process"], "pid", None), "return_code": return_code})
    payload["process"] = process_payload
    payload["updated_at"] = time.strftime("%Y%m%d_%H%M%S")
    if error:
        payload["error"] = {"type": "StopError", "message": error}
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_recipe_dataset(
    *,
    source_folder: str,
    source_type: str,
    recipe_template: str,
    export_format: str,
    clean_mode: str,
    min_tokens: float,
    max_tokens: float,
    dedupe: bool,
    pii_scan: bool,
    synthetic_expand: bool,
    train_split: float,
    eval_split: float,
    test_split: float,
    output_root: str | None = None,
) -> dict[str, Any]:
    export_root = _resolve_path(output_root or str(DEFAULT_DATASET_EXPORT_ROOT))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_dir = export_root / f"recipe-{_safe_name(recipe_template)}-{timestamp}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source = _resolve_path(source_folder)
    files = _scan_source_files(source, {".json", ".jsonl", ".csv", ".txt", ".md", ".markdown", ".pdf", ".parquet"})
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in files:
        text = _source_preview_text(path)
        token_count = _rough_token_count(text)
        if token_count < int(min_tokens) or token_count > int(max_tokens):
            continue
        key = text.strip().lower()
        if dedupe and key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"Use this source material from {path.name}."},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {
                    "dataset_version": ACTIVE_DATASET_VERSION,
                    "source_path": str(path),
                    "source_type": source_type,
                    "recipe": recipe_template,
                    "export_format": export_format,
                    "clean_mode": clean_mode,
                    "pii_scan_requested": bool(pii_scan),
                    "synthetic_expand_requested": bool(synthetic_expand),
                    "rough_tokens": token_count,
                },
            }
        )
    splits = _split_rows(rows, train_split, eval_split, test_split)
    for split, split_rows in splits.items():
        _write_jsonl(dataset_dir / f"{split}.jsonl", split_rows)
    manifest = {
        "schema_version": "retrain-gui-recipe-dataset-v1",
        "dataset_version": ACTIVE_DATASET_VERSION,
        "created_at": timestamp,
        "source_folder": str(source),
        "dataset_dir": str(dataset_dir),
        "recipe": recipe_template,
        "export_format": export_format,
        "split_counts": {split: len(split_rows) for split, split_rows in splits.items()},
        "source_file_count": len(files),
        "written_records": len(rows),
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ready" if rows else "empty", "dataset_dir": str(dataset_dir), "manifest": manifest}


def build_vision_dataset(
    *,
    image_folder: str,
    caption_mode: str,
    annotation_mode: str,
    crop_policy: str,
    max_side: float,
    include_boxes: bool,
    auto_caption: bool,
    export_format: str,
    output_root: str | None = None,
) -> dict[str, Any]:
    export_root = _resolve_path(output_root or str(DEFAULT_DATASET_EXPORT_ROOT))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_dir = export_root / f"vision-{_safe_name(export_format)}-{timestamp}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source = _resolve_path(image_folder)
    files = _scan_source_files(source, {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
    rows = []
    for path in files:
        caption = _caption_for_image(path, caption_mode=caption_mode, auto_caption=auto_caption)
        row = {
            "image": str(path),
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image for local VLM training."},
                {"from": "gpt", "value": caption},
            ],
            "metadata": {
                "dataset_version": ACTIVE_DATASET_VERSION,
                "annotation_mode": annotation_mode,
                "crop_policy": crop_policy,
                "max_side": int(max_side),
                "include_boxes": bool(include_boxes),
                "export_format": export_format,
            },
        }
        if include_boxes:
            row["boxes"] = []
        rows.append(row)
    splits = _split_rows(rows, 90, 5, 5)
    for split, split_rows in splits.items():
        _write_jsonl(dataset_dir / f"{split}.jsonl", split_rows)
    manifest = {
        "schema_version": "retrain-gui-vision-dataset-v1",
        "dataset_version": ACTIVE_DATASET_VERSION,
        "created_at": timestamp,
        "image_folder": str(source),
        "dataset_dir": str(dataset_dir),
        "export_format": export_format,
        "image_count": len(files),
        "split_counts": {split: len(split_rows) for split, split_rows in splits.items()},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ready" if rows else "empty", "dataset_dir": str(dataset_dir), "manifest": manifest}


def tensorboard_status(logdir: str, *, host: str = "127.12.6.3", port: int = 6006) -> dict[str, Any]:
    path = _resolve_path(logdir or str(DEFAULT_OUTPUT_ROOT / "tensorboard"))
    event_files = [item for item in path.rglob("events.out.tfevents*") if item.is_file()] if path.exists() else []
    if importlib.util.find_spec("tensorboard") is None:
        state = "missing_tensorboard"
    elif not path.exists():
        state = "missing_logdir"
    elif event_files:
        state = "ready"
    else:
        state = "waiting_for_events"
    return {
        "status": state,
        "logdir": str(path),
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "event_file_count": len(event_files),
    }


def start_tensorboard(
    logdir: str,
    *,
    host: str = "127.12.6.3",
    port: int = 6006,
    process_factory: ProcessFactory = subprocess.Popen,
) -> dict[str, Any]:
    status = tensorboard_status(logdir, host=host, port=port)
    log_path = DEFAULT_OUTPUT_ROOT / "tensorboard-server.log"
    result = dict(status)
    result["server_log"] = str(log_path)
    if status["status"] == "missing_tensorboard":
        return result
    Path(status["logdir"]).mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        status["logdir"],
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = process_factory(
            argv,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()
    result["status"] = "started"
    result["pid"] = getattr(process, "pid", None)
    return result


def _training_runner_argv(config: dict[str, Any]) -> list[str]:
    method = {
        "sft": "full_sft",
        "full_sft": "full_sft",
        "lora": "lora",
        "qlora": "qlora",
        "dpo": "dpo",
        "grpo": "grpo",
        "reward_model": "reward_model",
        "kto": "kto",
        "rloo": "rloo",
        "ppo": "ppo",
    }.get(config["method"], "qlora")
    precision = config["precision"] if config["precision"] in {"bf16", "fp16"} else "bf16"
    argv = [
        sys.executable,
        "scripts\\run_posttrain_bakeoff.py",
        "--model",
        config["model_key"],
        "--data-dir",
        config["dataset_path"],
        "--output-root",
        config["output_root"],
        "--method",
        method,
        "--max-seq-length",
        str(config["context_length"]),
        "--batch-size",
        str(config["micro_batch_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--precision",
        precision,
        "--optim",
        "paged_adamw_8bit" if method == "qlora" else config["optimizer"],
        "--lr-scheduler-type",
        config["scheduler"],
        "--lora-r",
        str(config["lora_rank"]),
        "--max-steps",
        str(config["max_steps"] or 1),
        "--no-save-checkpoints",
    ]
    if method == "full_sft":
        argv.extend(["--tune-scope", config["tune_scope"], "--last-n-layers", str(config["last_n_layers"])])
    if method in {"dpo", "kto", "grpo", "rloo"}:
        argv.extend(["--beta", str(config["beta"])])
    if method in {"grpo", "rloo"}:
        argv.extend(["--num-generations", str(config["num_generations"]), "--reward-kind", config["reward_kind"]])
    if config["model_path"]:
        model_path = _resolve_path(config["model_path"])
        argv.extend(["--model-path", str(model_path), "--model-name", _safe_name(model_path.name)])
    if not config["gradient_checkpointing"]:
        argv.append("--no-gradient-checkpointing")
    if config["trust_remote_code"]:
        argv.append("--trust-remote-code")
    if config["dry_run"]:
        argv.append("--dry-run")
    return argv


def _gate(name: str, ok: bool, detail: str, *, fail_state: Literal["blocked", "warning"] = "blocked") -> dict[str, str]:
    return {"gate": name, "state": "ready" if ok else fail_state, "detail": detail}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _parse_model_ref(ref: str) -> dict[str, str]:
    parsed = urlparse(ref)
    if parsed.netloc == "huggingface.co":
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            repo_id = "/".join(parts[:2])
            return {"kind": "huggingface", "repo_id": repo_id, "name": repo_id.replace("/", "--")}
    if "://" not in ref and ref.count("/") == 1:
        return {"kind": "huggingface", "repo_id": ref, "name": ref.replace("/", "--")}
    if parsed.scheme in {"http", "https"}:
        name = Path(parsed.path).stem or parsed.netloc
        return {"kind": "direct", "name": name}
    return {"kind": "huggingface", "repo_id": ref, "name": ref.replace("/", "--")}


def _safe_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name.strip())
    return cleaned.strip("-") or "downloaded-model"


def _scan_source_files(root: Path, suffixes: set[str], limit: int = 5000) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
            if len(files) >= limit:
                break
    return sorted(files, key=lambda path: str(path).lower())


def _source_preview_text(path: Path, limit: int = 6000) -> str:
    if path.suffix.lower() == ".pdf":
        return f"PDF source queued for extraction: {path.name}"
    if path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".md", ".markdown"}:
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except TypeError:
            text = path.read_text(encoding="utf-8-sig")
        return " ".join(text.split())[:limit] or f"Empty text source: {path.name}"
    return f"Source queued for conversion: {path.name}"


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


def _split_rows(rows: list[dict[str, Any]], train_pct: float, eval_pct: float, test_pct: float) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {"train": [], "validation": [], "test": []}
    total_pct = max(1.0, float(train_pct) + float(eval_pct) + float(test_pct))
    train_count = int(round(len(rows) * (float(train_pct) / total_pct)))
    eval_count = int(round(len(rows) * (float(eval_pct) / total_pct)))
    if len(rows) > 1 and train_count >= len(rows):
        train_count = len(rows) - 1
    if len(rows) > 2 and eval_count == 0:
        eval_count = 1
    test_start = min(len(rows), train_count + eval_count)
    return {
        "train": rows[:train_count],
        "validation": rows[train_count:test_start],
        "test": rows[test_start:],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _caption_for_image(path: Path, *, caption_mode: str, auto_caption: bool) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if caption_mode == "filename" or auto_caption:
        return f"Image showing {stem or path.name}."
    if caption_mode == "local-assist":
        return f"Caption draft requested for {stem or path.name}."
    return f"Manual caption needed for {stem or path.name}."


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(plan_training_job(payload), indent=2))
