from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[2]
GRADIO_DIR = ROOT / "gui" / "gradio"
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

import backend_worker as worker  # noqa: E402
from gui.api import live_routes  # noqa: E402
from gui.api import model_layers  # noqa: E402
from gui.api import pipeline_blueprints  # noqa: E402


VERSION = "api-v0.1"

MODEL_ID_MAP = {
    "lfm25-350m": "lfm2.5-350m",
    "qwen25-coder-15b": "qwen2.5-coder-1.5b",
    "smollm3-3b": "smollm3-3b",
    "qwen25-7b": "qwen2.5-7b",
    "qwen3-vl-4b": "qwen-vl-4b",
}
METHOD_MAP = {
    "Full fine-tune": "full_sft",
    "LoRA": "lora",
    "QLoRA": "qlora",
    "DPO": "dpo",
    "GRPO": "grpo",
    "Reward model": "reward_model",
    "KTO": "kto",
    "RLOO": "rloo",
    "PPO": "ppo",
}
TUNE_SCOPE_MAP = {
    "Full model": "all",
    "Last layers": "last_n_layers",
    "Output head only": "lm_head",
}
PRECISION_MAP = {
    "BF16": "bf16",
    "FP16": "fp16",
    "8-bit": "8bit",
    "4-bit": "4bit",
}
OPTIMIZER_MAP = {
    "AdamW 8-bit": "adamw_8bit",
    "Paged AdamW": "paged_adamw_8bit",
    "Lion": "adamw_torch",
    "CPU AdamW": "adamw_torch",
}
REWARD_KIND_MAP = {
    "Heuristic": "heuristic",
    "Length": "length",
    "Tag match": "tag_match",
}
SCHEDULER_MAP = {
    "Cosine": "cosine",
    "Linear": "linear",
    "Constant with warmup": "constant",
}
SENSITIVE_KEYS = {"argv", "command", "internal_launch_spec", "entrypoint", "runner"}
WORKSPACE_PATHS = {
    "root": ROOT,
    "models": worker.DEFAULT_MODEL_ROOT,
    "datasets": ROOT / "datasets",
    "engines": worker.DEFAULT_ENGINE_ROOT,
    "outputs": worker.DEFAULT_OUTPUT_ROOT,
    "receipts": worker.DEFAULT_OUTPUT_ROOT / "receipts",
    "tensorboard": worker.DEFAULT_OUTPUT_ROOT / "tensorboard",
    "dataset_exports": worker.DEFAULT_DATASET_EXPORT_ROOT,
    "presets": worker.DEFAULT_OUTPUT_ROOT / "presets",
    "exports": worker.DEFAULT_OUTPUT_ROOT / "exports",
    "pipelines": worker.DEFAULT_OUTPUT_ROOT / "pipelines",
}


def create_app() -> FastAPI:
    app = FastAPI(title="ReTrain Local API", version=VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.12.6.3:4173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "http://127.12.6.3:4174",
            "http://127.0.0.1:4174",
            "http://localhost:4174",
            "retrain://app",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    live_routes.register(app)          # /api/retrain/live/*: watch a running training job

    @app.get("/api/retrain/bootstrap")
    def bootstrap() -> dict[str, Any]:
        return bootstrap_payload()

    @app.get("/api/retrain/runtime")
    def runtime() -> dict[str, Any]:
        return runtime_payload()

    @app.get("/api/retrain/workspace")
    def workspace() -> dict[str, Any]:
        return workspace_payload()

    @app.get("/api/retrain/datasets/preview")
    def dataset_preview() -> dict[str, Any]:
        return sanitize_value(dataset_preview_payload())

    @app.get("/api/retrain/models/inventory")
    def model_inventory() -> dict[str, Any]:
        return sanitize_value(worker.model_inventory(str(WORKSPACE_PATHS["models"])))

    @app.get("/api/retrain/models/local")
    def local_models() -> dict[str, Any]:
        """Every folder under the model root that holds safetensors weights."""
        return sanitize_value({"models": model_layers.list_local_models(WORKSPACE_PATHS["models"])})

    @app.get("/api/retrain/models/layers")
    def model_layer_map(folder: str = "", modelId: str = "", sample: bool = True) -> dict[str, Any]:
        """Layer geometry, and sampled weight statistics, for one local model.

        Read straight from the safetensors headers, so this costs a header read
        rather than a model load. `sample=false` skips the weight sampling and
        returns structure only, which is instant for any size of checkpoint.
        """
        target = folder.strip()
        # A model id from the bootstrap catalogue resolves through the preset
        # table; only the folder form ever comes from the client as a path
        # fragment, and that is validated against the model root.
        if not target and modelId:
            preset_key = MODEL_ID_MAP.get(modelId, modelId)
            preset = worker.MODEL_PRESETS.get(preset_key, {})
            preset_path = preset.get("path")
            if not preset_path:
                return {"status": "blocked", "message": f"No local path is configured for {modelId}."}
            target = Path(preset_path).name
        if not target:
            return {"status": "blocked", "message": "Pass folder or modelId."}
        try:
            model_dir = model_layers.resolve_model_dir(WORKSPACE_PATHS["models"], target)
            return sanitize_value(_cached_layer_map(model_dir, bool(sample)))
        except (FileNotFoundError, ValueError) as error:
            return {"status": "blocked", "message": str(error)}

    @app.post("/api/retrain/plan")
    def plan(request: dict[str, Any]) -> dict[str, Any]:
        payload = request_to_training_payload(request)
        return sanitize_plan(worker.plan_training_job(payload))

    @app.post("/api/retrain/jobs/dry-run")
    def dry_run(request: dict[str, Any]) -> dict[str, Any]:
        payload = request_to_training_payload(request)
        payload["dry_run"] = True
        payload["confirmed"] = False
        return sanitize_receipt(worker.run_training_job(payload, execute=False))

    @app.post("/api/retrain/jobs/start")
    def start_job(request: dict[str, Any]) -> dict[str, Any]:
        payload = request_to_training_payload(request)
        payload["dry_run"] = False
        payload["confirmed"] = bool(request.get("confirmed", payload.get("confirmed", False)))
        return sanitize_receipt(worker.run_training_job(payload, execute=True))

    @app.get("/api/retrain/jobs/status")
    def job_status() -> dict[str, Any]:
        return sanitize_value(worker.training_job_status())

    @app.post("/api/retrain/jobs/stop")
    def stop_job(request: dict[str, Any]) -> dict[str, Any]:
        raw_pid = request.get("pid")
        try:
            pid = int(raw_pid) if raw_pid not in (None, "") else None
        except (TypeError, ValueError):
            return {"status": "blocked", "message": "pid must be an integer"}
        return sanitize_value(worker.stop_training_job(pid))

    @app.get("/api/retrain/engines/qlora")
    def qlora_engine() -> dict[str, Any]:
        return worker.engine_status("qlora")

    @app.post("/api/retrain/engines/qlora/install")
    def install_qlora(request: dict[str, Any]) -> dict[str, Any]:
        execute = bool(request.get("execute", False))
        return sanitize_value(worker.install_engine("qlora", execute=execute))

    @app.post("/api/retrain/models/download/plan")
    def plan_model_download(request: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(
            worker.stage_model_download(
                str(request.get("downloadRef", "")),
                model_root=str(WORKSPACE_PATHS["models"]),
                alias=str(request.get("alias", "")),
                execute=False,
            )
        )

    @app.post("/api/retrain/models/download/start")
    def start_model_download(request: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(
            worker.stage_model_download(
                str(request.get("downloadRef", "")),
                model_root=str(WORKSPACE_PATHS["models"]),
                alias=str(request.get("alias", "")),
                execute=True,
            )
        )

    @app.get("/api/retrain/tensorboard/status")
    def tensorboard_status() -> dict[str, Any]:
        return sanitize_value(worker.tensorboard_status(str(WORKSPACE_PATHS["tensorboard"])))

    @app.post("/api/retrain/tensorboard/start")
    def start_tensorboard(request: dict[str, Any]) -> dict[str, Any]:
        logdir = resolve_workspace_path(request.get("logdir"), "tensorboard")
        if logdir is None:
            return {"status": "blocked", "message": "TensorBoard logdir must be inside the ReTrain workspace."}
        port = int(request.get("port", 6006))
        host = str(request.get("host", "127.12.6.3"))
        return sanitize_value(worker.start_tensorboard(str(logdir), host=host, port=port))

    @app.post("/api/retrain/datasets/recipe/build")
    def build_recipe_dataset(request: dict[str, Any]) -> dict[str, Any]:
        source = resolve_workspace_path(request.get("sourceFolder"), "datasets")
        if source is None:
            return {"status": "blocked", "message": "Dataset source must be inside the ReTrain workspace."}
        return sanitize_value(
            worker.build_recipe_dataset(
                source_folder=str(source),
                source_type=str(request.get("sourceType", "local-folder")),
                recipe_template=str(request.get("recipeTemplate", "qwen-chat-sft")),
                export_format=str(request.get("exportFormat", "chatml-jsonl")),
                clean_mode=str(request.get("cleanMode", "balanced")),
                min_tokens=float(request.get("minTokens", 1)),
                max_tokens=float(request.get("maxTokens", 4096)),
                dedupe=bool(request.get("dedupe", True)),
                pii_scan=bool(request.get("piiScan", True)),
                synthetic_expand=bool(request.get("syntheticExpand", False)),
                train_split=float(request.get("trainSplit", 90)),
                eval_split=float(request.get("evalSplit", 5)),
                test_split=float(request.get("testSplit", 5)),
                output_root=str(WORKSPACE_PATHS["dataset_exports"]),
            )
        )

    @app.post("/api/retrain/datasets/vision/build")
    def build_vision_dataset(request: dict[str, Any]) -> dict[str, Any]:
        source = resolve_workspace_path(request.get("imageFolder"), "datasets")
        if source is None:
            return {"status": "blocked", "message": "Image source must be inside the ReTrain workspace."}
        return sanitize_value(
            worker.build_vision_dataset(
                image_folder=str(source),
                caption_mode=str(request.get("captionMode", "filename")),
                annotation_mode=str(request.get("annotationMode", "caption")),
                crop_policy=str(request.get("cropPolicy", "full")),
                max_side=float(request.get("maxSide", 1024)),
                include_boxes=bool(request.get("includeBoxes", False)),
                auto_caption=bool(request.get("autoCaption", False)),
                export_format=str(request.get("exportFormat", "qwen-vl-json")),
                output_root=str(WORKSPACE_PATHS["dataset_exports"]),
            )
        )

    @app.get("/api/retrain/presets")
    def presets() -> dict[str, Any]:
        return sanitize_value(presets_payload())

    @app.post("/api/retrain/presets/save")
    def save_preset(request: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(save_preset_payload(request))

    @app.get("/api/retrain/checkpoints")
    def checkpoints() -> dict[str, Any]:
        return sanitize_value(checkpoint_inventory_payload())

    @app.get("/api/retrain/compare")
    def compare_runs() -> dict[str, Any]:
        return sanitize_value(compare_runs_payload())

    @app.post("/api/retrain/pipelines/preview")
    def preview_pipelines(request: dict[str, Any]) -> dict[str, Any]:
        training_payload = request_to_training_payload(request)
        return sanitize_value(pipeline_blueprints.preview_pipeline_bundle(training_payload, request, WORKSPACE_PATHS))

    @app.post("/api/retrain/pipelines/export")
    def export_pipelines(request: dict[str, Any]) -> dict[str, Any]:
        training_payload = request_to_training_payload(request)
        return sanitize_value(pipeline_blueprints.export_pipeline_bundle(training_payload, request, WORKSPACE_PATHS))

    @app.post("/api/retrain/advisor")
    def training_advisor(request: dict[str, Any]) -> dict[str, Any]:
        training_payload = request_to_training_payload(request)
        return sanitize_value(training_advisor_payload(training_payload, request))

    @app.post("/api/retrain/artifacts/delete")
    def delete_artifact(request: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(delete_artifact_payload(request))

    @app.post("/api/retrain/artifacts/export")
    def export_artifact(request: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(export_artifact_payload(request))

    @app.post("/api/retrain/open-path")
    def open_path(request: dict[str, Any]) -> dict[str, Any]:
        return open_path_payload(request)

    return app


# Inspecting a checkpoint reads its headers and samples its weights, which is
# fast but not free. The UI re-asks whenever the operator switches models or
# toggles a metric, so results are held per (folder, mtime, sample) and dropped
# the moment the file changes on disk.
_LAYER_MAP_CACHE: dict[tuple[str, int, bool], dict[str, Any]] = {}
_LAYER_MAP_CACHE_LIMIT = 8


def _cached_layer_map(model_dir: Path, sample: bool) -> dict[str, Any]:
    shards = model_layers.shard_files(model_dir)
    newest = max((shard.stat().st_mtime_ns for shard in shards), default=0)
    key = (str(model_dir), newest, sample)
    cached = _LAYER_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    result = model_layers.inspect_model(model_dir, sample=sample)
    if len(_LAYER_MAP_CACHE) >= _LAYER_MAP_CACHE_LIMIT:
        _LAYER_MAP_CACHE.pop(next(iter(_LAYER_MAP_CACHE)))
    _LAYER_MAP_CACHE[key] = result
    return result


def bootstrap_payload() -> dict[str, Any]:
    config = worker.normalize_payload({})
    return {
        "workspaceName": "ReTrain",
        "subtitle": "Local no-code training workbench",
        "version": VERSION,
        "activeDatasetLabel": worker.ACTIVE_DATASET_VERSION,
        "models": [
            {
                "id": "lfm25-350m",
                "name": worker.MODEL_PRESETS["lfm2.5-350m"]["label"],
                "family": "LFM2",
                "sizeB": worker.MODEL_PRESETS["lfm2.5-350m"]["size_b"],
                "backend": "HF local",
                "status": "Smallest verified LoRA smoke model",
            },
            {
                "id": "qwen25-coder-15b",
                "name": worker.MODEL_PRESETS["qwen2.5-coder-1.5b"]["label"],
                "family": "Qwen",
                "sizeB": worker.MODEL_PRESETS["qwen2.5-coder-1.5b"]["size_b"],
                "backend": "HF local",
                "status": "Safest full-SFT starter",
            },
            {
                "id": "smollm3-3b",
                "name": worker.MODEL_PRESETS["smollm3-3b"]["label"],
                "family": "SmolLM",
                "sizeB": worker.MODEL_PRESETS["smollm3-3b"]["size_b"],
                "backend": "HF local",
                "status": "Good LoRA comparison",
            },
            {
                "id": "qwen25-7b",
                "name": worker.MODEL_PRESETS["qwen2.5-7b"]["label"],
                "family": "Qwen",
                "sizeB": worker.MODEL_PRESETS["qwen2.5-7b"]["size_b"],
                "backend": "Ollama eval / HF train",
                "status": "Needs VRAM care",
            },
            {
                "id": "qwen3-vl-4b",
                "name": worker.MODEL_PRESETS["qwen-vl-4b"]["label"],
                "family": "Qwen-VL",
                "sizeB": worker.MODEL_PRESETS["qwen-vl-4b"]["size_b"],
                "backend": "Ollama vision eval",
                "status": "Vision workflow target",
            },
        ],
        "datasets": [
            {
                "id": "qwen-research-v1",
                "name": worker.ACTIVE_DATASET_VERSION,
                "source": config.dataset_path,
                "rows": 0,
                "rejected": 0,
                "status": "Ready" if Path(config.dataset_path).exists() else "Needs data",
            },
            {
                "id": "preferences-v1",
                "name": "Preference pairs",
                "source": "prompt/chosen/rejected JSONL",
                "rows": 0,
                "rejected": 0,
                "status": "Draft",
            },
            {
                "id": "vision-vl-v1",
                "name": "Qwen-VL image captions",
                "source": "Image folders",
                "rows": 0,
                "rejected": 0,
                "status": "Draft",
            },
        ],
        "defaults": {
            "modelId": "qwen25-coder-15b",
            "datasetId": "qwen-research-v1",
            "method": "QLoRA",
            "tuneScope": "Last layers",
            "lastNLayers": config.last_n_layers,
            "beta": config.beta,
            "numGenerations": config.num_generations,
            "rewardKind": "Heuristic",
            "contextLength": config.context_length,
            "microBatch": config.micro_batch_size,
            "gradAccum": config.gradient_accumulation_steps,
            "loraRank": config.lora_rank,
            "precision": "4-bit",
            "optimizer": "Paged AdamW",
            "scheduler": "Cosine",
            "trustRemoteCode": False,
            "gradientCheckpointing": config.gradient_checkpointing,
            "flashAttention": config.flash_attention,
            "qlora": config.qlora,
            "cpuOffload": config.cpu_offload,
            "lowVramMode": config.low_vram_mode,
            "tensorboard": config.tensorboard,
            "confirmed": config.confirmed,
            "dryRun": config.dry_run,
        },
        "receipts": [
            {
                "id": "preflight",
                "title": "Hardware preflight",
                "detail": "Local planning profile loaded",
                "state": "ready",
                "time": "live",
            },
            {
                "id": "trl",
                "title": "TRL lanes",
                "detail": "DPO, GRPO, reward model, KTO, and RLOO available when training deps are installed",
                "state": "ready",
                "time": "now",
            },
        ],
        "visionSamples": [],
        "deployTargets": [
            {"id": "gguf", "name": "llama.cpp GGUF", "status": "Planned", "detail": "Merge adapter, quantize, validate smoke prompt"},
            {"id": "ollama", "name": "Ollama", "status": "Ready path", "detail": "Package local model card and Modelfile controls"},
            {"id": "vllm", "name": "vLLM", "status": "Configured", "detail": "Adapter serving check against OpenAI-compatible endpoint"},
        ],
    }


def runtime_payload() -> dict[str, Any]:
    engine = worker.engine_status("qlora")
    deps = worker.dependency_status()
    jobs = worker.training_job_status()
    gpu = _gpu_status()
    ready_count = sum(1 for dep in deps if dep["available"])
    total_count = len(deps)
    core_ready = all(dep["available"] for dep in deps if dep["package"] != "bitsandbytes")
    active_jobs = int(jobs["active_count"])
    vram_total = float(gpu["vram_total_gb"])
    vram_free = float(gpu["vram_free_gb"])
    vram_used_percent = int(round((1.0 - (vram_free / max(0.1, vram_total))) * 100))
    return {
        "state": "Training" if active_jobs else ("Ready" if engine["ready"] else ("Ready (LoRA)" if core_ready else "Needs engine install")),
        "backend": "ReTrain local API",
        "device": gpu["name"],
        "vramTotalGb": vram_total,
        "vramFreeGb": vram_free,
        "engine": engine,
        "dependencies": deps,
        "resources": [
            {"label": "VRAM", "value": f"{vram_total - vram_free:.1f} / {vram_total:.1f} GB used", "percent": vram_used_percent, "tone": "amber" if vram_used_percent >= 80 else "mint"},
            {"label": "GPU", "value": f"{gpu['utilization_percent']}% at {gpu['temperature_c']} C", "percent": int(gpu["utilization_percent"]), "tone": "amber" if int(gpu["temperature_c"]) >= 80 else "neutral"},
            {"label": "Deps", "value": f"{ready_count} / {total_count} ready", "percent": int((ready_count / max(1, total_count)) * 100), "tone": "neutral"},
            {"label": "QLoRA", "value": "ready" if engine["ready"] else "native CUDA unavailable", "percent": 100 if engine["ready"] else 30, "tone": "mint" if engine["ready"] else "amber"},
            {"label": "Queue", "value": f"{active_jobs} active job{'s' if active_jobs != 1 else ''}", "percent": 100 if active_jobs else 0, "tone": "amber" if active_jobs else "neutral"},
        ],
    }


def _gpu_status() -> dict[str, Any]:
    fallback = {
        "name": "NVIDIA GPU (live query unavailable)",
        "vram_total_gb": 16.0,
        "vram_free_gb": 14.8,
        "utilization_percent": 0,
        "temperature_c": 0,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    if result.returncode != 0 or not result.stdout.strip():
        return fallback
    fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        return fallback
    try:
        return {
            "name": fields[0],
            "vram_total_gb": round(float(fields[1]) / 1024, 1),
            "vram_free_gb": round(float(fields[2]) / 1024, 1),
            "utilization_percent": int(float(fields[3])),
            "temperature_c": int(float(fields[4])),
        }
    except ValueError:
        return fallback


def workspace_payload() -> dict[str, Any]:
    worker.ensure_mvp_folders()
    return {
        "root": str(ROOT),
        "paths": [path_summary(key, path) for key, path in WORKSPACE_PATHS.items()],
    }


def dataset_preview_payload() -> dict[str, Any]:
    export_root = WORKSPACE_PATHS["dataset_exports"].resolve()
    latest_dataset_dir = _latest_dataset_export_dir(export_root)
    if latest_dataset_dir is not None:
        manifest_path = latest_dataset_dir / "manifest.json"
        manifest = _read_json_file(manifest_path)
        split_name, split_path = _select_preview_split(latest_dataset_dir)
        rows = _read_preview_rows(split_path)
        valid_rows = [row for row in rows if row["prompt"] or row["response"]]
        token_values = [row["tokens"] for row in valid_rows]
        split_counts = manifest.get("split_counts", {}) if isinstance(manifest, dict) else {}
        total_samples = int(sum(int(value) for value in split_counts.values())) if isinstance(split_counts, dict) else len(valid_rows)
        max_tokens = max(token_values, default=0)
        avg_tokens = round(sum(token_values) / max(1, len(token_values)))
        return {
            "dataset_name": latest_dataset_dir.name,
            "source": str(latest_dataset_dir),
            "rows": valid_rows,
            "stats": {
                "total_samples": total_samples,
                "valid_samples": len(valid_rows),
                "total_tokens_est": _format_token_estimate(sum(token_values)),
                "avg_tokens": avg_tokens,
                "max_tokens": max_tokens,
                "warnings": max(0, total_samples - len(valid_rows)),
                "errors": 0,
                "language": "English",
                "format": "Conversation",
                "split": split_name,
                "last_scanned": _format_timestamp(latest_dataset_dir.stat().st_mtime),
            },
        }

    source_root = WORKSPACE_PATHS["datasets"].resolve()
    source_rows = _scan_workspace_dataset_rows(source_root)
    token_values = [row["tokens"] for row in source_rows]
    return {
        "dataset_name": source_root.name,
        "source": str(source_root),
        "rows": source_rows,
        "stats": {
            "total_samples": len(source_rows),
            "valid_samples": len(source_rows),
            "total_tokens_est": _format_token_estimate(sum(token_values)),
            "avg_tokens": round(sum(token_values) / max(1, len(token_values))),
            "max_tokens": max(token_values, default=0),
            "warnings": 0,
            "errors": 0,
            "language": "English",
            "format": "Source Files",
            "split": "Workspace",
            "last_scanned": _format_timestamp(source_root.stat().st_mtime) if source_root.exists() else "Not scanned",
        },
    }


def presets_payload() -> dict[str, Any]:
    root = WORKSPACE_PATHS["presets"].resolve()
    root.mkdir(parents=True, exist_ok=True)
    presets = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json_file(path)
        presets.append(
            {
                "name": str(payload.get("name", path.stem)),
                "path": str(path),
                "saved_at": str(payload.get("saved_at", _format_timestamp(path.stat().st_mtime))),
            }
        )
    return {"root": str(root), "count": len(presets), "presets": presets}


def save_preset_payload(request: dict[str, Any]) -> dict[str, Any]:
    settings = request.get("settings", {})
    if not isinstance(settings, dict):
        return {"status": "blocked", "message": "Preset settings must be a JSON object."}
    root = WORKSPACE_PATHS["presets"].resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    requested_name = str(request.get("name", "")).strip()
    label = requested_name or f"{settings.get('method', 'preset')}-{settings.get('modelId', 'model')}"
    safe_name = _safe_filename(label)
    path = root / f"{timestamp}-{safe_name}.json"
    payload = {
        "schema_version": "retrain-preset-v1",
        "name": label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "status": "saved",
        "preset": {
            "name": label,
            "path": str(path),
            "saved_at": payload["saved_at"],
        },
    }


def checkpoint_inventory_payload() -> dict[str, Any]:
    items = _collect_run_artifacts()
    return {"count": len(items), "items": items}


def compare_runs_payload() -> dict[str, Any]:
    items = _collect_run_artifacts()
    runs = [
        {
            "id": item["id"],
            "label": item["label"],
            "status": item["status"],
            "method": item["method"],
            "model": item["model"],
            "fit_state": item["fit_state"],
            "estimated_gb": item["estimated_gb"],
            "created_at": item["created_at"],
            "path": item["path"],
        }
        for item in items[:6]
    ]
    baseline = runs[0] if len(runs) > 0 else None
    candidate_a = runs[1] if len(runs) > 1 else None
    candidate_b = runs[2] if len(runs) > 2 else None
    metrics = [
        _compare_metric_row("Status", baseline, candidate_a, candidate_b, key="status"),
        _compare_metric_row("Method", baseline, candidate_a, candidate_b, key="method"),
        _compare_metric_row("Fit State", baseline, candidate_a, candidate_b, key="fit_state"),
        _compare_metric_row("Estimated VRAM (GB)", baseline, candidate_a, candidate_b, key="estimated_gb"),
    ]
    return {
        "runs": runs,
        "metrics": metrics,
        "sample_prompt": "No evaluation sample has been stored in local run receipts yet.",
        "sample_outputs": [
            {
                "run_id": run["id"],
                "title": run["label"],
                "text": f"{run['status']} | {run['method']} | {run['model']}",
            }
            for run in runs[:3]
        ],
    }


def training_advisor_payload(training_payload: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, Any]:
    request = request or {}
    plan = worker.plan_training_job(training_payload)
    validation = plan.get("validation", {})
    estimate = validation.get("estimate", {}) if isinstance(validation.get("estimate"), dict) else {}
    gates = validation.get("gates", []) if isinstance(validation.get("gates"), list) else []
    dependencies = validation.get("dependencies", []) if isinstance(validation.get("dependencies"), list) else []
    recommendations = _advisor_recommendations(plan, gates, dependencies, estimate)
    coach = _local_coach_payload(plan, recommendations, request)
    tools = [
        {
            "name": "inspect_model_inventory",
            "description": "Read local model folders and supported checkpoint files.",
            "input_schema": {"type": "object", "properties": {"model_root": {"type": "string"}}},
        },
        {
            "name": "preview_dataset",
            "description": "Load the active dataset preview, counts, warnings, and sample rows.",
            "input_schema": {"type": "object", "properties": {"dataset_path": {"type": "string"}}},
        },
        {
            "name": "plan_training",
            "description": "Run the same VRAM, dependency, and method gates used by Start Training.",
            "input_schema": {"type": "object", "properties": {"settings": {"type": "object"}}},
        },
        {
            "name": "export_pipeline_bundle",
            "description": "Write local pipeline specs for training, evaluation, deployment, and agent workflow.",
            "input_schema": {"type": "object", "properties": {"settings": {"type": "object"}}},
        },
        {
            "name": "ask_local_training_coach",
            "description": "Ask a small local Ollama model for a plain-language training recommendation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "ollama_url": {"type": "string"},
                    "use_model": {"type": "boolean"},
                },
            },
        },
    ]
    return {
        "mode": "local_model" if coach["status"] == "ready" else "rules",
        "langchain_ready": importlib.util.find_spec("langchain") is not None,
        "summary": _advisor_summary(plan, recommendations),
        "recommendations": recommendations,
        "coach": coach,
        "tools": tools,
        "memory": {
            "scope": "workspace-local",
            "stores": ["training receipts", "dataset manifests", "pipeline bundles"],
            "persistent_chat_memory": False,
        },
    }


def _advisor_recommendations(
    plan: dict[str, Any],
    gates: list[Any],
    dependencies: list[Any],
    estimate: dict[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("state") not in {"blocked", "warning"}:
            continue
        recommendations.append(
            {
                "priority": "high" if gate.get("state") == "blocked" else "medium",
                "title": f"Review {gate.get('gate', 'training gate')}",
                "detail": str(gate.get("detail", "This gate needs review before training.")),
                "action": "Open the related screen and fix the gate before queueing a real run.",
            }
        )
    missing_deps = [dep for dep in dependencies if isinstance(dep, dict) and not dep.get("available")]
    if missing_deps:
        recommendations.append(
            {
                "priority": "high",
                "title": "Install the selected training engine",
                "detail": f"{len(missing_deps)} package(s) are missing for the current method.",
                "action": "Use Settings > Check QLoRA Engine, then install only the engine you need.",
            }
        )
    fit_state = str(estimate.get("fit_state", "")).lower()
    if fit_state in {"tight", "unsafe"}:
        recommendations.append(
            {
                "priority": "high" if fit_state == "unsafe" else "medium",
                "title": "Lower the VRAM risk",
                "detail": f"Current fit state is {fit_state}.",
                "action": "Reduce context length, micro batch, LoRA rank, or enable low VRAM settings.",
            }
        )
    method = str(plan.get("summary", {}).get("method", "qlora")).lower()
    if method in {"dpo", "grpo", "kto", "rloo", "reward_model"}:
        recommendations.append(
            {
                "priority": "medium",
                "title": "Confirm alignment data shape",
                "detail": "Alignment methods need preference pairs or prompt rows with a clear reward path.",
                "action": "Preview the dataset and export a pipeline bundle before running.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": "low",
                "title": "Queue a dry run",
                "detail": "The current plan has no blocking gates.",
                "action": "Create a dry-run receipt before starting a confirmed training job.",
            }
        )
    return recommendations[:6]


def _advisor_summary(plan: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
    status = str(plan.get("status", "unknown"))
    high_count = sum(1 for item in recommendations if item.get("priority") == "high")
    if status == "blocked":
        return f"Training is blocked. Fix {high_count or 1} high-priority item(s) before running."
    if status == "warning":
        return "Training can be planned, but the current settings need review before a real run."
    return "Training is ready for a dry run. Create a receipt before starting the real job."


def _local_coach_payload(
    plan: dict[str, Any],
    recommendations: list[dict[str, Any]],
    request: dict[str, Any],
) -> dict[str, Any]:
    advisor_options = request.get("advisor", {})
    if not isinstance(advisor_options, dict):
        advisor_options = {}
    use_model = bool(advisor_options.get("useModel", advisor_options.get("use_model", False)))
    model = str(advisor_options.get("model") or os.environ.get("RETRAIN_COACH_MODEL") or "gemma:2b")
    url = str(advisor_options.get("ollamaUrl") or os.environ.get("RETRAIN_OLLAMA_URL") or "http://127.0.0.1:11434")
    timeout = float(advisor_options.get("timeoutSeconds") or os.environ.get("RETRAIN_COACH_TIMEOUT", "8"))
    base = {
        "backend": "ollama",
        "model": model,
        "url": url,
        "enabled": use_model,
        "status": "disabled",
        "message": "Local 2B coach is available on demand. It is not called during automatic readiness refresh.",
    }
    if not use_model:
        return base
    prompt = _local_coach_prompt(plan, recommendations)
    try:
        response = _ollama_generate(url=url, model=model, prompt=prompt, timeout=timeout)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {
            **base,
            "enabled": True,
            "status": "unavailable",
            "message": f"Ollama coach unavailable: {exc}",
        }
    return {
        **base,
        "enabled": True,
        "status": "ready",
        "message": response,
    }


def _local_coach_prompt(plan: dict[str, Any], recommendations: list[dict[str, Any]]) -> str:
    summary = plan.get("summary", {}) if isinstance(plan.get("summary"), dict) else {}
    gates = plan.get("validation", {}).get("gates", []) if isinstance(plan.get("validation"), dict) else []
    compact_gates = [
        {
            "gate": gate.get("gate"),
            "state": gate.get("state"),
            "detail": gate.get("detail"),
        }
        for gate in gates
        if isinstance(gate, dict) and gate.get("state") != "ready"
    ][:5]
    compact_recs = [
        {
            "priority": item.get("priority"),
            "title": item.get("title"),
            "action": item.get("action"),
        }
        for item in recommendations[:5]
    ]
    payload = {
        "plan_status": plan.get("status"),
        "summary": summary,
        "problem_gates": compact_gates,
        "recommendations": compact_recs,
    }
    return (
        "You are ReTrain's local training coach. Give concise, practical guidance for a consumer GPU user. "
        "Do not invent files or commands. Prefer safe dry-runs and VRAM safety. "
        "Return 3 bullets max: next step, why, risk.\n\n"
        f"Training state JSON:\n{json.dumps(payload, indent=2)}"
    )


def _ollama_generate(url: str, model: str, prompt: str, timeout: float) -> str:
    endpoint = f"{url.rstrip('/')}/api/generate"
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 180,
                "num_ctx": 2048,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = str(payload.get("response", "")).strip()
    if not text:
        raise ValueError("empty model response")
    return text[:1600]


def path_summary(key: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.exists()
    entries: list[dict[str, Any]] = []
    file_count = 0
    folder_count = 0
    latest_mtime = 0.0
    if exists and resolved.is_dir():
        children = sorted(resolved.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children:
            try:
                stat = child.stat()
            except OSError:
                continue
            latest_mtime = max(latest_mtime, stat.st_mtime)
            if child.is_dir():
                folder_count += 1
            else:
                file_count += 1
            if len(entries) < 16:
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "kind": "folder" if child.is_dir() else "file",
                        "size_bytes": 0 if child.is_dir() else stat.st_size,
                    }
                )
    return {
        "key": key,
        "label": path_label(key),
        "path": str(resolved),
        "exists": exists,
        "kind": "folder" if resolved.is_dir() else "file" if resolved.is_file() else "missing",
        "file_count": file_count,
        "folder_count": folder_count,
        "latest_mtime": latest_mtime,
        "entries": entries,
    }


def path_label(key: str) -> str:
    labels = {
        "root": "Project root",
        "models": "Models",
        "datasets": "Datasets",
        "engines": "Engines",
        "outputs": "Training outputs",
        "receipts": "Receipts",
        "tensorboard": "TensorBoard logs",
        "dataset_exports": "Dataset exports",
        "presets": "Saved presets",
        "exports": "Exports",
    }
    return labels.get(key, key.replace("_", " ").title())


def open_path_payload(request: dict[str, Any]) -> dict[str, Any]:
    key = str(request.get("key", ""))
    requested = WORKSPACE_PATHS.get(key)
    if requested is None and request.get("path"):
        requested = Path(str(request["path"]))
    if requested is None:
        return {"status": "blocked", "message": "Choose a known ReTrain folder first."}

    resolved = requested.expanduser().resolve()
    if not path_is_allowed(resolved):
        return {"status": "blocked", "message": "Only ReTrain project paths can be opened.", "path": str(resolved)}
    if not resolved.exists():
        return {"status": "blocked", "message": "Folder or file does not exist yet.", "path": str(resolved)}
    if not bool(request.get("execute", True)):
        return {"status": "planned", "message": "Path is ready to open.", "path": str(resolved)}

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(resolved))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved)])
        else:
            subprocess.Popen(["xdg-open", str(resolved)])
    except Exception as exc:
        return {"status": "blocked", "message": f"Could not open path: {exc}", "path": str(resolved)}
    return {"status": "opened", "message": "Opened in the system file browser.", "path": str(resolved)}


def delete_artifact_payload(request: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_workspace_path(request.get("path"), "outputs")
    if resolved is None:
        return {"status": "blocked", "message": "Artifact must stay inside the ReTrain workspace."}
    if resolved == ROOT.resolve():
        return {"status": "blocked", "message": "Refusing to delete the project root."}
    if not resolved.exists():
        return {"status": "blocked", "message": "Artifact does not exist.", "path": str(resolved)}
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return {"status": "deleted", "message": "Artifact deleted.", "path": str(resolved)}


def export_artifact_payload(request: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_workspace_path(request.get("path"), "outputs")
    if resolved is None:
        return {"status": "blocked", "message": "Artifact must stay inside the ReTrain workspace."}
    if not resolved.exists():
        return {"status": "blocked", "message": "Artifact does not exist.", "path": str(resolved)}
    export_root = WORKSPACE_PATHS["exports"].resolve() / "artifacts"
    export_root.mkdir(parents=True, exist_ok=True)
    destination = export_root / resolved.name
    if resolved.is_dir():
        destination = export_root / f"{resolved.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copytree(resolved, destination)
    else:
        if destination.exists():
            destination = export_root / f"{resolved.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}{resolved.suffix}"
        shutil.copy2(resolved, destination)
    return {"status": "exported", "message": "Artifact copied into the exports folder.", "path": str(destination)}


def resolve_workspace_path(raw_path: Any, fallback_key: str) -> Path | None:
    requested = Path(str(raw_path)) if raw_path else WORKSPACE_PATHS[fallback_key]
    resolved = requested.expanduser().resolve()
    if path_is_allowed(resolved):
        return resolved
    for allowed_root in WORKSPACE_PATHS.values():
        allowed = allowed_root.expanduser().resolve()
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    return None


def _latest_dataset_export_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def _select_preview_split(dataset_dir: Path) -> tuple[str, Path]:
    for split in ("train", "validation", "eval", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            return split.title(), path
    return "Unknown", dataset_dir / "train.jsonl"


def _read_preview_rows(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(_preview_row_from_record(record, index + 1))
    return rows


def _preview_row_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = ""
    response = ""
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    if isinstance(record.get("messages"), list):
        for item in record["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            content = _flatten_text(item.get("content"))
            if role == "user" and not prompt:
                prompt = content
            if role == "assistant" and not response:
                response = content
    elif isinstance(record.get("conversations"), list):
        for item in record["conversations"]:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("from", "")).lower()
            content = _flatten_text(item.get("value"))
            if speaker == "human" and not prompt:
                prompt = content
            if speaker in {"gpt", "assistant"} and not response:
                response = content
    else:
        prompt = _flatten_text(record.get("prompt", ""))
        response = _flatten_text(record.get("response", record.get("answer", "")))
    token_count = int(metadata.get("rough_tokens", _rough_token_estimate(f"{prompt} {response}")))
    return {
        "id": str(index),
        "prompt": _truncate_text(prompt, 120),
        "response": _truncate_text(response, 140),
        "tokens": token_count,
    }


def _scan_workspace_dataset_rows(root: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.name.lower()):
        if len(rows) >= limit or not path.is_file():
            continue
        if path.suffix.lower() not in {".jsonl", ".json", ".txt", ".md", ".markdown", ".csv"}:
            continue
        text = _read_text_preview(path)
        if not text:
            continue
        rows.append(
            {
                "id": str(len(rows) + 1),
                "prompt": f"Source file: {path.name}",
                "response": _truncate_text(text, 140),
                "tokens": _rough_token_estimate(text),
            }
        )
    return rows


def _collect_run_artifacts() -> list[dict[str, Any]]:
    receipt_root = WORKSPACE_PATHS["receipts"].resolve()
    if not receipt_root.exists():
        return []
    items = []
    for path in sorted(receipt_root.glob("retrain-training-*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = _read_json_file(path)
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        created_at = str(payload.get("created_at", _format_timestamp(path.stat().st_mtime)))
        status = str(payload.get("status", "unknown"))
        label = f"{summary.get('method', 'run')} | {summary.get('model', 'Unknown model')}"
        items.append(
            {
                "id": path.stem,
                "label": label,
                "note": status,
                "time": created_at.replace("T", " "),
                "step": "n/a",
                "loss": "n/a",
                "lr": "n/a",
                "vram": f"{summary.get('estimated_gb', 'n/a')} GB" if summary.get("estimated_gb") is not None else "n/a",
                "path": str(path),
                "created_at": created_at,
                "status": status,
                "kind": "receipt",
                "type": "Training receipt",
                "size": _format_size(path.stat().st_size),
                "model": str(summary.get("model", "Unknown model")),
                "method": str(summary.get("method", "unknown")),
                "fit_state": str(summary.get("fit_state", "unknown")),
                "estimated_gb": summary.get("estimated_gb", "n/a"),
                "dataset_version": str(summary.get("dataset_version", worker.ACTIVE_DATASET_VERSION)),
            }
        )
    return items


def _compare_metric_row(
    label: str,
    baseline: dict[str, Any] | None,
    candidate_a: dict[str, Any] | None,
    candidate_b: dict[str, Any] | None,
    *,
    key: str,
) -> dict[str, Any]:
    base_value = baseline.get(key, "n/a") if baseline else "n/a"
    a_value = candidate_a.get(key, "n/a") if candidate_a else "n/a"
    b_value = candidate_b.get(key, "n/a") if candidate_b else "n/a"
    return {
        "label": label,
        "baseline": str(base_value),
        "candidate_a": str(a_value),
        "candidate_b": str(b_value),
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text_preview(path: Path, char_limit: int = 800) -> str:
    try:
        return _flatten_text(path.read_text(encoding="utf-8", errors="ignore"))[:char_limit]
    except Exception:
        return ""


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value if item)
    return str(value).strip()


def _truncate_text(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else f"{value[: limit - 3].rstrip()}..."


def _rough_token_estimate(value: str) -> int:
    return max(1, round(len(value.split()) * 1.35))


def _format_token_estimate(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M %p")


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def _safe_filename(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return cleaned or "preset"


def path_is_allowed(path: Path) -> bool:
    root = ROOT.resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        pass
    for allowed_root in WORKSPACE_PATHS.values():
        allowed = allowed_root.expanduser().resolve()
        try:
            path.relative_to(allowed)
            return True
        except ValueError:
            continue
    return False


def request_to_training_payload(request: dict[str, Any]) -> dict[str, Any]:
    settings = request.get("settings", request)
    if not isinstance(settings, dict):
        settings = {}
    payload: dict[str, Any] = {
        "model_key": MODEL_ID_MAP.get(str(settings.get("modelId", "")), "qwen2.5-coder-1.5b"),
        "method": METHOD_MAP.get(str(settings.get("method", "")), "qlora"),
        "tune_scope": TUNE_SCOPE_MAP.get(str(settings.get("tuneScope", "")), "last_n_layers"),
        "last_n_layers": settings.get("lastNLayers", 4),
        "beta": settings.get("beta", 0.1),
        "num_generations": settings.get("numGenerations", 2),
        "reward_kind": REWARD_KIND_MAP.get(str(settings.get("rewardKind", "")), "heuristic"),
        "context_length": settings.get("contextLength", 4096),
        "micro_batch_size": settings.get("microBatch", 1),
        "gradient_accumulation_steps": settings.get("gradAccum", 8),
        "lora_rank": settings.get("loraRank", 16),
        "precision": PRECISION_MAP.get(str(settings.get("precision", "")), "4bit"),
        "optimizer": OPTIMIZER_MAP.get(str(settings.get("optimizer", "")), "paged_adamw_8bit"),
        "scheduler": SCHEDULER_MAP.get(str(settings.get("scheduler", "")), "cosine"),
        "trust_remote_code": settings.get("trustRemoteCode", False),
        "gradient_checkpointing": settings.get("gradientCheckpointing", True),
        "flash_attention": settings.get("flashAttention", True),
        "qlora": settings.get("qlora", True),
        "bitsandbytes": settings.get("qlora", True),
        "cpu_offload": settings.get("cpuOffload", False) or settings.get("optimizer") == "CPU AdamW",
        "low_vram_mode": settings.get("lowVramMode", True),
        "tensorboard": settings.get("tensorboard", True),
        "confirmed": settings.get("confirmed", False),
        "dry_run": settings.get("dryRun", True),
    }
    overrides = request.get("payload", {})
    if isinstance(overrides, dict):
        payload.update(overrides)
    return payload


def sanitize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    launch_spec = plan.get("internal_launch_spec", {})
    training_args = launch_spec.get("training_args", {}) if isinstance(launch_spec, dict) else {}
    return sanitize_value(
        {
            "status": plan.get("status"),
            "start_enabled": plan.get("start_enabled"),
            "summary": plan.get("summary", {}),
            "validation": plan.get("validation", {}),
            "training_args": training_args,
        }
    )


def sanitize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return sanitize_value(
        {
            "schema_version": receipt.get("schema_version"),
            "created_at": receipt.get("created_at"),
            "status": receipt.get("status"),
            "execute_requested": receipt.get("execute_requested"),
            "summary": receipt.get("summary", {}),
            "validation": receipt.get("validation", {}),
            "outputs": receipt.get("outputs", {}),
            "process": receipt.get("process"),
            "error": receipt.get("error"),
        }
    )


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items() if key not in SENSITIVE_KEYS}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gui.api.app:app", host="127.12.6.3", port=8787, reload=False)
