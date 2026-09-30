from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from gui.gradio import backend_worker as worker


PIPELINE_SCHEMA_VERSION = "retrain-pipeline-v1"


def preview_pipeline_bundle(
    training_payload: dict[str, Any] | None,
    request: dict[str, Any] | None,
    workspace_paths: dict[str, Path],
) -> dict[str, Any]:
    normalized = worker.normalize_payload(training_payload or {})
    config = asdict(normalized)
    request = request or {}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = _safe_name(config["model_key"])
    method_slug = _safe_name(config["method"])
    bundle_id = f"{timestamp}-{method_slug}-{model_slug}"
    output_root = Path(config["output_root"]).expanduser().resolve()
    bundle_root = output_root / "pipelines" / bundle_id
    export_name = str(request.get("exportName", f"retrain-{method_slug}-{model_slug}")).strip() or f"retrain-{method_slug}-{model_slug}"
    deploy_target = str(request.get("deployTarget", config["backend_target"]))
    eval_tasks = _normalize_eval_tasks(request.get("evalTasks"))
    artifact_root = workspace_paths["outputs"].resolve() / "pipelines"

    verl_spec = build_verl_spec(config, request)
    nemo_spec = build_nemo_eval_spec(config, request, eval_tasks, bundle_root / "nemo-eval")
    llama_spec = build_llama_cpp_spec(config, request, export_name, deploy_target, bundle_root / "llama-cpp")
    agent_spec = build_agent_workflow_spec(config, request, bundle_id)
    infra_spec = build_infra_readiness_spec(config, request, eval_tasks)

    pipelines = [
        {
            "id": "verl",
            "title": "VERL Training Pipeline",
            "status": "ready" if config["method"] in {"grpo", "rloo", "dpo", "kto", "reward_model"} else "adapt",
            "summary": verl_spec["summary"],
            "artifact": "01-verl-training.json",
            "spec": verl_spec,
        },
        {
            "id": "nemo-eval",
            "title": "NeMo Evaluator Pipeline",
            "status": "ready",
            "summary": nemo_spec["summary"],
            "artifact": "02-nemo-evaluator.json",
            "spec": nemo_spec,
        },
        {
            "id": "llama-cpp",
            "title": "llama.cpp Deployment Pipeline",
            "status": "ready",
            "summary": llama_spec["summary"],
            "artifact": "03-llama-cpp-deploy.json",
            "spec": llama_spec,
        },
        {
            "id": "autogpt",
            "title": "AutoGPT Workflow Graph",
            "status": "ready",
            "summary": agent_spec["summary"],
            "artifact": "04-autogpt-workflow.json",
            "spec": agent_spec,
        },
        {
            "id": "infra",
            "title": "Scale Readiness",
            "status": "review",
            "summary": infra_spec["summary"],
            "artifact": "05-infra-readiness.json",
            "spec": infra_spec,
        },
    ]

    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_root": str(workspace_paths["root"].resolve()),
        "artifact_root": str(artifact_root),
        "bundle_root": str(bundle_root),
        "model_key": config["model_key"],
        "method": config["method"],
        "dataset_path": config["dataset_path"],
        "deploy_target": deploy_target,
        "export_name": export_name,
        "pipeline_count": len(pipelines),
        "pipelines": [
            {
                "id": item["id"],
                "title": item["title"],
                "status": item["status"],
                "summary": item["summary"],
                "artifact": item["artifact"],
            }
            for item in pipelines
        ],
    }
    return {"manifest": manifest, "pipelines": pipelines}


def export_pipeline_bundle(
    training_payload: dict[str, Any] | None,
    request: dict[str, Any] | None,
    workspace_paths: dict[str, Path],
) -> dict[str, Any]:
    bundle = preview_pipeline_bundle(training_payload, request, workspace_paths)
    manifest = bundle["manifest"]
    bundle_root = Path(manifest["bundle_root"])
    bundle_root.mkdir(parents=True, exist_ok=True)

    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    exported_files = [str(manifest_path)]
    for item in bundle["pipelines"]:
        artifact_path = bundle_root / item["artifact"]
        artifact_path.write_text(json.dumps(item["spec"], indent=2), encoding="utf-8")
        exported_files.append(str(artifact_path))

    return {
        "status": "exported",
        "bundle_id": manifest["bundle_id"],
        "bundle_root": str(bundle_root),
        "manifest_path": str(manifest_path),
        "file_count": len(exported_files),
        "files": exported_files,
        "pipelines": manifest["pipelines"],
    }


def build_verl_spec(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    method = str(config["method"])
    algorithm = {
        "grpo": "grpo",
        "rloo": "rloo",
        "dpo": "dpo",
        "kto": "kto",
        "reward_model": "reward_model",
    }.get(method, "adapter_sft_bridge")
    prompt_source = "train.jsonl prompt rows" if method in {"grpo", "rloo"} else "preference pairs"
    return {
        "schema_version": "verl-blueprint-v1",
        "summary": f"{algorithm.upper()} recipe for {config['model_key']} with {config['precision']} precision.",
        "runtime": {
            "framework": "verl",
            "mode": algorithm,
            "notes": [
                "Keep the desktop path on TRL for day-one training; treat this spec as the promotion path into verl.",
                "Use the same dataset contract the UI already collects so migration is mechanical.",
            ],
        },
        "model": {
            "base_model_path": config["model_path"],
            "model_key": config["model_key"],
            "precision": config["precision"],
            "context_length": config["context_length"],
            "backend_target": config["backend_target"],
        },
        "data": {
            "dataset_path": config["dataset_path"],
            "dataset_format": config["dataset_format"],
            "prompt_source": prompt_source,
            "train_file": "train.jsonl",
            "validation_file": "validation.jsonl",
            "expected_columns": ["prompt", "chosen", "rejected"] if method in {"dpo", "kto", "reward_model"} else ["prompt"],
        },
        "trainer": {
            "micro_batch_size": config["micro_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "warmup_ratio": config["warmup_ratio"],
            "save_every_steps": config["save_every_steps"],
            "eval_every_steps": config["eval_every_steps"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "flash_attention": config["flash_attention"],
            "sequence_packing": config["sequence_packing"],
        },
        "rollout": {
            "num_generations": config["num_generations"],
            "reward_kind": config["reward_kind"],
            "beta": config["beta"],
            "judge_backend": str(request.get("judgeBackend", "ollama")),
            "judge_model": str(request.get("judgeModel", "qwen2.5-1.5b-instruct")),
        },
        "promotion_checks": [
            "Dataset rows validate under the local TRL runner first.",
            "Reward function is deterministic enough for offline replay.",
            "A vLLM or OpenAI-compatible endpoint exists before distributed rollouts.",
        ],
    }


def build_nemo_eval_spec(config: dict[str, Any], request: dict[str, Any], eval_tasks: list[str], output_dir: Path) -> dict[str, Any]:
    eval_backend = str(request.get("evalBackend", "openai-compatible"))
    endpoint = str(request.get("evalEndpoint", "http://127.12.6.3:8001/v1"))
    judge_model = str(request.get("judgeModel", "qwen2.5-7b-judge"))
    return {
        "schema_version": "nemo-evaluator-blueprint-v1",
        "summary": f"{len(eval_tasks)} eval task(s) routed through {eval_backend}.",
        "runtime": {
            "framework": "nemo-evaluator",
            "output_dir": str(output_dir),
            "max_concurrency": int(request.get("evalConcurrency", 4)),
        },
        "target_model": {
            "model_key": config["model_key"],
            "endpoint_backend": eval_backend,
            "endpoint_url": endpoint,
            "max_batch_size": int(request.get("evalBatchSize", 8)),
            "context_length": config["context_length"],
        },
        "judge": {
            "mode": str(request.get("judgeMode", "pairwise")),
            "model": judge_model,
            "temperature": float(request.get("judgeTemperature", 0.0)),
        },
        "tasks": [
            {
                "name": task,
                "type": "benchmark" if task in {"gsm8k", "mmlu", "ifeval"} else "custom",
                "split": "validation",
                "metrics": ["accuracy", "win_rate"] if task in {"arena-hard", "mt-bench"} else ["accuracy"],
            }
            for task in eval_tasks
        ],
        "artifacts": {
            "predictions_file": str(output_dir / "predictions.jsonl"),
            "summary_file": str(output_dir / "summary.json"),
            "trace_dir": str(output_dir / "traces"),
        },
    }


def build_llama_cpp_spec(config: dict[str, Any], request: dict[str, Any], export_name: str, deploy_target: str, output_dir: Path) -> dict[str, Any]:
    quantization = str(request.get("quantization", "Q4_K_M"))
    adapter_path = str(request.get("adapterPath", output_dir.parent / "adapter"))
    base_model_path = config["model_path"] or str(worker.DEFAULT_MODEL_ROOT / config["model_key"])
    return {
        "schema_version": "llama-cpp-blueprint-v1",
        "summary": f"{quantization} export plan for {deploy_target}.",
        "runtime": {
            "framework": "llama.cpp",
            "output_dir": str(output_dir),
            "quantization": quantization,
            "gpu_layers": int(request.get("gpuLayers", 99)),
            "runtime_context": int(request.get("runtimeContext", config["context_length"])),
        },
        "inputs": {
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "merge_adapter": bool(request.get("mergeAdapter", True)),
            "copy_to_target": bool(request.get("copyToTarget", False)),
            "target": deploy_target,
            "export_name": export_name,
        },
        "steps": [
            "Validate base model and adapter paths.",
            "Merge LoRA adapter into the base checkpoint when requested.",
            "Convert merged weights into GGUF.",
            f"Quantize GGUF to {quantization}.",
            "Run a one-prompt smoke test before copying into the target app.",
        ],
        "artifacts": {
            "merged_model_dir": str(output_dir / "merged-model"),
            "gguf_path": str(output_dir / f"{export_name}.gguf"),
            "modelfile_path": str(output_dir / "Modelfile"),
            "smoke_report_path": str(output_dir / "smoke-test.json"),
        },
    }


def build_agent_workflow_spec(config: dict[str, Any], request: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    agent_mode = str(request.get("agentMode", "preference-labeller"))
    return {
        "schema_version": "autogpt-workflow-blueprint-v1",
        "summary": f"{agent_mode} loop for {config['method']} receipts and evaluation artifacts.",
        "runtime": {
            "framework": "autogpt-style workflow",
            "bundle_id": bundle_id,
            "checkpoint_policy": "persist after every review stage",
        },
        "agents": [
            {"id": "dataset-curator", "goal": "Watch new exports and flag schema drift."},
            {"id": "trainer-operator", "goal": "Queue local runs only when VRAM and engine gates pass."},
            {"id": "eval-analyst", "goal": "Compare eval outputs and rank candidate checkpoints."},
            {"id": "deployment-operator", "goal": "Prepare GGUF and target-specific packaging once a run is approved."},
        ],
        "graph": {
            "nodes": [
                {"id": "dataset_watch", "type": "watch", "label": "Dataset export watcher"},
                {"id": "train_receipt", "type": "job", "label": "Training receipt review"},
                {"id": "nemo_eval", "type": "eval", "label": "NeMo evaluator pass"},
                {"id": "human_gate", "type": "approval", "label": "Human sign-off"},
                {"id": "deploy_bundle", "type": "deploy", "label": "GGUF + target packaging"},
            ],
            "edges": [
                ["dataset_watch", "train_receipt"],
                ["train_receipt", "nemo_eval"],
                ["nemo_eval", "human_gate"],
                ["human_gate", "deploy_bundle"],
            ],
        },
        "memory": {
            "receipts_root": str(worker.DEFAULT_RECEIPT_ROOT),
            "compare_runs_endpoint": "/api/retrain/compare",
            "checkpoint_inventory_endpoint": "/api/retrain/checkpoints",
        },
    }


def build_infra_readiness_spec(config: dict[str, Any], request: dict[str, Any], eval_tasks: list[str]) -> dict[str, Any]:
    stage = str(request.get("clusterProfile", "single-gpu-dev-box"))
    current_vram = float(config["max_vram_gb"])
    distributed_ready = config["method"] in {"grpo", "rloo"} and current_vram >= 24
    return {
        "schema_version": "infra-readiness-blueprint-v1",
        "summary": f"Current stage: {stage}. Distributed-ready: {'yes' if distributed_ready else 'not yet'}.",
        "current_profile": {
            "stage": stage,
            "max_vram_gb": current_vram,
            "free_vram_gb": float(config["free_vram_gb"]),
            "tensorboard": bool(config["tensorboard"]),
            "eval_tasks": eval_tasks,
        },
        "promotion_path": [
            {
                "stage": "desktop",
                "entry_rule": "Single consumer GPU, local datasets, one active run.",
                "must_have": ["dry-run receipts", "TensorBoard", "exportable pipeline manifests"],
            },
            {
                "stage": "workstation",
                "entry_rule": "24 GB+ VRAM or multiple local GPUs.",
                "must_have": ["dedicated eval endpoint", "persistent checkpoints", "GGUF smoke tests"],
            },
            {
                "stage": "cluster",
                "entry_rule": "Distributed RL or multi-node eval becomes routine.",
                "must_have": ["shared object store", "job scheduler", "centralized telemetry", "separate rollout and judge pools"],
            },
        ],
        "gaps": [
            "No distributed launcher is wired into the UI yet.",
            "No central queue or retry policy exists for failed eval shards.",
            "No remote artifact registry is configured for multi-machine promotion.",
        ],
    }


def _normalize_eval_tasks(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = []
    return values or ["gsm8k", "ifeval", "arena-hard"]


def _safe_name(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return cleaned or "bundle"
