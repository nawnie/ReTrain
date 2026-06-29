#!/usr/bin/env python3
"""Run local full-SFT, LoRA, and QLoRA smoke training for ReTrain."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "training" / "posttrain_smoke" / "data"
DEFAULT_OUTPUT_ROOT = ROOT / "training" / "runs"
DEFAULT_MODELS = {
    "qwen2.5-coder-1.5b": {
        "path": Path(r"F:\Ai_Models\hf\posttrain_candidates\Qwen--Qwen2.5-Coder-1.5B"),
        "family": "qwen-coder",
        "fit_note": "Safest full-SFT and LoRA starter on this 16GB-class local GPU.",
    },
    "smollm3-3b-instruct": {
        "path": Path(r"F:\Ai_Models\hf\posttrain_candidates\HuggingFaceTB--SmolLM3-3B"),
        "family": "smollm3",
        "fit_note": "3B instruct model with dual-mode reasoning behavior; full-SFT smoke may fit only with short context.",
    },
    "gemma4-e2b": {
        "path": Path(r"F:\Ai_Models\hf\posttrain_candidates\google--gemma-4-E2B"),
        "family": "gemma4",
        "fit_note": "Gemma 4 E2B base weights fit on disk, but full-weight SFT is expected to be tight on 16GB VRAM.",
    },
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    family: str
    fit_note: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_model_specs(names: list[str], *, model_path: Path | None = None, model_name: str = "") -> list[ModelSpec]:
    if model_path is not None:
        name = model_name or model_path.name or "selected-model"
        return [
            ModelSpec(
                name=name,
                path=model_path,
                family="custom",
                fit_note="User-selected local model path from ReTrain.",
            )
        ]
    selected = list(DEFAULT_MODELS) if names == ["all"] else names
    specs = []
    for name in selected:
        if name not in DEFAULT_MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(DEFAULT_MODELS)} or all")
        raw = DEFAULT_MODELS[name]
        specs.append(ModelSpec(name=name, path=raw["path"], family=raw["family"], fit_note=raw["fit_note"]))
    return specs


def validate_model_path(spec: ModelSpec) -> list[str]:
    issues = []
    required = ["config.json", "tokenizer.json"]
    for filename in required:
        if not (spec.path / filename).exists():
            issues.append(f"{spec.name}: missing {filename} at {spec.path}")
    if not list(spec.path.glob("*.safetensors")):
        issues.append(f"{spec.name}: no safetensors found at {spec.path}")
    return issues


def split_counts(data_dir: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "validation", "test", "eval_prompts"):
        path = data_dir / f"{split}.jsonl"
        counts[split] = len(read_jsonl(path)) if path.exists() else 0
    return counts


def manual_chat_template(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    parts = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", "")).strip()
        parts.append(f"[{role}]\n{content}")
    if add_generation_prompt:
        parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts).strip() + "\n"


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return manual_chat_template(messages, add_generation_prompt=add_generation_prompt)


def build_supervised_example(tokenizer: Any, row: dict[str, Any], *, max_seq_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    assistant_index = max(i for i, msg in enumerate(messages) if msg.get("role") == "assistant")
    prompt_messages = messages[:assistant_index]
    full_text = render_chat(tokenizer, messages[: assistant_index + 1], add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)

    full = tokenizer(full_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
    prompt_len = min(len(prompt["input_ids"]), len(full["input_ids"]))
    labels = [-100] * prompt_len + full["input_ids"][prompt_len:]
    labels = labels[: len(full["input_ids"])]
    if not any(label != -100 for label in labels):
        # Some chat templates render the generation prompt longer than the
        # rendered assistant turn. Keep smoke training meaningful instead of
        # silently producing zero supervised tokens.
        labels = list(full["input_ids"])
    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


def import_training_stack() -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


class SupervisedDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, *, max_seq_length: int):
        self.examples = [
            build_supervised_example(tokenizer, row, max_seq_length=max_seq_length)
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(example["input_ids"]) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            pad = max_len - len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(example["attention_mask"] + [0] * pad)
            batch["labels"].append(example["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def training_arguments(cls: Any, *, output_dir: Path, args: argparse.Namespace) -> Any:
    tensorboard_enabled = importlib.util.find_spec("tensorboard") is not None
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "logging_dir": str(output_dir / "tensorboard"),
        "report_to": ["tensorboard"] if tensorboard_enabled else [],
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": args.optim,
    }
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    if args.save_checkpoints:
        kwargs["save_steps"] = args.max_steps
        kwargs["save_total_limit"] = 1
    elif "save_strategy" in params:
        kwargs["save_strategy"] = "no"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps"
    if "fp16" in params:
        kwargs["fp16"] = args.precision == "fp16"
    if "bf16" in params:
        kwargs["bf16"] = args.precision == "bf16"
    return cls(**kwargs)


def _tensorboard_scalars_from_mapping(prefix: str, mapping: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        normalized_key = key
        for strip_prefix in ("train_", "eval_", "generation_", "cuda_"):
            if normalized_key.startswith(strip_prefix):
                normalized_key = normalized_key[len(strip_prefix) :]
                break
        scalars[f"{prefix}/{normalized_key}"] = float(value)
    return scalars


def tensorboard_scalar_items(metrics: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    train_metrics = metrics.get("train_metrics", {})
    if isinstance(train_metrics, dict):
        scalars.update(_tensorboard_scalars_from_mapping("train", train_metrics))
    eval_metrics = metrics.get("eval_metrics", {})
    if isinstance(eval_metrics, dict):
        scalars.update(_tensorboard_scalars_from_mapping("eval", eval_metrics))
    generation_metrics = metrics.get("generation_metrics", {})
    if isinstance(generation_metrics, dict):
        scalars.update(_tensorboard_scalars_from_mapping("generation", generation_metrics))
    cuda_metrics = metrics.get("cuda", {})
    if isinstance(cuda_metrics, dict):
        scalars.update(_tensorboard_scalars_from_mapping("cuda", cuda_metrics))

    runtime_seconds = metrics.get("train_runtime_seconds")
    if isinstance(runtime_seconds, (int, float)) and not isinstance(runtime_seconds, bool) and math.isfinite(
        float(runtime_seconds)
    ):
        scalars["run/runtime_seconds"] = float(runtime_seconds)

    eval_perplexity = metrics.get("eval_perplexity")
    if isinstance(eval_perplexity, (int, float)) and not isinstance(eval_perplexity, bool) and math.isfinite(
        float(eval_perplexity)
    ):
        scalars["eval/perplexity"] = float(eval_perplexity)

    return scalars


def write_tensorboard_logs(
    log_dir: Path,
    metrics: dict[str, Any],
    *,
    writer_factory: Any | None = None,
) -> dict[str, Any]:
    scalars = tensorboard_scalar_items(metrics)
    result: dict[str, Any] = {
        "log_dir": str(log_dir),
        "scalar_count": len(scalars),
        "tags": sorted(scalars)[:25],
        "written": False,
    }
    if not scalars:
        result["status"] = "no_scalars"
        return result

    if writer_factory is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            result["status"] = "unavailable"
            result["error"] = str(exc)
            return result
        writer_factory = SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = writer_factory(log_dir=str(log_dir))
    try:
        for tag, value in sorted(scalars.items()):
            writer.add_scalar(tag, value, global_step=1)
        writer.flush()
    finally:
        writer.close()

    result["written"] = True
    result["status"] = "ready"
    return result


def run_dir_for(output_root: Path, spec: ModelSpec, method: str) -> Path:
    return output_root / f"{spec.name}-{method}"


def parse_lora_target_modules(value: str) -> str | list[str]:
    if value == "all-linear":
        return value
    return [part.strip() for part in value.split(",") if part.strip()]


LAYER_INDEX_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)",
        r"(?:^|\.)gpt_neox\.layers\.(\d+)(?:\.|$)",
    )
]


def parameter_layer_index(name: str) -> int | None:
    for pattern in LAYER_INDEX_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def infer_layer_count(model: Any) -> int:
    config = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    layer_indices = [
        index
        for name, _param in model.named_parameters()
        for index in [parameter_layer_index(name)]
        if index is not None
    ]
    return max(layer_indices) + 1 if layer_indices else 0


def is_output_head_or_final_norm(name: str) -> bool:
    if name.startswith(("lm_head.", "score.", "embed_out.", "output.")):
        return True
    return name in {
        "model.norm.weight",
        "model.norm.bias",
        "norm.weight",
        "norm.bias",
        "ln_f.weight",
        "ln_f.bias",
        "final_layernorm.weight",
        "final_layernorm.bias",
    }


def apply_tune_scope(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.tune_scope == "all":
        return {"tune_scope": "all"}
    if args.method != "full_sft":
        raise ValueError("--tune-scope is only supported with --method full_sft")

    for _name, param in model.named_parameters():
        param.requires_grad = False

    layer_count = infer_layer_count(model)
    cutoff = max(0, layer_count - args.last_n_layers)
    trainable_names = []
    for name, param in model.named_parameters():
        layer_index = parameter_layer_index(name)
        should_train = is_output_head_or_final_norm(name)
        if args.tune_scope == "last_n_layers":
            should_train = should_train or (
                layer_index is not None and layer_index >= cutoff
            )
        elif args.tune_scope == "lm_head":
            should_train = should_train and layer_index is None
        if should_train:
            param.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise ValueError(f"{args.tune_scope} did not match any trainable parameters")
    return {
        "tune_scope": args.tune_scope,
        "layer_count": layer_count,
        "last_n_layers": args.last_n_layers if args.tune_scope == "last_n_layers" else None,
        "trainable_parameter_name_count": len(trainable_names),
        "trainable_parameter_name_examples": trainable_names[:8],
    }


def apply_lora(model: Any, stack: dict[str, Any], args: argparse.Namespace) -> Any:
    config = stack["LoraConfig"](
        task_type=stack["TaskType"].CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_lora_target_modules(args.lora_target_modules),
        bias="none",
    )
    return stack["get_peft_model"](model, config)


def trainable_parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round((trainable / total) * 100, 6) if total else 0.0,
    }


def required_tags(text: str) -> list[str]:
    known_tags = ("[TASK]", "[CODE]", "[INTERP]", "[ROUTE]", "[CHECK]", "[GATE]", "[FINAL]")
    return [tag for tag in known_tags if tag in text]


def run_generation_test(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    rows = read_jsonl(args.data_dir / "eval_prompts.jsonl")[: args.max_test_records]
    outputs = []
    for row in rows:
        prompt_text = render_chat(tokenizer, row["messages"], add_generation_prompt=True)
        encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_length)
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        expected = str(row.get("expected", ""))
        expected_tags = required_tags(expected)
        hit_tags = [tag for tag in expected_tags if tag in output_text]
        outputs.append(
            {
                "id": row["id"],
                "prompt_messages": row["messages"],
                "expected": expected,
                "output": output_text,
                "expected_tags": expected_tags,
                "hit_tags": hit_tags,
            }
        )

    write_jsonl(run_dir / "generations.jsonl", outputs)
    total_expected_tags = sum(len(row["expected_tags"]) for row in outputs)
    total_hit_tags = sum(len(row["hit_tags"]) for row in outputs)
    return {
        "generation_records": len(outputs),
        "max_new_tokens": args.max_new_tokens,
        "required_tag_hit_rate": (total_hit_tags / total_expected_tags) if total_expected_tags else None,
    }


def run_one_model(spec: ModelSpec, args: argparse.Namespace) -> dict[str, Any]:
    stack = import_training_stack()
    torch = stack["torch"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]

    train_rows = read_jsonl(args.data_dir / "train.jsonl")[: args.max_train_records]
    val_rows = read_jsonl(args.data_dir / "validation.jsonl")[: args.max_eval_records]
    run_dir = run_dir_for(args.output_root, spec, args.method)
    run_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    started = time.time()
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.method == "qlora":
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(spec.path, **model_kwargs)
    if args.method == "qlora":
        model = stack["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
        model = apply_lora(model, stack, args)
        tune_scope = {"tune_scope": "qlora", "quantization": "4bit-nf4"}
    elif args.method == "lora":
        model = apply_lora(model, stack, args)
        tune_scope = {"tune_scope": "lora"}
    else:
        tune_scope = apply_tune_scope(model, args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if torch.cuda.is_available() and args.method != "qlora":
        model.to("cuda")
    parameter_counts = trainable_parameter_counts(model)

    trainer = Trainer(
        model=model,
        args=training_arguments(TrainingArguments, output_dir=run_dir, args=args),
        train_dataset=SupervisedDataset(train_rows, tokenizer, max_seq_length=args.max_seq_length),
        eval_dataset=SupervisedDataset(val_rows, tokenizer, max_seq_length=args.max_seq_length),
        data_collator=CausalCollator(tokenizer),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    if args.save_final:
        trainer.save_model(str(run_dir / "final"))
        tokenizer.save_pretrained(str(run_dir / "final"))
    generation_metrics = (
        None
        if args.skip_generation
        else run_generation_test(model=model, tokenizer=tokenizer, torch=torch, args=args, run_dir=run_dir)
    )

    metrics = {
        "model": spec.name,
        "base_path": str(spec.path),
        "method": args.method,
        "tune_scope": tune_scope,
        "parameter_counts": parameter_counts,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "save_checkpoints": args.save_checkpoints,
        "save_final": args.save_final,
        "train_runtime_seconds": round(time.time() - started, 3),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": eval_result,
        "generation_metrics": generation_metrics,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if "eval_loss" in eval_result:
        metrics["eval_perplexity"] = math.exp(eval_result["eval_loss"]) if eval_result["eval_loss"] < 20 else None
    if torch.cuda.is_available():
        metrics["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
        }
    metrics["tensorboard"] = write_tensorboard_logs(run_dir / "tensorboard", metrics)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def dry_run(specs: list[ModelSpec], data_dir: Path) -> dict[str, Any]:
    model_reports = []
    for spec in specs:
        model_reports.append(
            {
                "name": spec.name,
                "path": str(spec.path),
                "exists": spec.path.exists(),
                "issues": validate_model_path(spec),
                "fit_note": spec.fit_note,
            }
        )
    return {
        "schema_version": "retrain-posttrain-smoke-dry-run-v1",
        "data_dir": str(data_dir),
        "split_counts": split_counts(data_dir),
        "models": model_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ReTrain post-training smoke runs.")
    parser.add_argument("--model", nargs="+", default=["all"], help="Model key(s), or all.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional local model folder selected outside the built-in bakeoff list.")
    parser.add_argument("--model-name", default="", help="Display name for --model-path runs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/model paths without importing ML stack.")
    parser.add_argument("--method", choices=["full_sft", "lora", "qlora"], default="full_sft")
    parser.add_argument("--tune-scope", choices=["all", "last_n_layers", "lm_head"], default="all")
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--max-train-records", type=int, default=8)
    parser.add_argument("--max-eval-records", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-test-records", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--optim", default="adafactor")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--save-final", action="store_true", default=True)
    parser.add_argument("--no-save-final", action="store_false", dest="save_final")
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cuda-alloc-conf", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    args = parser.parse_args()
    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()

    specs = load_model_specs(args.model, model_path=args.model_path, model_name=args.model_name)
    if args.dry_run:
        print(json.dumps(dry_run(specs, args.data_dir), indent=2, sort_keys=True))
        return 0

    results = []
    failed = False
    for spec in specs:
        try:
            results.append(run_one_model(spec, args))
        except Exception as exc:
            failed = True
            run_dir = run_dir_for(args.output_root, spec, args.method)
            run_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "model": spec.name,
                "base_path": str(spec.path),
                "method": args.method,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (run_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results.append(failure)
    print(json.dumps({"status": "completed", "results": results}, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
