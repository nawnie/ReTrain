#!/usr/bin/env python3
"""Run lightweight text-side training targets for ReTrain."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def row_to_pair(row: dict[str, Any], dataset_format: str) -> tuple[str, str]:
    fmt = dataset_format.lower()
    if fmt in {"auto", "prompt_completion"}:
        prompt = row.get("prompt") or row.get("instruction") or row.get("input")
        completion = row.get("completion") or row.get("response") or row.get("output")
        if isinstance(prompt, str) and isinstance(completion, str):
            return prompt, completion
    if fmt in {"auto", "messages"} and isinstance(row.get("messages"), list):
        messages = [item for item in row["messages"] if isinstance(item, dict)]
        user = next((str(item.get("content", "")) for item in messages if item.get("role") == "user"), "")
        assistant = next((str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "assistant"), "")
        if user and assistant:
            return user, assistant
    if fmt in {"auto", "text"} and isinstance(row.get("text"), str):
        text = row["text"].strip()
        if text:
            return text, text
    raise ValueError("Row does not match prompt/completion, messages, or text format.")


def row_to_text(row: dict[str, Any], dataset_format: str) -> str:
    fmt = dataset_format.lower()
    if fmt in {"auto", "text"} and isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"].strip()
    source, target = row_to_pair(row, dataset_format)
    return f"{source}\n{target}".strip()


def split_counts(data_dir: Path) -> dict[str, int]:
    return {
        split: len(read_jsonl(data_dir / f"{split}.jsonl"))
        for split in ("train", "validation", "test", "eval_prompts")
    }


def validate_model_path(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        issues.append(f"model path is missing: {path}")
        return issues
    if not (path / "config.json").exists():
        issues.append(f"missing config.json at {path}")
    if not ((path / "tokenizer.json").exists() or (path / "spiece.model").exists()):
        issues.append(f"missing tokenizer.json or spiece.model at {path}")
    if not list(path.glob("*.safetensors")) and not list(path.glob("*.bin")):
        issues.append(f"no safetensors or bin weights found at {path}")
    return issues


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    validation_rows = read_jsonl(args.data_dir / "validation.jsonl")
    sample_errors: list[str] = []
    for split_name, rows in (("train", train_rows[:3]), ("validation", validation_rows[:3])):
        for index, row in enumerate(rows, start=1):
            try:
                if args.target_type == "seq2seq_t5":
                    row_to_pair(row, args.dataset_format)
                else:
                    row_to_text(row, args.dataset_format)
            except Exception as exc:
                sample_errors.append(f"{split_name}:{index}: {exc}")
    return {
        "schema_version": "retrain-text-target-dry-run-v1",
        "target_type": args.target_type,
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "dataset_format": args.dataset_format,
        "split_counts": split_counts(args.data_dir),
        "model_issues": validate_model_path(args.model_path),
        "sample_errors": sample_errors,
    }


class Seq2SeqDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace):
        self.items = []
        for row in rows:
            source, target = row_to_pair(row, args.dataset_format)
            model_inputs = tokenizer(source, max_length=args.max_seq_length, truncation=True)
            labels = tokenizer(text_target=target, max_length=args.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            self.items.append(model_inputs)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class TextDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace):
        texts = [row_to_text(row, args.dataset_format) for row in rows]
        self.items = [tokenizer(text, max_length=args.max_seq_length, truncation=True) for text in texts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def training_arguments(cls: Any, args: argparse.Namespace, output_dir: Path) -> Any:
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
        "save_strategy": "no" if not args.save_checkpoints else "steps",
        "eval_strategy": "steps",
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
    }
    return cls(**kwargs)


def run_seq2seq(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if torch.cuda.is_available():
        model.to("cuda")

    train_rows = read_jsonl(args.data_dir / "train.jsonl")[: args.max_train_records]
    validation_rows = read_jsonl(args.data_dir / "validation.jsonl")[: args.max_eval_records]
    output_dir = args.output_root / f"{args.model_name}-seq2seq"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    trainer = Trainer(
        model=model,
        args=training_arguments(TrainingArguments, args, output_dir),
        train_dataset=Seq2SeqDataset(train_rows, tokenizer, args),
        eval_dataset=Seq2SeqDataset(validation_rows, tokenizer, args),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    if args.save_final:
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
    metrics = {
        "target_type": args.target_type,
        "model": args.model_name,
        "base_path": str(args.model_path),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "max_steps": args.max_steps,
        "train_runtime_seconds": round(time.time() - started, 3),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": dict(eval_result),
    }
    if "eval_loss" in eval_result:
        metrics["eval_perplexity"] = math.exp(eval_result["eval_loss"]) if eval_result["eval_loss"] < 20 else None
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_text_encoder(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
    )
    if torch.cuda.is_available():
        model.to("cuda")

    train_rows = read_jsonl(args.data_dir / "train.jsonl")[: args.max_train_records]
    validation_rows = read_jsonl(args.data_dir / "validation.jsonl")[: args.max_eval_records]
    output_dir = args.output_root / f"{args.model_name}-text-encoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    trainer = Trainer(
        model=model,
        args=training_arguments(TrainingArguments, args, output_dir),
        train_dataset=TextDataset(train_rows, tokenizer, args),
        eval_dataset=TextDataset(validation_rows, tokenizer, args),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    if args.save_final:
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
    metrics = {
        "target_type": args.target_type,
        "model": args.model_name,
        "base_path": str(args.model_path),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "max_steps": args.max_steps,
        "train_runtime_seconds": round(time.time() - started, 3),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": dict(eval_result),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight text-side training targets.")
    parser.add_argument("--target-type", choices=["seq2seq_t5", "text_encoder"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="text-target")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-format", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=8)
    parser.add_argument("--max-eval-records", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--save-final", action="store_true", default=True)
    parser.add_argument("--no-save-final", action="store_false", dest="save_final")
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.model_path = args.model_path.resolve()

    if args.dry_run:
        print(json.dumps(dry_run(args), indent=2, sort_keys=True))
        return 0
    result = run_seq2seq(args) if args.target_type == "seq2seq_t5" else run_text_encoder(args)
    print(json.dumps({"status": "completed", "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
