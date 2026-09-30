"""The runners must build TrainingArguments with whatever transformers is installed (5.x removed logging_dir)."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses in the script look themselves up here
    spec.loader.exec_module(module)
    return module


def test_bakeoff_builds_training_arguments(tmp_path):
    transformers = pytest.importorskip("transformers")
    bake = load("run_posttrain_bakeoff")
    args = argparse.Namespace(batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-4, max_steps=1, gradient_checkpointing=True,
                              optim="adamw_torch", lr_scheduler_type="linear", save_checkpoints=False, precision="bf16")
    made = bake.training_arguments(transformers.TrainingArguments, output_dir=tmp_path, args=args)
    assert made.max_steps == 1


def test_text_target_builds_training_arguments(tmp_path):
    transformers = pytest.importorskip("transformers")
    text = load("run_text_target_training")
    args = argparse.Namespace(batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-4, max_steps=1, save_checkpoints=False, precision="bf16")
    made = text.training_arguments(transformers.TrainingArguments, args, tmp_path)
    assert made.max_steps == 1
