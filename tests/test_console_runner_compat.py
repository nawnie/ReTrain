"""Console plans must match the public runner without loading a model."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from gui.gradio import backend_worker as worker


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["sft", "full_sft", "lora", "qlora"])
@pytest.mark.parametrize("scheduler", ["linear", "cosine", "constant"])
def test_console_argv_is_accepted_by_public_runner(method, scheduler, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("console_test_bakeoff", ROOT / "scripts" / "run_posttrain_bakeoff.py")
    bakeoff = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, bakeoff)
    spec.loader.exec_module(bakeoff)
    config = asdict(worker.TrainingConfig(
        method=method,
        scheduler=scheduler,
        model_path=str(tmp_path / "model"),
        dataset_path=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        gradient_checkpointing=False,
        trust_remote_code=True,
        dry_run=True,
    ))
    argv = worker._training_runner_argv(config)
    parsed = {}
    original_parse = bakeoff.argparse.ArgumentParser.parse_args

    def capture_parse(parser, *args, **kwargs):
        result = original_parse(parser, *args, **kwargs)
        parsed.update(vars(result))
        return result

    monkeypatch.setattr(bakeoff.argparse.ArgumentParser, "parse_args", capture_parse)
    monkeypatch.setattr(bakeoff, "load_model_specs", lambda *args, **kwargs: [])
    monkeypatch.setattr(bakeoff, "dry_run", lambda *args, **kwargs: {"status": "validated"})
    monkeypatch.setattr(bakeoff, "import_training_stack", lambda *args, **kwargs: pytest.fail("dry run must not load the ML stack"))
    monkeypatch.setattr(sys, "argv", argv[1:])
    assert bakeoff.main() == 0
    assert parsed["method"] == ("full_sft" if method == "sft" else method)
    assert parsed["lr_scheduler_type"] == scheduler
    assert parsed["model_path"] == tmp_path / "model"
    assert parsed["gradient_checkpointing"] is False
    assert parsed["trust_remote_code"] is True
    assert parsed["dry_run"] is True


@pytest.mark.parametrize("method", ["dpo", "grpo", "reward_model", "kto", "rloo", "ppo"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_unimplemented_console_methods_are_blocked(method, dry_run, monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "dependency_status", lambda: [
        {"package": name, "available": True}
        for name in ("torch", "transformers", "peft", "bitsandbytes", "trl", "datasets", "tensorboard")
    ])
    monkeypatch.setattr(worker, "estimate_vram", lambda config: {"fit_state": "safe"})
    result = worker.validate_training_config({
        "method": method,
        "model_path": str(tmp_path),
        "dataset_path": str(tmp_path),
        "output_root": str(tmp_path / "output"),
        "confirmed": True,
        "dry_run": dry_run,
    })
    assert result["config"]["method"] == method
    assert result["status"] == "blocked"
    assert result["start_enabled"] is False
    gate = next(item for item in result["gates"] if item["gate"] == "Training runner")
    assert gate["state"] == "blocked"
    assert "not implemented" in gate["detail"]
