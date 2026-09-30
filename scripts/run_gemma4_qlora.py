#!/usr/bin/env python3
"""run_gemma4_qlora.py - ReTrain's QLoRA lane for Gemma 4 "Unified" models (the 12B), with live learning checks.

WHAT THIS RUNNER ADDS OVER THE GENERIC ONE (run_posttrain_bakeoff.py)
  1. It loads the model the way Gemma 4 needs (the multimodal class, picture/audio projections left alone) and puts
     LoRA only on the language layers.
  2. The prompt is built with the chat template's real generation prefix, so training text matches what the model sees
     when it is asked to answer (Gemma 4's template inserts an empty thinking block there).
  3. The loss is computed ONLY on the answer tokens. The model's vocabulary is 262K words, so scoring every prompt token
     wastes memory and time; asking for just the answer positions (`logits_to_keep`) avoids it.
  4. A "learning check" runs BEFORE training (the untrained baseline) and after every epoch: it asks the model to answer
     held-out examples and scores the answers, so you can see it learning, not only the loss number falling.
  5. It records how much each layer's adapter learns (gradient size and adapter change) to layers.jsonl.
  6. It registers itself with the ReTrain console (training/live_runs) so the Live pane can chart it.

DATA
  --data-dir holds train.jsonl (and optionally validation.jsonl). Each line: {"messages": [system?, user, ..., assistant],
  "category": "optional skill name"}. The LAST assistant message is what the model learns to write.
  If there is no validation.jsonl, --val-fraction of train.jsonl is held out.

WHAT IS NOT DONE
  Pictures are not used (text-only adapter first). Nothing here decides whether the adapter is good enough to ship;
  the learning-check numbers are the evidence to read.

OUTPUT (under --output-root/<model-name>-gemma4_qlora/)
  final/ (the adapter), metrics.json (ReTrain receipt), layers.jsonl, validation/<check>.jsonl, tensorboard/.

Heavy libraries are imported inside functions so `--help` and the unit tests work with only the standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

END_OF_TURN = "<turn|>\n"                            # what Gemma 4 writes to finish its answer (id 106, one of its EOS ids)
LM_SCOPE = r".*language_model\.layers\.\d+\."          # LoRA goes only on the language layers
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


# ------------------------------------------------------------------------------------------------------------------
# SECTION 1: data - reading rows and turning them into masked token ids
# ------------------------------------------------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not msgs or msgs[-1].get("role") != "assistant" or not isinstance(msgs[-1].get("content"), str):
                raise ValueError(f"{path}:{n}: 'messages' must end with an assistant text reply")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def split_rows(data_dir: Path, val_fraction: float, seed: int, max_train: int) -> tuple[list, list]:
    """train/validation rows: use validation.jsonl when present, otherwise hold out a slice of train.jsonl"""
    train = read_rows(data_dir / "train.jsonl")
    val_path = data_dir / "validation.jsonl"
    random.Random(seed).shuffle(train)
    if val_path.exists():
        val = read_rows(val_path)
    else:
        n_val = int(len(train) * val_fraction) if len(train) >= 20 else 0
        val, train = train[:n_val], train[n_val:]
    return (train[:max_train] if max_train else train), val


def end_marker(tokenizer) -> str:
    """what the chat template writes right after an assistant answer (for Gemma 4: '<turn|>' and a newline).
    Read from the template itself so a different Gemma build cannot silently disagree with a constant here."""
    probe = "ANSWER_PROBE"
    text = tokenizer.apply_chat_template([{"role": "user", "content": "Q"}, {"role": "assistant", "content": probe}], tokenize=False)
    tail = text.split(probe, 1)[1] if probe in text else ""
    return tail or (getattr(tokenizer, "eos_token", "") or "")


def build_example(tokenizer, messages: list[dict[str, str]], max_len: int, end: str) -> dict[str, list[int]] | None:
    """prompt tokens are hidden from the loss (-100); only the answer (plus the end marker) is learned. None if it does not fit."""
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]     # the template already starts with <bos>
    answer_ids = tokenizer(messages[-1]["content"] + end, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def tokenise(rows: list[dict], tokenizer, max_len: int) -> tuple[list[dict], int]:
    out, dropped, end = [], 0, end_marker(tokenizer)
    for row in rows:
        ex = build_example(tokenizer, row["messages"], max_len, end)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


def collate(batch: list[dict], pad_id: int) -> dict[str, Any]:
    import torch
    longest = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        pad = longest - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        att.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


def lora_target_regex(names: list[str]) -> str:
    return LM_SCOPE + r"(?:[A-Za-z_]+\.)*(?:" + "|".join(re.escape(n) for n in names) + r")$"


# ------------------------------------------------------------------------------------------------------------------
# SECTION 2: scoring a reply against the expected reply (pure Python, unit-tested)
# ------------------------------------------------------------------------------------------------------------------
def parse_json_object(text: str) -> Any:
    """the JSON object in a model reply (prose around it is tolerated), or None"""
    text = text.strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if candidate:
            try:
                return json.loads(candidate)
            except ValueError:
                continue
    return None


def valid_actions(actions: Any) -> bool:
    """each action must be {"buttons": [names], "frames": whole number}"""
    return isinstance(actions, list) and bool(actions) and all(
        isinstance(a, dict) and isinstance(a.get("buttons"), list) and all(isinstance(b, str) for b in a["buttons"])
        and isinstance(a.get("frames"), int) and not isinstance(a.get("frames"), bool) for a in actions)


def score_reply(reply_text: str, expected_text: str) -> dict[str, bool]:
    """valid_json: parses with the expected keys (and valid actions); actions_match: same actions; first_button_match: same first press;
    exact: the whole reply is identical (a memorisation gauge - the 'think' wording rarely matches)"""
    expected, got = parse_json_object(expected_text), parse_json_object(reply_text)
    valid = isinstance(got, dict) and isinstance(expected, dict) and set(expected) <= set(got)
    if valid and "actions" in expected:
        valid = valid_actions(got.get("actions"))
    actions = bool(valid and got.get("actions") == expected.get("actions"))
    first = False
    if valid and isinstance(expected.get("actions"), list) and expected["actions"] and got.get("actions"):
        first = got["actions"][0].get("buttons") == expected["actions"][0].get("buttons")
    return {"valid_json": bool(valid), "actions_match": actions, "first_button_match": first, "exact": got is not None and got == expected}


def summarise(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """percentages (0-100) over a list of {"category", "score"}; also the right-action rate per category"""
    n = max(1, len(scored))
    out: dict[str, Any] = {k: round(100.0 * sum(1 for s in scored if s["score"][k]) / n, 1)
                           for k in ("valid_json", "actions_match", "first_button_match", "exact")}
    by: dict[str, list[bool]] = {}
    for s in scored:
        by.setdefault(s.get("category") or "all", []).append(s["score"]["actions_match"])
    out["by_category"] = {c: {"n": len(v), "actions_match": round(100.0 * sum(v) / len(v), 1)} for c, v in sorted(by.items())}
    out["n"] = len(scored)
    return out


def pick_check_rows(val_rows: list[dict], n: int) -> list[dict]:
    """a fixed, category-balanced sample so every check scores the SAME questions"""
    if n <= 0 or not val_rows:
        return []
    groups: dict[str, list[dict]] = {}
    for r in val_rows:
        groups.setdefault(r.get("category") or "all", []).append(r)
    picked, i = [], 0
    while len(picked) < min(n, len(val_rows)):
        for g in groups.values():
            if i < len(g) and len(picked) < n:
                picked.append(g[i])
        i += 1
    return picked


# ------------------------------------------------------------------------------------------------------------------
# SECTION 3: training pieces that need torch (built lazily)
# ------------------------------------------------------------------------------------------------------------------
def build_trainer_classes():
    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainerCallback

    class AnswerOnlyTrainer(Trainer):
        """Trainer whose loss looks only at answer positions, so the 262K-word output layer is not run on the prompt."""

        def __init__(self, *a, answer_only: bool = True, **kw):
            super().__init__(*a, **kw)
            self.answer_only = answer_only
            self.model_accepts_loss_kwargs = False      # we return a per-micro-batch mean; the Trainer divides by accumulation steps

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            if not self.answer_only:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            return answer_only_loss(model, inputs["input_ids"], inputs["attention_mask"], labels, return_outputs)

    def answer_only_loss(model, input_ids, attention_mask, labels, return_outputs=False):
        # positions whose output predicts a labelled token (union over the batch; right padding keeps answers at the end)
        need = (labels[:, 1:] != -100).any(dim=0).nonzero().squeeze(-1)
        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=need, use_cache=False)
        logits = out.logits.float()
        targets = labels[:, 1:][:, need]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return (loss, out) if return_outputs else loss

    class LayerStats(TrainerCallback):
        """records how much each layer's adapter is learning: gradient size (before the update) and adapter size (after)"""

        def __init__(self, path: Path, every: int):
            self.path, self.every, self.grad_sq = path, max(1, every), {}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        @staticmethod
        def _slot(name: str):
            m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.", name)
            return (int(m.group(1)), "attn" if m.group(2) == "self_attn" else "mlp") if m else None

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if (state.global_step + 1) % self.every:
                return
            self.grad_sq = {}
            for name, p in model.named_parameters():
                slot = self._slot(name)
                if p.grad is not None and slot:
                    self.grad_sq[slot] = self.grad_sq.get(slot, 0.0) + float(p.grad.float().pow(2).sum())

        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % self.every or not self.grad_sq:
                return
            upd: dict[int, float] = {}
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                        slot = self._slot(name + ".")
                        if not slot:
                            continue
                        A, B = mod.lora_A["default"].weight.float(), mod.lora_B["default"].weight.float()
                        # ||B A||_F^2 = sum((B^T B) * (A A^T)), computed on tiny r x r matrices instead of the full product
                        f2 = float(((B.T @ B) * (A @ A.T)).sum()) * float(mod.scaling["default"]) ** 2
                        upd[slot[0]] = upd.get(slot[0], 0.0) + math.sqrt(max(f2, 0.0))
            layers = sorted({i for i, _ in self.grad_sq})
            row = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4),
                   "layers": [{"i": i, "attn": math.sqrt(self.grad_sq.get((i, "attn"), 0.0)), "mlp": math.sqrt(self.grad_sq.get((i, "mlp"), 0.0)),
                               "upd": upd.get(i, 0.0)} for i in layers]}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.grad_sq = {}

    class LearningCheck(TrainerCallback):
        """before training and after each epoch: answer held-out examples and score them"""

        def __init__(self, tokenizer, rows, val_tok, out_dir: Path, max_new: int, batch: int, history: list):
            self.tok, self.rows, self.val_tok, self.out_dir = tokenizer, rows, val_tok, out_dir
            self.max_new, self.batch, self.history = max_new, batch, history

        @torch.no_grad()
        def _eval_loss(self, model) -> float | None:
            if not self.val_tok:
                return None
            total, count = 0.0, 0
            for i in range(0, len(self.val_tok), 2):
                b = collate(self.val_tok[i:i + 2], self.tok.pad_token_id)
                b = {k: v.to(model.device) for k, v in b.items()}
                loss = answer_only_loss(model, b["input_ids"], b["attention_mask"], b["labels"])
                total, count = total + float(loss), count + 1
            return total / count if count else None

        @torch.no_grad()
        def run(self, model, tag: str, epoch: float, step: int = 0) -> None:
            t0 = time.time()
            was_training = model.training
            model.eval()
            self.tok.padding_side = "left"
            replies = []
            for i in range(0, len(self.rows), self.batch):
                chunk = self.rows[i:i + self.batch]
                prompts = [self.tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in chunk]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                gen = model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, use_cache=True, pad_token_id=self.tok.pad_token_id)
                replies += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            scores = [score_reply(rep, r["messages"][-1]["content"]) for rep, r in zip(replies, self.rows)]
            result = summarise([{"category": r.get("category", ""), "score": sc} for r, sc in zip(self.rows, scores)])
            loss = self._eval_loss(model)
            result.update({"tag": tag, "epoch": epoch, "step": step, "eval_loss": loss,
                           "eval_perplexity": math.exp(loss) if loss is not None and loss < 20 else None, "seconds": round(time.time() - t0, 1)})
            self.history.append(result)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with (self.out_dir / f"{re.sub(r'[^A-Za-z0-9]+', '-', tag)}.jsonl").open("w", encoding="utf-8") as f:
                for rep, r, s in zip(replies, self.rows, scores):
                    f.write(json.dumps({"category": r.get("category"), "prompt": r["messages"][-2]["content"] if len(r["messages"]) > 1 else "",
                                    "expected": r["messages"][-1]["content"], "got": rep, **s}, ensure_ascii=False) + "\n")
            print("VALIDATION " + json.dumps(result), flush=True)
            self.tok.padding_side = "right"
            if was_training:
                model.train()
            torch.cuda.empty_cache()

        def on_train_begin(self, args, state, control, model=None, **kw):
            if self.rows:
                self.run(model, "baseline", 0.0, 0)

        def on_epoch_end(self, args, state, control, model=None, **kw):
            if self.rows:
                self.run(model, f"epoch {round(state.epoch or 0)}", float(state.epoch or 0), int(state.global_step))

    return AnswerOnlyTrainer, LayerStats, LearningCheck


# ------------------------------------------------------------------------------------------------------------------
# SECTION 4: the run
# ------------------------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune of Gemma 4 Unified with live learning checks")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True, help="folder with the full-precision Gemma 4 weights (safetensors)")
    ap.add_argument("--model-name", default="gemma-4-12b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (0 = use --epochs)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default=DEFAULT_TARGETS)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--answer-only-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gpu-fraction", type=float, default=0.72, help="cap on PyTorch's share of GPU memory (Windows keeps ~4 GB for itself)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--eval-samples", type=int, default=24, help="held-out examples answered in each learning check (0 = off)")
    ap.add_argument("--eval-max-new-tokens", type=int, default=160)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-train-records", type=int, default=0)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--title", default="")
    ap.add_argument("--log-file", default="", help="write all output here (also lets the console follow the run)")
    ap.add_argument("--dry-run", action="store_true", help="check data + LoRA layout without loading the model onto the GPU")
    return ap.parse_args(argv)


def dry_run(a: argparse.Namespace) -> int:
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    train, val = split_rows(a.data_dir, a.val_fraction, a.seed, a.max_train_records)
    t_tok, dropped = tokenise(train, tok, a.max_seq_length)
    lens = sorted(len(x["input_ids"]) for x in t_tok)
    ans = sum(1 for x in t_tok for l in x["labels"] if l != -100)
    print(json.dumps({"train_rows": len(train), "validation_rows": len(val), "too_long": dropped,
                      "tokens_per_example_median": lens[len(lens) // 2] if lens else 0, "answer_tokens_total": ans,
                      "answer_share_of_tokens": round(ans / max(1, sum(lens)), 3)}))
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(AutoConfig.from_pretrained(a.model_path))
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=lora_target_regex([x.strip() for x in a.lora_target_modules.split(",") if x.strip()]))
    peft_model = get_peft_model(model, cfg)
    hits = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
    outside = [n for n in hits if "language_model" not in n]
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"LoRA layers: {len(hits)} (outside the language model: {len(outside)}) | trainable params: {trainable / 1e6:.1f} M")
    import inspect
    print("model accepts logits_to_keep (answer-only loss):", "logits_to_keep" in inspect.signature(model.forward).parameters)
    return 0 if hits and not outside and train else 1


def train(a: argparse.Namespace) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from backend import run_telemetry

    if not torch.cuda.is_available():
        print("no CUDA GPU visible"); return 2
    free = torch.cuda.mem_get_info()[0] / 2**20
    if free < 11000:
        print(f"only {free:.0f} MiB of GPU memory is free; this run needs ~11000. Close ComfyUI / llama-server first."); return 2
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)

    run_dir = a.output_root / f"{a.model_name}-gemma4_qlora"
    run_dir.mkdir(parents=True, exist_ok=True)
    layers_path = run_dir / "layers.jsonl"
    started = time.time()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    train_rows, val_rows = split_rows(a.data_dir, a.val_fraction, a.seed, a.max_train_records)
    train_tok, dropped = tokenise(train_rows, tok, a.max_seq_length)
    val_tok, _ = tokenise(val_rows, tok, a.max_seq_length)
    print(f"dataset: {{'examples': {len(train_rows) + len(val_rows)}, 'too_long': {dropped}, 'train': {len(train_tok)}, 'eval': {len(val_tok)}}}", flush=True)
    if not train_tok:
        print("nothing to train on"); return 2
    total_epochs = a.epochs if not a.max_steps else a.max_steps * a.batch_size * a.gradient_accumulation_steps / len(train_tok)
    run_telemetry.register_run(f"{a.model_name}-{int(started)}", title=a.title or f"{a.model_name} QLoRA", log=a.log_file, epochs=round(total_epochs, 3),
                               started=started, pid=os.getpid(), layers=str(layers_path))

    # 4-bit NormalFloat + double quantisation (the QLoRA recipe). The picture/audio projections and output head stay 16-bit.
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_storage=torch.bfloat16, llm_int8_skip_modules=["embed_vision", "embed_audio", "lm_head"])
    model = AutoModelForMultimodalLM.from_pretrained(str(a.model_path), dtype=torch.bfloat16, device_map={"": 0}, quantization_config=bnb)
    model.config.use_cache = False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=lora_target_regex([x.strip() for x in a.lora_target_modules.split(",") if x.strip()]))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    Trainer, LayerStats, LearningCheck = build_trainer_classes()
    history: list[dict] = []
    check = LearningCheck(tok, pick_check_rows(val_rows, a.eval_samples), val_tok, run_dir / "validation", a.eval_max_new_tokens, a.eval_batch_size, history)
    args = TrainingArguments(
        output_dir=str(run_dir), num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps else -1,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.gradient_accumulation_steps, learning_rate=a.learning_rate,
        lr_scheduler_type="cosine", warmup_steps=0.03, optim="paged_adamw_8bit", bf16=True, logging_steps=a.logging_steps, save_strategy="no",
        eval_strategy="no", report_to=["tensorboard"],   # TensorBoard files go under <output_dir>/runs (transformers 5 removed logging_dir)
         remove_unused_columns=False, seed=a.seed,
        dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=lambda b: collate(b, tok.pad_token_id),
                      answer_only=a.answer_only_loss, callbacks=[check, LayerStats(layers_path, a.logging_steps)])
    result = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 2**30
    if check.rows and (not history or history[-1]["tag"] != f"epoch {round(total_epochs)}"):
        check.run(model, "final", float(trainer.state.epoch or 0), int(trainer.state.global_step))
    print(f"peak VRAM allocated: {peak:.2f} GiB", flush=True)
    final_dir = run_dir / "final"
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    last = history[-1] if history else {}
    metrics = {"model": a.model_name, "method": "gemma4_qlora", "base_path": str(a.model_path), "train_rows": len(train_tok), "validation_rows": len(val_tok),
               "max_steps": trainer.state.global_step, "max_seq_length": a.max_seq_length, "train_runtime_seconds": round(time.time() - started, 1),
               "train_metrics": dict(result.metrics), "eval_metrics": {"eval_loss": last.get("eval_loss")} if last.get("eval_loss") is not None else {},
               "learning_checks": history, "answer_only_loss": a.answer_only_loss, "gradient_checkpointing": a.gradient_checkpointing,
               "cuda": {"device": torch.cuda.get_device_name(0), "peak_allocated_gb": round(peak, 3)}}
    if last.get("eval_loss") is not None and last["eval_loss"] < 20:
        metrics["eval_perplexity"] = math.exp(last["eval_loss"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print("adapter saved to", final_dir, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.7")
    if a.log_file:
        os.environ.setdefault("TQDM_DISABLE", "1")
        sys.stdout = sys.stderr = open(a.log_file, "w", buffering=1, encoding="utf-8")
    try:
        return dry_run(a) if a.dry_run else train(a)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
