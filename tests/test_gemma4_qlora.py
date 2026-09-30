"""Tests for the pure (no GPU, no model download) parts of scripts/run_gemma4_qlora.py."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gemma4_qlora.py"
spec = importlib.util.spec_from_file_location("run_gemma4_qlora", SCRIPT)
g4 = importlib.util.module_from_spec(spec)
sys.modules["run_gemma4_qlora"] = g4
spec.loader.exec_module(g4)


class FakeTokenizer:
    """one token per character; the chat template is a readable string that ends the answer with '<end>' like Gemma's '<turn|>'"""

    chat_template = "fake"
    eos_token = "<end>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<bos>" + "".join(f"[{m['role']}]{m['content']}<end>" for m in messages)
        return text + ("[model]<thought>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_end_marker_is_found_from_the_template():
    assert g4.end_marker(FakeTokenizer()) == "<end>"


def test_prompt_hidden_answer_learned_including_end_marker():
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    ex = g4.build_example(tok, msgs, 1000, g4.end_marker(tok))
    learned = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100)
    hidden = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100)
    assert learned == "ok<end>"
    assert hidden.endswith("[model]<thought>")          # the same prompt shape the model sees when asked to answer
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_example_that_does_not_fit_is_dropped_not_cut():
    tok = FakeTokenizer()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert g4.build_example(tok, msgs, 5, "<end>") is None


def test_lora_regex_only_matches_language_model_layers():
    rx = re.compile(g4.lora_target_regex(["q_proj", "down_proj"]))
    assert rx.fullmatch("model.language_model.layers.7.self_attn.q_proj")
    assert rx.fullmatch("model.language_model.layers.47.mlp.down_proj")
    assert not rx.fullmatch("model.embed_vision.patch_dense")
    assert not rx.fullmatch("model.language_model.layers.7.self_attn.k_proj")
    assert not rx.fullmatch("lm_head")


def test_read_rows_rejects_chat_not_ending_in_assistant(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        g4.read_rows(path)


def test_split_uses_validation_file_when_present(tmp_path):
    row = lambda i: json.dumps({"category": "a", "messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "y"}]})
    (tmp_path / "train.jsonl").write_text("\n".join(row(i) for i in range(30)) + "\n", encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text("\n".join(row(i) for i in range(100, 103)) + "\n", encoding="utf-8")
    train, val = g4.split_rows(tmp_path, 0.1, 1, 0)
    assert (len(train), len(val)) == (30, 3)


def test_split_holds_out_a_fraction_deterministically(tmp_path):
    row = lambda i: json.dumps({"category": "a", "messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "y"}]})
    (tmp_path / "train.jsonl").write_text("\n".join(row(i) for i in range(50)) + "\n", encoding="utf-8")
    a, b = g4.split_rows(tmp_path, 0.1, 7, 0), g4.split_rows(tmp_path, 0.1, 7, 0)
    assert a == b and (len(a[0]), len(a[1])) == (45, 5)


def test_check_rows_spread_across_categories():
    rows = [{"category": c, "messages": []} for c in ["a"] * 10 + ["b"] * 2 + ["c"] * 2]
    picked = g4.pick_check_rows(rows, 6)
    assert {r["category"] for r in picked} == {"a", "b", "c"} and len(picked) == 6


GOOD = json.dumps({"think": "t", "say": "", "actions": [{"buttons": ["A"], "frames": 8}], "remember": [], "look": False})


def test_score_exact_reply():
    s = g4.score_reply(GOOD, GOOD)
    assert s == {"valid_json": True, "actions_match": True, "first_button_match": True, "exact": True}


def test_score_right_action_wrong_wording():
    other = json.dumps({"think": "different", "say": "", "actions": [{"buttons": ["A"], "frames": 8}], "remember": [], "look": False})
    s = g4.score_reply(other, GOOD)
    assert s["actions_match"] and not s["exact"]


def test_score_wrong_action_and_broken_json():
    wrong = json.dumps({"think": "t", "actions": [{"buttons": ["B"], "frames": 8}]})
    assert not g4.score_reply(wrong, GOOD)["actions_match"]
    broken = g4.score_reply("sure! pressing A now", GOOD)
    assert not broken["valid_json"] and not broken["actions_match"]


def test_json_found_inside_extra_prose():
    assert g4.parse_json_object('Here you go: {"actions": []} thanks') == {"actions": []}


def test_summary_percentages_and_categories():
    scored = [{"category": "x", "score": {"valid_json": True, "actions_match": True, "first_button_match": True, "exact": True}},
              {"category": "y", "score": {"valid_json": True, "actions_match": False, "first_button_match": False, "exact": False}}]
    out = g4.summarise(scored)
    assert out["n"] == 2 and out["actions_match"] == 50.0 and out["valid_json"] == 100.0
    assert out["by_category"]["x"]["actions_match"] == 100.0


def test_answer_only_positions_cover_exactly_the_answer():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, -100, -100, 5, 6, 7, -100]])
    keep = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]
    # positions 2,3,4 predict tokens 3,4,5 (the answer)
    assert keep.tolist() == [2, 3, 4]
    assert labels[:, 1:][:, keep].tolist() == [[5, 6, 7]]
