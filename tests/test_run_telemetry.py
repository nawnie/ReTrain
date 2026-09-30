"""Tests for the live-run telemetry (backend/run_telemetry.py): log lines -> snapshot, run status, per-layer readings."""

import json
import tempfile
import unittest
from pathlib import Path

from backend import run_telemetry as tele

class TelemetryTests(unittest.TestCase):
    def test_log_lines_become_a_snapshot(self):
        st = tele.RunState(total_epochs=2.0, start_time=1000.0)
        st.feed("dataset: {'examples': 556, 'train': 501, 'eval': 55}", now=1010)
        st.feed("{'loss': '2.0', 'grad_norm': '3.0', 'learning_rate': '0.0002', 'epoch': '0.5'}", now=1100)
        st.feed("{'loss': '0.5', 'grad_norm': '1.0', 'learning_rate': '0.0001', 'epoch': '1.0'}", now=1200)
        st.feed('VALIDATION {"tag": "epoch 1", "valid_json": 100.0, "actions_match": 80.0, "exact": 10.0, "eval_loss": 0.3}', now=1210)
        snap = st.snapshot(now=1220, alive=True)
        self.assertEqual(snap["status"], "training")
        self.assertEqual(snap["lossNow"], 0.5)
        self.assertAlmostEqual(snap["progress"], 0.5)
        self.assertEqual(snap["checks"][0]["actions_match"], 80.0)
        self.assertGreater(snap["examplesPerSecond"], 0)

    def test_finished_and_failed(self):
        done = tele.RunState(2.0)
        done.feed("adapter saved to X:\a")
        self.assertEqual(done.snapshot()["status"], "done")
        failed = tele.RunState(2.0)
        failed.feed("Traceback (most recent call last):")
        self.assertEqual(failed.snapshot()["status"], "failed")
        gone = tele.RunState(2.0)
        self.assertEqual(gone.snapshot(alive=False)["status"], "stopped")

    def test_layer_stats_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "layers.jsonl"
            p.write_text("\n".join(json.dumps({"step": s, "epoch": s / 10, "layers": [{"i": 0, "attn": 1.0, "mlp": 2.0}, {"i": 1, "attn": 3.0, "mlp": 0.0}]}) for s in (5, 10)), encoding="utf-8")
            out = tele.layer_stats({"layers": str(p)})
            self.assertTrue(out["available"])
            self.assertEqual(out["totals"][0], {"i": 0, "attn": 2.0, "mlp": 4.0})
            self.assertFalse(tele.layer_stats({"layers": str(Path(d) / "missing.jsonl")})["available"])


if __name__ == "__main__":
    unittest.main()
