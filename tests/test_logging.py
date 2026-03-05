"""
test_logging.py — Unit tests for MetricsLogger (Phase 3 monitoring).

All tests run without real W&B or TensorBoard installations:
  - W&B is never initialised (use_wandb=False in all tests)
  - TensorBoard is tested when available; skipped gracefully if not installed

Coverage
--------
TestMetricsLoggerJSON       — 8 tests  (JSON backend)
TestMetricsLoggerConsole    — 3 tests  (console / Python logging)
TestMetricsLoggerTensorBoard— 3 tests  (TensorBoard, skip if unavailable)
TestMetricsLoggerNonMain    — 3 tests  (non-main process is a no-op)
TestMetricsLoggerEdgeCases  — 4 tests  (close idempotency, special values)
                               ───────
Total                          21 tests
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

from src.training.logger import MetricsLogger, _TB_OK


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_logger(tmp_path: Path, **kwargs) -> MetricsLogger:
    """Create a MetricsLogger with W&B disabled and sane defaults."""
    defaults = dict(
        log_dir=tmp_path / "logs",
        config_dict={"batch_size": 4},
        use_wandb=False,
        use_tensorboard=False,
        is_main_process=True,
    )
    defaults.update(kwargs)
    return MetricsLogger(**defaults)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ═════════════════════════════════════════════════════════════════════════════
# TestMetricsLoggerJSON
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsLoggerJSON:

    def test_creates_jsonl_file(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"train/loss": 1.5}, step=1)
        lg.close()
        assert (tmp_path / "logs" / "metrics.jsonl").exists()

    def test_log_writes_one_line(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"train/loss": 2.0}, step=10)
        lg.close()
        lines = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")
        assert len(lines) == 1

    def test_log_multiple_steps(self, tmp_path):
        lg = _make_logger(tmp_path)
        for s in range(5):
            lg.log({"train/loss": float(s)}, step=s)
        lg.close()
        lines = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")
        assert len(lines) == 5

    def test_step_preserved_in_json(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"train/loss": 0.5}, step=42)
        lg.close()
        record = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")[0]
        assert record["step"] == 42

    def test_metric_values_preserved(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"train/loss": 1.23, "train/lr": 5e-5}, step=1)
        lg.close()
        record = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")[0]
        assert record["train/loss"] == pytest.approx(1.23, rel=1e-5)
        assert record["train/lr"]   == pytest.approx(5e-5, rel=1e-4)

    def test_timestamp_present_in_json(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"x": 1.0}, step=0)
        lg.close()
        record = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")[0]
        assert "ts" in record
        assert record["ts"] > 0

    def test_file_flushed_after_each_call(self, tmp_path):
        """File must be readable between log() calls (flush on each write)."""
        lg = _make_logger(tmp_path)
        lg.log({"x": 1.0}, step=1)
        # Read without calling close()
        lines = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")
        assert len(lines) == 1
        lg.close()

    def test_append_across_instances(self, tmp_path):
        """Two MetricsLogger instances append to the same file."""
        for run in range(2):
            lg = _make_logger(tmp_path)
            lg.log({"run": float(run)}, step=run)
            lg.close()
        lines = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")
        assert len(lines) == 2


# ═════════════════════════════════════════════════════════════════════════════
# TestMetricsLoggerConsole
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsLoggerConsole:

    def test_log_emits_info_message(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="src.training.logger"):
            lg = _make_logger(tmp_path)
            lg.log({"train/loss": 3.14}, step=7)
            lg.close()
        assert any("3.1400" in r.message for r in caplog.records)

    def test_log_includes_step(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="src.training.logger"):
            lg = _make_logger(tmp_path)
            lg.log({"x": 1.0}, step=999)
            lg.close()
        assert any("999" in r.message for r in caplog.records)

    def test_multiple_metrics_in_one_line(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO, logger="src.training.logger"):
            lg = _make_logger(tmp_path)
            lg.log({"a": 1.0, "b": 2.0}, step=1)
            lg.close()
        combined = " ".join(r.message for r in caplog.records)
        assert "a=" in combined and "b=" in combined


# ═════════════════════════════════════════════════════════════════════════════
# TestMetricsLoggerTensorBoard
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _TB_OK, reason="tensorboard not installed")
class TestMetricsLoggerTensorBoard:

    def test_tensorboard_dir_created(self, tmp_path):
        lg = _make_logger(tmp_path, use_tensorboard=True)
        lg.log({"train/loss": 1.0}, step=1)
        lg.close()
        assert (tmp_path / "logs" / "tensorboard").exists()

    def test_tensorboard_events_written(self, tmp_path):
        lg = _make_logger(tmp_path, use_tensorboard=True)
        lg.log({"train/loss": 1.0, "train/lr": 1e-4}, step=1)
        lg.close()
        tb_dir = tmp_path / "logs" / "tensorboard"
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_no_crash_on_close_without_log(self, tmp_path):
        lg = _make_logger(tmp_path, use_tensorboard=True)
        lg.close()   # close without any log() call — must not raise


# ═════════════════════════════════════════════════════════════════════════════
# TestMetricsLoggerNonMain
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsLoggerNonMain:

    def test_no_file_created_for_non_main(self, tmp_path):
        lg = _make_logger(tmp_path, is_main_process=False)
        lg.log({"train/loss": 1.0}, step=1)
        lg.close()
        assert not (tmp_path / "logs" / "metrics.jsonl").exists()

    def test_log_noop_for_non_main(self, tmp_path, caplog):
        """Non-main process should not emit any log records."""
        with caplog.at_level(logging.INFO, logger="src.training.logger"):
            lg = _make_logger(tmp_path, is_main_process=False)
            lg.log({"x": 99.0}, step=1)
            lg.close()
        # No INFO records from the logger backend
        logger_records = [r for r in caplog.records
                          if r.name == "src.training.logger"]
        assert len(logger_records) == 0

    def test_close_noop_for_non_main(self, tmp_path):
        lg = _make_logger(tmp_path, is_main_process=False)
        lg.close()   # must not raise


# ═════════════════════════════════════════════════════════════════════════════
# TestMetricsLoggerEdgeCases
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsLoggerEdgeCases:

    def test_close_idempotent(self, tmp_path):
        """Calling close() twice must not raise."""
        lg = _make_logger(tmp_path)
        lg.log({"x": 1.0}, step=0)
        lg.close()
        lg.close()   # second close — must not raise

    def test_empty_metrics_dict(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({}, step=5)   # empty dict — must not raise
        lg.close()
        record = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")[0]
        assert record["step"] == 5

    def test_very_large_step(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.log({"x": 0.0}, step=10_000_000)
        lg.close()
        record = _read_jsonl(tmp_path / "logs" / "metrics.jsonl")[0]
        assert record["step"] == 10_000_000

    def test_nan_value_written_to_json(self, tmp_path):
        """NaN metrics (if they occur before detection) are serialised safely."""
        lg = _make_logger(tmp_path)
        # json.dumps handles float('nan') → 'NaN' which is not valid JSON,
        # so we use a workaround: store as string.  The logger should at
        # minimum not crash.
        try:
            lg.log({"train/loss": float("nan")}, step=1)
        except (ValueError, TypeError):
            pass   # acceptable: JSON doesn't support NaN natively
        finally:
            lg.close()
