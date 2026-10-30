# Tests for LatencyMonitor: mark/elapsed bookkeeping + per-stage summary logging.
# Implementation: Part 6
import logging
import time

import pytest

from pipeline.latency_monitor import LatencyMonitor


def _config(log_latency: bool = True) -> dict:
    return {"logging": {"level": "INFO", "log_latency": log_latency, "log_file": None}}


def test_mark_before_start_turn_raises():
    monitor = LatencyMonitor(_config())
    with pytest.raises(RuntimeError):
        monitor.mark("stt")


def test_start_turn_then_mark_records_positive_elapsed():
    monitor = LatencyMonitor(_config())
    monitor.start_turn()
    time.sleep(0.01)
    elapsed = monitor.mark("stt")
    assert elapsed > 0


def test_marks_are_monotonically_increasing_across_stages():
    monitor = LatencyMonitor(_config())
    monitor.start_turn()
    time.sleep(0.005)
    first = monitor.mark("stt")
    time.sleep(0.005)
    second = monitor.mark("llm_first_token")
    assert second > first


def test_elapsed_ms_returns_none_for_unmarked_stage():
    monitor = LatencyMonitor(_config())
    monitor.start_turn()
    monitor.mark("stt")
    assert monitor.elapsed_ms("tts_first_chunk") is None


def test_elapsed_ms_returns_recorded_value():
    monitor = LatencyMonitor(_config())
    monitor.start_turn()
    elapsed = monitor.mark("stt")
    assert monitor.elapsed_ms("stt") == elapsed


def test_start_turn_resets_marks():
    monitor = LatencyMonitor(_config())
    monitor.start_turn()
    monitor.mark("stt")
    monitor.start_turn()
    assert monitor.elapsed_ms("stt") is None


def test_log_summary_emits_per_stage_breakdown(caplog):
    monitor = LatencyMonitor(_config(log_latency=True))
    with caplog.at_level(logging.INFO, logger="pipeline.latency_monitor"):
        monitor.start_turn()
        monitor.mark("stt")
        monitor.mark("llm_first_token")
        monitor.log_summary()

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "stt:" in message
    assert "llm_first_token:" in message
    assert "total:" in message


def test_log_summary_noop_when_log_latency_disabled(caplog):
    monitor = LatencyMonitor(_config(log_latency=False))
    with caplog.at_level(logging.INFO, logger="pipeline.latency_monitor"):
        monitor.start_turn()
        monitor.mark("stt")
        monitor.log_summary()

    assert len(caplog.records) == 0


def test_log_summary_noop_when_no_marks(caplog):
    monitor = LatencyMonitor(_config(log_latency=True))
    with caplog.at_level(logging.INFO, logger="pipeline.latency_monitor"):
        monitor.start_turn()
        monitor.log_summary()

    assert len(caplog.records) == 0
