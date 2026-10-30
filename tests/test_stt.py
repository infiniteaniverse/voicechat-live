"""
Tests for WhisperSTT.
These load the real model — run after `python scripts/download_models.py --part 1`.

What we're testing:
  - Interface: transcribe() always returns a str
  - Silence handling: empty/silent audio produces empty or near-empty output
  - Type handling: input doesn't have to be float32 — WhisperSTT converts internally
  - Latency: median over 5 runs is under 400ms (the STT budget slice)
"""
import time

import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def stt(config):
    from audio.stt import WhisperSTT
    return WhisperSTT(config)


@pytest.fixture
def sample_rate(config) -> int:
    return config["audio"]["sample_rate"]


# ── Interface tests ─────────────────────────────────────────────────────────────

def test_transcribe_returns_string(stt, sample_rate):
    audio = np.zeros(3 * sample_rate, dtype=np.float32)
    assert isinstance(stt.transcribe(audio), str)


def test_transcribe_silent_audio_produces_minimal_output(stt, sample_rate):
    """Whisper on pure silence should return empty or a very short filler."""
    audio = np.zeros(3 * sample_rate, dtype=np.float32)
    result = stt.transcribe(audio)
    assert len(result) < 30, f"Silent audio returned unexpectedly long output: '{result}'"


def test_transcribe_accepts_non_float32_and_converts(stt, sample_rate):
    """WhisperSTT must handle float64 input (converts internally, must not crash)."""
    audio = np.zeros(sample_rate, dtype=np.float64)
    result = stt.transcribe(audio)
    assert isinstance(result, str)


def test_transcribe_noise_does_not_crash(stt, sample_rate):
    """Random noise input must not raise any exceptions."""
    audio = (np.random.randn(2 * sample_rate) * 0.1).astype(np.float32)
    result = stt.transcribe(audio)
    assert isinstance(result, str)


def test_transcribe_short_clip_does_not_crash(stt, sample_rate):
    """Even a 0.5-second clip (very short) must not raise exceptions."""
    audio = np.zeros(sample_rate // 2, dtype=np.float32)
    result = stt.transcribe(audio)
    assert isinstance(result, str)


# ── Latency test ────────────────────────────────────────────────────────────────

def test_stt_median_latency_within_budget(stt, sample_rate):
    """
    Median STT latency over 5 runs must be under 400ms.
    With faster-whisper small on RTX 4060 CUDA, expect ~80ms median.
    This test catches if someone accidentally changes device back to CPU.
    """
    audio = (np.random.randn(5 * sample_rate) * 0.05).astype(np.float32)

    # One warm-up before measuring
    stt.transcribe(audio)

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        stt.transcribe(audio)
        latencies.append((time.perf_counter() - t0) * 1000)

    median = sorted(latencies)[len(latencies) // 2]
    assert median < 400, (
        f"STT median latency {median:.0f}ms exceeds 400ms budget. "
        f"Check config.yaml: device should be 'cuda', compute_type 'int8'."
    )
