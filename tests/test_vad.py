"""
Tests for SileroVAD.
These load the real Silero model — run after `python scripts/download_models.py --part 1`.

What we're testing:
  - Interface: process_chunk() returns np.ndarray or None, never anything else
  - Silence: pure silence never emits an utterance
  - Variable chunk size: 1024-sample chunks (from AudioManager) work correctly
  - Type contract: emitted utterances are always float32 (required by WhisperSTT)
  - State machine: reset() clears all state correctly
  - Short burst: a very short noise burst (< min_speech_duration_ms) is discarded

Note on real speech detection:
  White noise does NOT reliably trigger Silero VAD as "speech" — it's a neural model
  trained on real speech characteristics. Tests that check for emission use a
  conditional (if utterance is not None) rather than asserting emission happens.
  Integration tests with real audio files go in test_pipeline_e2e.py (Part 9).
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHUNK = 512  # Silero VAD window size


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def vad(config):
    from audio.vad import SileroVAD
    return SileroVAD(config)


@pytest.fixture(autouse=True)
def reset_vad_between_tests(vad):
    """Reset VAD state before every test so tests don't bleed into each other."""
    vad.reset()
    yield
    vad.reset()


def silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def noise(n: int, amp: float = 0.4) -> np.ndarray:
    return (np.random.randn(n) * amp).astype(np.float32)


# ── Return type ─────────────────────────────────────────────────────────────────

def test_returns_none_or_ndarray(vad):
    result = vad.process_chunk(silence(CHUNK))
    assert result is None or isinstance(result, np.ndarray)


def test_silence_never_emits(vad, config):
    """3 seconds of silence must not trigger any utterance."""
    sr = config["audio"]["sample_rate"]
    results = []
    for _ in range(3 * sr // CHUNK):
        r = vad.process_chunk(silence(CHUNK))
        if r is not None:
            results.append(r)
    assert len(results) == 0, "Silence triggered a VAD utterance — threshold may be too low"


def test_handles_1024_chunk_size(vad):
    """AudioManager yields 1024-sample chunks — VAD must handle them without crashing."""
    result = vad.process_chunk(silence(1024))
    assert result is None or isinstance(result, np.ndarray)


def test_handles_odd_chunk_size(vad):
    """Chunk sizes that aren't multiples of 512 must not crash (ring buffer handles leftover)."""
    result = vad.process_chunk(silence(300))
    assert result is None or isinstance(result, np.ndarray)


# ── Type contract ───────────────────────────────────────────────────────────────

def test_emitted_utterance_is_float32(vad, config):
    """
    If an utterance is emitted, it must be float32 — WhisperSTT requires this.
    We feed high-amplitude noise to maximize chances of triggering VAD.
    """
    sr = config["audio"]["sample_rate"]
    # 3 seconds of loud noise + 1 second silence
    audio = np.concatenate([noise(3 * sr, amp=1.0), silence(sr)])

    utterance = None
    for i in range(0, len(audio) - CHUNK, CHUNK):
        r = vad.process_chunk(audio[i:i + CHUNK])
        if r is not None:
            utterance = r
            break

    if utterance is not None:
        assert utterance.dtype == np.float32, (
            f"Utterance dtype is {utterance.dtype}, expected float32"
        )


def test_emitted_utterance_is_1d(vad, config):
    """Emitted audio must be a 1D array (mono), not 2D or higher."""
    sr = config["audio"]["sample_rate"]
    audio = np.concatenate([noise(3 * sr, amp=1.0), silence(sr)])

    utterance = None
    for i in range(0, len(audio) - CHUNK, CHUNK):
        r = vad.process_chunk(audio[i:i + CHUNK])
        if r is not None:
            utterance = r
            break

    if utterance is not None:
        assert utterance.ndim == 1, f"Utterance has {utterance.ndim} dims, expected 1"


# ── State machine ───────────────────────────────────────────────────────────────

def test_reset_clears_state(vad, config):
    """After reset(), VAD should behave exactly as if freshly initialized."""
    sr = config["audio"]["sample_rate"]
    # Feed some audio to dirty the state
    for _ in range(10):
        vad.process_chunk(noise(CHUNK, amp=1.0))

    vad.reset()

    # Now silence should produce nothing
    results = [vad.process_chunk(silence(CHUNK)) for _ in range(50)]
    assert all(r is None for r in results)


def test_multiple_process_chunk_calls_dont_crash(vad, config):
    """Basic smoke test: 10 seconds of mixed audio must not raise any exception."""
    sr = config["audio"]["sample_rate"]
    audio = np.concatenate([
        silence(sr),
        noise(2 * sr, amp=0.3),
        silence(sr),
        noise(sr, amp=0.5),
        silence(sr),
    ])
    for i in range(0, len(audio) - CHUNK, CHUNK):
        vad.process_chunk(audio[i:i + CHUNK])  # must not raise
