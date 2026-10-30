# Tests for FallbackManager: timeout-vs-first-token race + filler phrase playback.
# Implementation: Part 6
#
# Uses real Kokoro synthesis (matches this project's "real models, no mocks" testing
# philosophy — see tests/test_tts.py) with a FakeAudioManager standing in only for the
# speaker hardware, so tests don't require actual audio output.
import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.fallback import FallbackManager

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def kokoro(config):
    from audio.tts import KokoroTTS
    return KokoroTTS(config)


class FakeAudioManager:
    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))


def _fast_fallback_config(config: dict, timeout_ms: int = 80) -> dict:
    """A copy of the real fallback config with a short timeout, so tests that
    expect the fallback to fire don't have to wait out the real 800ms budget.
    """
    import copy
    fast = copy.deepcopy(config)
    fast["fallback"]["llm_first_token_timeout_ms"] = timeout_ms
    return fast


def test_pick_filler_phrase_returns_one_of_configured_phrases(config, kokoro):
    audio_manager = FakeAudioManager()
    manager = FallbackManager(config, kokoro, audio_manager)
    for _ in range(20):
        assert manager.pick_filler_phrase() in config["fallback"]["filler_phrases"]


@pytest.mark.asyncio
async def test_watch_returns_false_and_stays_silent_when_first_token_arrives_in_time(config, kokoro):
    audio_manager = FakeAudioManager()
    manager = FallbackManager(_fast_fallback_config(config), kokoro, audio_manager)

    event = asyncio.Event()
    event.set()  # first token "already arrived"

    fired = await manager.watch(event)

    assert fired is False
    assert audio_manager.played == []


@pytest.mark.asyncio
async def test_watch_returns_true_and_speaks_a_filler_phrase_on_timeout(config, kokoro):
    audio_manager = FakeAudioManager()
    manager = FallbackManager(_fast_fallback_config(config, timeout_ms=80), kokoro, audio_manager)

    event = asyncio.Event()  # never set — simulates a slow LLM

    fired = await manager.watch(event)

    assert fired is True
    assert len(audio_manager.played) == 1
    pcm, sample_rate = audio_manager.played[0]
    assert sample_rate == 24000
    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.float32
    assert np.abs(pcm).max() > 0.01, "Filler phrase audio is silent — TTS may be misconfigured"


@pytest.mark.asyncio
async def test_watch_fires_close_to_the_configured_timeout(config, kokoro):
    import time

    audio_manager = FakeAudioManager()
    timeout_ms = 100
    manager = FallbackManager(_fast_fallback_config(config, timeout_ms=timeout_ms), kokoro, audio_manager)

    event = asyncio.Event()
    t0 = time.perf_counter()
    await manager.watch(event)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Should fire at ~timeout_ms, not noticeably later than that plus phrase synthesis.
    assert elapsed_ms >= timeout_ms
    assert elapsed_ms < timeout_ms + 2000
