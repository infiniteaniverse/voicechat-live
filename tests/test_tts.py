# Tests for TTS: audio output correctness + first-chunk latency.
# Implementation: Part 4
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def kokoro(config):
    from audio.tts import KokoroTTS
    return KokoroTTS(config)


# ── SentenceChunker — pure logic, no model needed ────────────────────────────────

def test_emits_sentence_on_period():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    sentences = chunker.feed("Hello there. ")
    assert sentences == ["Hello there."]


def test_emits_sentence_on_question_mark():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    sentences = chunker.feed("How are you? ")
    assert sentences == ["How are you?"]


def test_emits_sentence_on_exclamation():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    sentences = chunker.feed("Watch out! ")
    assert sentences == ["Watch out!"]


def test_no_emit_without_trailing_whitespace():
    """A token ending mid-sentence must not emit until a boundary is actually seen."""
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    assert chunker.feed("Hello") == []
    assert chunker.feed(" there") == []


def test_boundary_split_across_multiple_feed_calls():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    assert chunker.feed("The answer") == []
    assert chunker.feed(" is") == []
    assert chunker.feed(" 42.") == []  # no trailing whitespace yet
    assert chunker.feed(" Next sentence.") == ["The answer is 42."]


def test_multiple_sentences_in_one_feed_call():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    sentences = chunker.feed("First one. Second one. Third pending")
    assert sentences == ["First one.", "Second one."]


def test_flush_returns_trailing_partial():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    chunker.feed("Trailing thought with no boundary")
    assert chunker.flush() == "Trailing thought with no boundary"


def test_flush_returns_none_when_buffer_empty():
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    chunker.feed("Complete sentence. ")
    assert chunker.flush() is None


def test_decimal_number_does_not_split_early():
    """A '.' not followed by whitespace (like in '3.14') must not trigger a boundary."""
    from audio.tts import SentenceChunker
    chunker = SentenceChunker()
    sentences = chunker.feed("Pi is about 3.14 which is neat. ")
    assert sentences == ["Pi is about 3.14 which is neat."]


# ── KokoroTTS — real model ───────────────────────────────────────────────────────

def test_synthesize_returns_float32_mono_array(kokoro):
    audio = kokoro.synthesize("This is a short test sentence.")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1


def test_synthesize_sample_rate_is_24000(kokoro):
    assert kokoro.sample_rate == 24000


def test_synthesize_produces_nonsilent_audio(kokoro):
    audio = kokoro.synthesize("This sentence should produce audible sound.")
    assert np.abs(audio).max() > 0.01, "Synthesized audio is silent — engine may be misconfigured"


def test_synthesize_latency_within_budget(kokoro):
    """
    Median warm synthesis latency for a normal sentence should be well under 300ms
    (measured ~100-170ms on RTX 4060 with CUDA torch; 300ms leaves headroom for
    slower hardware while still catching a regression back to CPU-only torch).
    """
    sentence = "Machine learning engineers build systems that learn from data."

    kokoro.synthesize(sentence)  # warm-up

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        kokoro.synthesize(sentence)
        latencies.append((time.perf_counter() - t0) * 1000)

    median = sorted(latencies)[len(latencies) // 2]
    assert median < 300, (
        f"TTS median latency {median:.0f}ms exceeds 300ms budget. "
        f"Check that torch has CUDA support (torch.cuda.is_available())."
    )


# ── create_tts_engine ────────────────────────────────────────────────────────────

def test_create_tts_engine_returns_kokoro_for_kokoro_config(config):
    from audio.tts import create_tts_engine, KokoroTTS
    engine = create_tts_engine(config)
    assert isinstance(engine, KokoroTTS)


def test_create_tts_engine_raises_for_unimplemented_engine(config):
    import copy
    from audio.tts import create_tts_engine

    bad_config = copy.deepcopy(config)
    bad_config["tts"]["engine"] = "piper"
    with pytest.raises(NotImplementedError):
        create_tts_engine(bad_config)


# ── SentenceStreamPlayer — orchestration, with a fake token stream ───────────────

async def _fake_token_stream(sentences: list[str]):
    """Yields tokens one word at a time, joined so SentenceChunker sees real boundaries."""
    for sentence in sentences:
        for word in sentence.split(" "):
            yield word + " "


@pytest.mark.asyncio
async def test_sentence_stream_player_synthesizes_each_sentence(config, kokoro):
    from audio.tts import SentenceStreamPlayer

    played: list[tuple[np.ndarray, int]] = []

    class FakeAudioManager:
        def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
            played.append((pcm, sample_rate))

    player = SentenceStreamPlayer(kokoro, FakeAudioManager())
    sentences = ["This is the first sentence.", "This is the second sentence."]
    await player.speak_stream(_fake_token_stream(sentences))

    assert len(played) == 2
    for pcm, sample_rate in played:
        assert sample_rate == 24000
        assert isinstance(pcm, np.ndarray)
        assert pcm.dtype == np.float32
        assert np.abs(pcm).max() > 0.01
