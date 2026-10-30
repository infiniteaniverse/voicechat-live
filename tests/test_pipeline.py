# Fast, isolated tests of ConversationPipeline's own orchestration logic —
# no real models needed (contrast with tests/test_pipeline_e2e.py, which uses
# the real STT/LLM/TTS stack). Currently covers the bounded sentence-queue
# backpressure added on top of Part 7's producer/consumer pipeline.
# Implementation: Part 9 follow-up (OS-concepts pass)
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np
import pytest

from guardrails.manager import GuardrailResult
from pipeline.pipeline import ConversationPipeline


class _TimestampedFakeGraph:
    """Fake LangGraph-like object: astream() yields pre-baked, already-complete
    sentence strings (so each one closes out a sentence on the very first
    chunker.feed() call) and records a timestamp every time a token is
    requested. produce() only requests the next token after successfully
    completing queue.put() for the previous one, so the gap between
    consecutive request timestamps directly measures how long produce() was
    blocked waiting for queue space — not an assumption about asyncio.Queue's
    documented behavior, a real measured effect of the actual code path.
    """

    def __init__(self, sentences: list[str]) -> None:
        self._sentences = sentences
        self.request_timestamps: list[float] = []

    def astream(self, state: dict, stream_mode: str = "custom") -> AsyncIterator[str]:
        async def _gen():
            for sentence in self._sentences:
                self.request_timestamps.append(time.perf_counter())
                yield sentence

        return _gen()


class _FakeGuardrails:
    """Always allows, near-instantly — isolates the test to the queue's own
    bounding behavior rather than guardrail latency."""

    async def check_output(self, text: str) -> GuardrailResult:
        return GuardrailResult(allowed=True, message=None)


class _FakeTTS:
    sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)


class _SlowFakeAudioManager:
    """Stands in for the consumer's slow step (real-time audio playback) with
    a controllable, deterministic delay instead of actually playing audio."""

    def __init__(self, play_delay_s: float) -> None:
        self._play_delay_s = play_delay_s
        self.play_count = 0

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        time.sleep(self._play_delay_s)
        self.play_count += 1


class _FakeMonitor:
    def mark(self, stage: str) -> None:
        pass


class _FakePipelineSelf:
    """Minimal stand-in exposing exactly what ConversationPipeline._stream_and_speak
    and ._speak_guarded need via `self`, so those real (unbound) methods can be
    exercised directly without constructing a full ConversationPipeline — which
    would load five real models (Whisper, Silero, Kokoro, the embedder, the
    guardrails config) just to test queue-bounding logic that doesn't depend on
    any of them. Borrowing the real methods as class attributes (Python's
    normal method-binding machinery applies to any class, not just the one
    that defined them) means this test exercises the actual production code,
    not a reimplementation of it.
    """

    _stream_and_speak = ConversationPipeline._stream_and_speak
    _speak_guarded = ConversationPipeline._speak_guarded
    _is_stalling_phrase = ConversationPipeline._is_stalling_phrase

    def __init__(self, graph, guardrails, tts, audio_manager, monitor, sentence_queue_maxsize: int) -> None:
        self.graph = graph
        self.guardrails = guardrails
        self.tts = tts
        self.audio_manager = audio_manager
        self.monitor = monitor
        self._sentence_queue_maxsize = sentence_queue_maxsize


@pytest.mark.asyncio
async def test_sentence_queue_backpressure_blocks_producer_when_full():
    """Proves the bounded sentence queue (config's pipeline.sentence_queue_maxsize)
    actually creates backpressure — not just that asyncio.Queue's maxsize is
    documented to work this way, but that this specific code path (the real,
    unmodified _stream_and_speak) actually exhibits it.

    Subtlety that shaped this test: queue.get() frees a slot the instant it's
    called, independent of how long the caller then spends on the item
    afterward — so a blocked put() for item N can unblock quickly (as soon as
    the consumer's *next* get() call happens), even while the consumer is
    still slowly finishing item N-1's real work in the background. That
    delay only becomes externally observable once the *following* item's
    token request has to wait for it — which is why this test uses one more
    sentence than the queue's maxsize, rather than exactly filling it: the
    delay from the last blocked put() would otherwise have no subsequent
    "next token request" timestamp to show up in.
    """
    consumer_delay_s = 0.2
    sentences = ["One. ", "Two. ", "Three. ", "Four. ", "Five. "]

    async def request_gaps(maxsize: int) -> list[float]:
        graph = _TimestampedFakeGraph(sentences)
        fake_self = _FakePipelineSelf(
            graph=graph,
            guardrails=_FakeGuardrails(),
            tts=_FakeTTS(),
            audio_manager=_SlowFakeAudioManager(consumer_delay_s),
            monitor=_FakeMonitor(),
            sentence_queue_maxsize=maxsize,
        )
        state = {"messages": [], "transcript": "", "retrieved_facts": [], "response_chunks": []}
        await fake_self._stream_and_speak(state, asyncio.Event())
        ts = graph.request_timestamps
        return [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]

    unbounded_gaps = await request_gaps(maxsize=len(sentences))  # room for everything at once
    bounded_gaps = await request_gaps(maxsize=2)

    assert max(unbounded_gaps) < 0.05, (
        f"with room for every sentence at once, every token should be requested "
        f"near-instantly, got gaps={[f'{g:.3f}' for g in unbounded_gaps]} — test harness may be wrong"
    )
    assert max(bounded_gaps) > consumer_delay_s * 0.5, (
        f"a small bounded queue (maxsize=2) against a slow consumer should measurably "
        f"delay at least one later token request, but the largest gap was only "
        f"{max(bounded_gaps):.3f}s (gaps={[f'{g:.3f}' for g in bounded_gaps]}) "
        f"— backpressure may not be engaging"
    )


@pytest.mark.asyncio
async def test_sentence_queue_backpressure_does_not_affect_typical_short_responses():
    """The honest counterpart to the test above: this project's real system
    prompt asks for 1-3 sentence answers, so the default maxsize=3 should
    essentially never engage in normal use — confirms the bound doesn't
    introduce any delay for a response that fits within it.
    """
    sentences = ["One. ", "Two. ", "Three. "]  # fits entirely within maxsize=3, never blocks

    graph = _TimestampedFakeGraph(sentences)
    fake_self = _FakePipelineSelf(
        graph=graph,
        guardrails=_FakeGuardrails(),
        tts=_FakeTTS(),
        audio_manager=_SlowFakeAudioManager(play_delay_s=0.2),
        monitor=_FakeMonitor(),
        sentence_queue_maxsize=3,
    )
    state = {"messages": [], "transcript": "", "retrieved_facts": [], "response_chunks": []}

    t0 = time.perf_counter()
    await fake_self._stream_and_speak(state, asyncio.Event())
    production_time = graph.request_timestamps[-1] - graph.request_timestamps[0]

    assert production_time < 0.05, (
        f"a 3-sentence response should never hit the maxsize=3 bound, expected near-instant "
        f"token production, got {production_time:.3f}s"
    )


def test_is_stalling_phrase():
    """Verify that _is_stalling_phrase correctly identifies stalling phrases."""
    fake_self = _FakePipelineSelf(None, None, None, None, None, 3)
    
    stalling = [
        "Let me think about that.",
        "Wait while I process that...",
        "Give me a second.",
        "Just a moment.",
        "Wait a moment.",
        "Let me check that.",
        "Let me see.",
        "One second, let me think."
    ]
    for phrase in stalling:
        assert fake_self._is_stalling_phrase(phrase) is True, f"Failed to detect stalling: {phrase}"

    normal = [
        "The capital of France is Paris.",
        "Yes, I can help you with that.",
        "Checking if a file exists is simple.",
        "Let's get started.",
        "Here is the information."
    ]
    for phrase in normal:
        assert fake_self._is_stalling_phrase(phrase) is False, f"Incorrectly flagged normal phrase: {phrase}"


@pytest.mark.asyncio
async def test_first_sentence_stalling_filter():
    """Verify that a stalling first sentence is discarded, but subsequent ones are kept."""
    sentences = ["Let me think about that. ", "The capital of France is Paris. ", "It is a beautiful city. "]
    graph = _TimestampedFakeGraph(sentences)
    audio_manager = _SlowFakeAudioManager(play_delay_s=0.0)
    fake_self = _FakePipelineSelf(
        graph=graph,
        guardrails=_FakeGuardrails(),
        tts=_FakeTTS(),
        audio_manager=audio_manager,
        monitor=_FakeMonitor(),
        sentence_queue_maxsize=3,
    )
    state = {"messages": [], "transcript": "", "retrieved_facts": [], "response_chunks": []}

    spoken = await fake_self._stream_and_speak(state, asyncio.Event())
    
    # The first stalling sentence should be dropped, but the rest kept
    assert spoken == "The capital of France is Paris. It is a beautiful city."
    # We should only have synthesized/played 2 sentences, not 3
    assert audio_manager.play_count == 2

