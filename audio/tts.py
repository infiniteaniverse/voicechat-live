# Text-to-Speech — Kokoro (primary) / Piper (fallback) wrapper.
# Accepts sentence strings, streams audio chunks to speaker.
# Sentence-level chunking keeps first-audio latency low.
# Implementation: Part 4
from __future__ import annotations

import asyncio
import logging
import re
import time
import warnings
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Suppress two deprecation warnings emitted by Kokoro's own model code during
# KPipeline init — both originate inside kokoro/modules.py and kokoro/istftnet.py
# calling torch APIs that have since been deprecated. We can't change those
# library files, so we filter by the exact torch source module.
#
# 1. UserWarning from torch/nn/modules/rnn.py — "dropout option adds dropout
#    after all but last recurrent layer..." (Kokoro passes dropout=0.2 to a
#    single-layer LSTM; harmless but noisy).
# 2. FutureWarning from torch/nn/utils/weight_norm.py — "torch.nn.utils.weight_norm
#    is deprecated in favor of torch.nn.utils.parametrizations.weight_norm"
#    (Kokoro's istftnet uses the old API).
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
    module=r"torch\.nn\.modules\.rnn",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)

# Matches sentence-ending punctuation followed by whitespace — the whitespace
# requirement is what keeps decimals ("3.14 ") from splitting (no space after
# the '.'), at the cost of also splitting on mid-sentence abbreviations
# ("Dr. Smith"). Accepted simplification — see docs/tradeoffs.md.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s")


class SentenceChunker:
    """Buffers streamed LLM tokens, emits complete sentences on '.', '?', '!' boundaries.

    Stateful: feed tokens as they arrive, call flush() once at end-of-stream to
    emit any trailing partial sentence (which won't have trailing whitespace to
    match on).
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Add a token, return any complete sentences it caused to close out."""
        self._buffer += token
        sentences: list[str] = []

        while True:
            match = _SENTENCE_BOUNDARY.search(self._buffer)
            if match is None:
                break
            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]
            if sentence:
                sentences.append(sentence)

        return sentences

    def flush(self) -> Optional[str]:
        """Call once the token stream ends. Returns any remaining partial text, or None."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None


class KokoroTTS:
    """Thin wrapper around Kokoro-82M. Pipeline loaded once at init and reused."""

    def __init__(self, config: dict) -> None:
        from kokoro import KPipeline

        kokoro_cfg = config["tts"]["kokoro"]
        self._voice: str = kokoro_cfg["voice"]
        self._speed: float = kokoro_cfg["speed"]
        self.sample_rate = 24000  # fixed by the Kokoro-82M model, not configurable

        logger.info("Loading Kokoro pipeline...")
        t0 = time.perf_counter()
        self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
        logger.info("Kokoro pipeline ready in %.2fs", time.perf_counter() - t0)

    def synthesize(self, sentence: str) -> np.ndarray:
        """Synthesize one sentence. Returns float32 PCM, mono, at self.sample_rate."""
        t0 = time.perf_counter()
        results = list(self._pipeline(sentence, voice=self._voice, speed=self._speed))
        audio = np.concatenate([r.audio.numpy() for r in results]).astype(np.float32)

        logger.info(
            "TTS [%.0fms]: %d samples (%.2fs audio) for %r",
            (time.perf_counter() - t0) * 1000,
            len(audio),
            len(audio) / self.sample_rate,
            sentence[:60],
        )
        return audio


def create_tts_engine(config: dict) -> KokoroTTS:
    """Config-driven TTS engine selection. Only 'kokoro' is implemented — the
    config's 'piper' section is a placeholder for a fallback that isn't wired up
    yet (piper-tts isn't installed; not needed while Kokoro works).
    """
    engine_name = config["tts"]["engine"]
    if engine_name == "kokoro":
        return KokoroTTS(config)
    raise NotImplementedError(
        f"tts.engine={engine_name!r} is not implemented — only 'kokoro' is currently supported."
    )


class SentenceStreamPlayer:
    """Ties SentenceChunker + a TTS engine + speaker playback together.

    Runs the "overlap" pattern from the implementation plan: as tokens stream
    in, completed sentences are queued for synthesis+playback immediately,
    concurrently with the rest of the response still generating. Synthesis and
    playback are blocking calls (Kokoro inference, PyAudio writes) so both run
    via asyncio.to_thread() to avoid stalling the event loop.
    """

    def __init__(self, tts_engine: KokoroTTS, audio_manager) -> None:
        self._tts = tts_engine
        self._audio_manager = audio_manager

    async def speak_stream(self, tokens: AsyncIterator[str]) -> None:
        """Consume a stream of raw tokens; synthesize+play each sentence as it completes."""
        chunker = SentenceChunker()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def produce() -> None:
            async for token in tokens:
                for sentence in chunker.feed(token):
                    await queue.put(sentence)
            trailing = chunker.flush()
            if trailing:
                await queue.put(trailing)
            await queue.put(None)  # sentinel: no more sentences coming

        async def consume() -> None:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                audio = await asyncio.to_thread(self._tts.synthesize, sentence)
                await asyncio.to_thread(self._audio_manager.play_audio, audio, self._tts.sample_rate)

        await asyncio.gather(produce(), consume())
