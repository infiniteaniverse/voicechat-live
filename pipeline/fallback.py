# Fallback Manager — fires filler phrases when LLM latency exceeds threshold.
# Proactive engagement: doesn't wait for an error, fires on slow response.
# Filler phrases loaded from config.yaml.
# Implementation: Part 6
from __future__ import annotations

import asyncio
import logging
import random
import time

from audio.audio_io import AudioIO
from audio.tts import KokoroTTS

logger = logging.getLogger(__name__)


class FallbackManager:
    """Races a timeout against a first-token signal; speaks a filler phrase if
    the timeout wins.

    Usage: call watch(first_token_event) concurrently with the real generation
    call (e.g. via asyncio.gather), setting first_token_event as soon as the
    LLM's first token arrives. If the event fires before the configured
    timeout, watch() returns False and does nothing. If the timeout fires
    first, watch() synthesizes and speaks a random filler phrase through the
    real TTS engine + audio manager, then returns True. Fires on slowness, not
    on error — the assistant should sound like it's still "with you" during a
    slow response, not silently wait or announce a failure.

    Wiring watch() concurrently with generate_node's real streaming call is
    Part 7's job (pipeline-level orchestration); this class only implements
    the race + phrase-speaking mechanism in isolation.
    """

    def __init__(self, config: dict, tts_engine: KokoroTTS, audio_manager: AudioIO) -> None:
        fb_cfg = config["fallback"]
        self._timeout_s: float = fb_cfg["llm_first_token_timeout_ms"] / 1000
        self._filler_phrases: list[str] = fb_cfg["filler_phrases"]
        self._tts = tts_engine
        self._audio_manager = audio_manager

    def pick_filler_phrase(self) -> str:
        return random.choice(self._filler_phrases)

    async def watch(self, first_token_event: asyncio.Event) -> bool:
        """Returns True if the fallback fired (timeout won the race), False if
        first_token_event was set before the timeout.
        """
        try:
            await asyncio.wait_for(first_token_event.wait(), timeout=self._timeout_s)
            return False
        except asyncio.TimeoutError:
            phrase = self.pick_filler_phrase()
            logger.info(
                "Fallback firing (no first token within %.0fms): %r",
                self._timeout_s * 1000,
                phrase,
            )
            await self._speak(phrase)
            return True

    async def _speak(self, phrase: str) -> None:
        t0 = time.perf_counter()
        pcm = await asyncio.to_thread(self._tts.synthesize, phrase)
        await asyncio.to_thread(self._audio_manager.play_audio, pcm, self._tts.sample_rate)
        logger.info("Fallback phrase spoken [%.0fms]", (time.perf_counter() - t0) * 1000)
