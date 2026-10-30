# Main pipeline coordinator — wires VAD → STT → guardrail → agent → TTS in the voice loop.
# Manages the conversation turn lifecycle and component hand-offs.
# Implementation: Part 7
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable

import numpy as np

from agent.graph import build_graph
from agent.nodes import wait_for_background_tasks
from agent.state import AgentState
from audio.audio_io import AudioIO
from audio.audio_manager import AudioManager
from audio.stt import WhisperSTT
from audio.tts import KokoroTTS, SentenceChunker, create_tts_engine
from audio.vad import SileroVAD
from guardrails.manager import GuardrailsManager
from memory.short_term import ShortTermMemory
from pipeline.fallback import FallbackManager
from pipeline.latency_monitor import LatencyMonitor

logger = logging.getLogger(__name__)


class ConversationPipeline:
    """Runs the full mic -> VAD -> STT -> guardrail -> agent -> TTS -> speaker loop.

    Every component here (VAD, STT, the LangGraph agent, guardrails, TTS,
    fallback, latency monitoring) was already built and independently tested in
    Parts 1-6. This class only orchestrates the hand-offs between them and owns
    the state that has to persist across turns — short-term memory and the open
    mic stream.

    Latency clock semantics: start_turn() is called the moment VAD confirms the
    user has stopped speaking (i.e. after the mandatory silence-detection wait,
    not from when the user actually stopped talking) — this matches how every
    prior Part's latency numbers were measured and reported (see
    docs/latency_report.md). The silence-detection wait itself is a separate,
    real cost disclosed alongside it, not folded into this "processing latency"
    number.
    """

    def __init__(self, config: dict, audio_manager: AudioIO | None = None) -> None:
        """audio_manager defaults to the real local-hardware AudioManager. Part 8's
        web demo passes its own WebSocketAudioIO instead — see docs/tradeoffs.md
        for why ConversationPipeline's turn logic is reused as-is rather than
        duplicated for the web path.
        """
        self._config = config
        self.audio_manager: AudioIO = audio_manager if audio_manager is not None else AudioManager(config)
        self.vad = SileroVAD(config)
        self.stt = WhisperSTT(config)
        self.tts: KokoroTTS = create_tts_engine(config)
        self.graph = build_graph(config)
        self.guardrails = GuardrailsManager(config)
        self.fallback = FallbackManager(config, self.tts, self.audio_manager)
        self.monitor = LatencyMonitor(config)
        self.short_term = ShortTermMemory(config)
        self._sentence_queue_maxsize: int = config["pipeline"]["sentence_queue_maxsize"]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def prewarm(self) -> None:
        """Send a dummy prompt through the LLM at startup so the first real turn
        isn't a ~36s cold start (see docs/latency_report.md's cold vs warm numbers).
        """
        if not self._config["pipeline"]["prewarm_llm"]:
            return
        prompt: str = self._config["pipeline"]["prewarm_prompt"]
        logger.info("Pre-warming LLM with dummy prompt %r...", prompt)
        t0 = time.perf_counter()
        state: AgentState = {
            "messages": [],
            "transcript": prompt,
            "retrieved_facts": [],
            "response_chunks": [],
        }
        async for _ in self.graph.astream(state, stream_mode="custom"):
            pass
        logger.info("LLM pre-warm complete [%.0fms]", (time.perf_counter() - t0) * 1000)

    async def run_forever(self) -> None:
        """Open the mic and process turns until interrupted (Ctrl+C).

        A turn-level exception is logged and recovered from rather than crashing
        the whole session — per this project's "no abrupt silences" UX
        requirement, one bad turn shouldn't end the conversation.
        """
        self.audio_manager.open_mic()
        try:
            while True:
                try:
                    await self.run_turn()
                except Exception:
                    logger.exception("Turn failed — recovering for next turn")
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
        finally:
            await self.close()

    async def close(self) -> None:
        """Waits for any in-flight extract_memory_node background task before
        releasing audio resources — otherwise stopping the process shortly
        after a turn silently loses that turn's long-term-memory write (see
        docs/tradeoffs.md).
        """
        await wait_for_background_tasks()
        self.audio_manager.close()

    # ── One turn ───────────────────────────────────────────────────────────────

    async def run_turn(self) -> None:
        """Capture one utterance, run it through the full pipeline, speak the response."""
        utterance = await asyncio.to_thread(self._listen_for_utterance)

        self.monitor.start_turn()
        transcript = await asyncio.to_thread(self.stt.transcribe, utterance)
        self.monitor.mark("stt")

        if not transcript:
            logger.info("Empty transcript (VAD fired but STT heard nothing) — skipping turn")
            return

        input_result = await self.guardrails.check_input(transcript)
        self.monitor.mark("input_guardrail")
        if not input_result.allowed:
            # Our own refusal text is fixed and known-safe — no need to run it
            # back through check_output, which would only add ~324ms for nothing.
            await self._speak_raw(input_result.message)
            self.monitor.mark("tts_complete")
            self.monitor.log_summary()
            return

        assistant_text = await self._generate_and_speak(transcript)
        self.monitor.mark("tts_complete")
        self.monitor.log_summary()

        if assistant_text:
            self.short_term.add_turn(transcript, assistant_text)

    def _listen_for_utterance(self) -> np.ndarray:
        """Blocking: read mic chunks through VAD until one complete utterance is captured."""
        while True:
            chunk = self.audio_manager.read_chunk()
            utterance = self.vad.process_chunk(chunk)
            if utterance is not None:
                return utterance

    # ── Generation + guarded, overlapped TTS ─────────────────────────────────────

    async def _generate_and_speak(self, transcript: str) -> str:
        """Race the fallback timer against the real LLM call, streaming sentences
        through the output guardrail into TTS as they complete.

        The fallback timer starts here — after the input guardrail check, not
        before — so it races only the LLM's own first-token latency, matching
        the plan's original intent ("if LLM hasn't emitted first token by
        then"). Starting it earlier (e.g. at run_turn's top) would make it race
        input-guardrail latency plus LLM latency combined, which — measured in
        Part 6 — pushes the combined time past the 800ms threshold on almost
        every turn once guardrails are in the loop. See docs/tradeoffs.md.

        Returns the full assistant response text actually spoken (for
        short-term memory) — this may differ from what the LLM generated if
        any sentence was blocked and replaced by the output guardrail's
        refusal line.
        """
        state: AgentState = {
            "messages": self.short_term.get_messages(),
            "transcript": transcript,
            "retrieved_facts": [],
            "response_chunks": [],
        }
        first_token_event = asyncio.Event()

        _fallback_fired, response = await asyncio.gather(
            self.fallback.watch(first_token_event),
            self._stream_and_speak(state, first_token_event),
        )
        return response

    def _is_stalling_phrase(self, sentence: str) -> bool:
        """Heuristic check to identify stalling/filler phrases generated by the LLM."""
        clean = re.sub(r"[^\w\s]", "", sentence.lower()).strip()
        stalling_patterns = [
            r"^let me think",
            r"^let me process",
            r"^let me check",
            r"^let me see",
            r"^let me look",
            r"^let me find",
            r"^give me a (moment|second|minute)",
            r"^just a (moment|second|minute)",
            r"^wait a (moment|second|minute)",
            r"^wait while i",
            r"^hold on a (moment|second|minute)",
            r"^one moment",
            r"^one second",
        ]
        return any(re.search(pat, clean) for pat in stalling_patterns)

    async def _stream_and_speak(self, state: AgentState, first_token_event: asyncio.Event) -> str:
        """Producer/consumer overlap, same pattern as audio.tts.SentenceStreamPlayer
        (Part 4): a producer pulls tokens from the graph and queues completed
        sentences; a consumer pulls off that queue and guardrail-checks +
        speaks them. Running these concurrently via asyncio.gather (not a
        single sequential loop) means sentence 2 can still be generating while
        sentence 1 is being checked/synthesized/played — losing this overlap
        by folding everything into one straight-line loop was an early version
        of this method's actual bug, caught during Part 7's own latency
        testing. SentenceStreamPlayer itself isn't reused here because it has
        no guardrail-check hook and guardrails must not leak into audio/tts.py
        (see docs/tradeoffs.md).

        The queue is bounded (config's pipeline.sentence_queue_maxsize) --
        backpressure between the producer and consumer: once it's full, the
        producer's queue.put() blocks until the consumer (guardrail check +
        synthesis + real-time playback, the slow side) frees a slot, so a
        fast-generating LLM can't queue sentences arbitrarily far ahead of
        how quickly they can actually be spoken. Rarely engages on this
        project's normal short (1-3 sentence) responses -- see docs/tradeoffs.md.
        """
        chunker = SentenceChunker()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self._sentence_queue_maxsize)
        spoken_parts: list[str] = []
        first_token_marked = False
        first_chunk_marked = False

        async def produce() -> None:
            nonlocal first_token_marked
            async for token in self.graph.astream(state, stream_mode="custom"):
                if not first_token_marked:
                    first_token_event.set()
                    self.monitor.mark("llm_first_token")
                    first_token_marked = True
                for sentence in chunker.feed(token):
                    await queue.put(sentence)
            trailing = chunker.flush()
            if trailing:
                await queue.put(trailing)
            await queue.put(None)  # sentinel: no more sentences coming

        async def consume() -> None:
            nonlocal first_chunk_marked
            is_first_sentence = True
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                if is_first_sentence:
                    is_first_sentence = False
                    if self._is_stalling_phrase(sentence):
                        logger.info("Discarding stalling first sentence: %r", sentence)
                        continue
                mark_first_chunk = None
                if not first_chunk_marked:
                    first_chunk_marked = True
                    mark_first_chunk = lambda: self.monitor.mark("tts_first_chunk")
                spoken = await self._speak_guarded(sentence, on_synthesized=mark_first_chunk)
                spoken_parts.append(spoken)

        await asyncio.gather(produce(), consume())
        return " ".join(spoken_parts)

    async def _speak_guarded(
        self, text: str, on_synthesized: Callable[[], None] | None = None
    ) -> str:
        """Output-guardrail-check one piece of LLM-generated text, then speak
        either it or the refusal line in its place. Returns whichever was
        actually spoken, so callers store the true spoken text, not the
        (possibly blocked) original.

        Runs the guardrail check concurrently with synthesizing `text`,
        instead of sequentially (Part 9 optimization) — the check typically
        takes longer than synthesis (~300-1000ms vs. ~100-400ms measured
        throughout this project), so overlapping them hides most of
        synthesis behind the check instead of stacking the two costs. The
        tradeoff: on the rare sentence that gets blocked, the synthesized
        audio for `text` is thrown away and the refusal line has to be
        synthesized separately — cheap wasted GPU work on an uncommon path,
        in exchange for real latency savings on every sentence. See
        docs/tradeoffs.md.
        """
        check_task = asyncio.create_task(self.guardrails.check_output(text))
        synth_task = asyncio.create_task(asyncio.to_thread(self.tts.synthesize, text))
        result, pcm = await asyncio.gather(check_task, synth_task)

        if on_synthesized is not None:
            on_synthesized()

        if result.allowed:
            await asyncio.to_thread(self.audio_manager.play_audio, pcm, self.tts.sample_rate)
            return text

        refusal_pcm = await asyncio.to_thread(self.tts.synthesize, result.message)
        await asyncio.to_thread(self.audio_manager.play_audio, refusal_pcm, self.tts.sample_rate)
        return result.message

    async def _speak_raw(self, text: str, on_synthesized: Callable[[], None] | None = None) -> None:
        """Synthesize + play text with no guardrail check — for text this
        pipeline generated itself (refusal lines, fallback filler phrases),
        which is already known-safe and doesn't need re-checking.

        on_synthesized, if given, fires right after synthesis (audio ready)
        and right before the blocking play_audio() call. This is where
        "first audio out" should be measured — play_audio() blocks for the
        real-time duration of the sentence being spoken, so marking after it
        returns would measure "finished speaking," not "started speaking."
        """
        pcm = await asyncio.to_thread(self.tts.synthesize, text)
        if on_synthesized is not None:
            on_synthesized()
        await asyncio.to_thread(self.audio_manager.play_audio, pcm, self.tts.sample_rate)
