# WebSocket handler — receives raw PCM audio from browser, feeds into pipeline,
# sends back TTS audio chunks as binary frames.
# Implementation: Part 8 (stretch goal)
from __future__ import annotations

import asyncio
import json
import logging
import math
import queue

import numpy as np
import scipy.signal
from fastapi import WebSocket, WebSocketDisconnect

from pipeline.pipeline import ConversationPipeline

logger = logging.getLogger(__name__)

_READ_TIMEOUT_S = 30.0   # safety net only — real audio arrives every ~32ms while connected
_SEND_TIMEOUT_S = 5.0
_DISCONNECT_SENTINEL = object()


class WebSocketSessionEnded(Exception):
    """Raised out of WebSocketAudioIO.read_chunk() when the browser has
    disconnected, so ConversationPipeline._listen_for_utterance's blocking
    while-True loop can't hang a thread-pool thread forever waiting for audio
    that will never arrive.
    """


def _resample_to_16k(chunk: np.ndarray, input_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Browser mic capture is at the browser's native rate (typically 44100 or
    48000Hz), but VAD/STT expect exactly 16kHz. Rational (polyphase) resample
    rather than forcing the browser's AudioContext to 16000Hz, which isn't
    reliably honored across browsers/hardware — see docs/tradeoffs.md.
    """
    if input_rate == target_rate:
        return chunk.astype(np.float32)
    g = math.gcd(input_rate, target_rate)
    up, down = target_rate // g, input_rate // g
    return scipy.signal.resample_poly(chunk, up, down).astype(np.float32)


class WebSocketAudioIO:
    """Implements the same shape as audio.audio_io.AudioIO, sourcing mic audio
    from and sinking TTS audio to a browser over a WebSocket instead of local
    PyAudio hardware. One instance is shared across sessions (attach()/detach()
    bind it to whichever connection is currently active); see
    docs/tradeoffs.md for why ConversationPipeline is reused as-is against
    this instead of a second, duplicated turn-orchestration path.
    """

    def __init__(self) -> None:
        self._websocket: WebSocket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbound: queue.Queue = queue.Queue()
        self._spoke_this_turn = False
        # True while the assistant's own TTS audio is playing (or has just
        # finished) this turn. The browser streams mic audio continuously,
        # including while the speakers are playing the response — with no
        # echo cancellation, the mic picks up the assistant's own voice.
        # Without this gate, that gets queued during playback and fed to VAD
        # as the "next" utterance the moment listening resumes, so the
        # assistant ends up responding to itself in a runaway chain. See
        # docs/tradeoffs.md.
        self._discard_incoming = False

    def attach(self, websocket: WebSocket, input_sample_rate: int) -> None:
        self._websocket = websocket
        self._loop = asyncio.get_running_loop()
        self._discard_incoming = False
        self._drain_inbound()

    def detach(self) -> None:
        self._websocket = None
        self._loop = None
        self._inbound.put(_DISCONNECT_SENTINEL)

    def begin_turn(self) -> None:
        """Call once per turn, before pipeline.run_turn(): resets the
        'speaking' status flag and, critically, drops any audio that was
        captured (and queued) while the previous turn's response was
        playing, then re-opens the gate so this turn can actually listen.
        """
        self._spoke_this_turn = False
        self._discard_incoming = False
        self._drain_inbound()

    def _drain_inbound(self) -> None:
        while not self._inbound.empty():
            try:
                self._inbound.get_nowait()
            except queue.Empty:
                break

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Called from the event loop (the receive loop) with an
        already-resampled 16kHz chunk. Dropped, not queued, while the
        assistant is speaking — see _discard_incoming above.
        """
        if self._discard_incoming:
            return
        self._inbound.put(chunk)

    # ── AudioIO shape ─────────────────────────────────────────────────────────

    def open_mic(self) -> None:
        pass  # nothing to open — the WebSocket is already accepted before the turn loop starts

    def read_chunk(self) -> np.ndarray:
        """Blocking — called via asyncio.to_thread from a worker thread."""
        try:
            chunk = self._inbound.get(timeout=_READ_TIMEOUT_S)
        except queue.Empty:
            raise WebSocketSessionEnded("No audio received within timeout") from None
        if chunk is _DISCONNECT_SENTINEL:
            raise WebSocketSessionEnded("WebSocket disconnected")
        return chunk

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Blocking — called via asyncio.to_thread from a worker thread.
        Bridges back onto the event loop so the send coroutine actually runs,
        and blocks the calling thread until it completes, matching
        AudioManager.play_audio's blocking-until-written semantics.
        """
        websocket = self._websocket
        loop = self._loop
        if websocket is None or loop is None:
            return  # session already ended — nothing to play to

        self._discard_incoming = True

        async def _send() -> None:
            if not self._spoke_this_turn:
                self._spoke_this_turn = True
                await websocket.send_json({"type": "status", "state": "speaking"})
            await websocket.send_bytes(pcm.astype(np.float32).tobytes())

        future = asyncio.run_coroutine_threadsafe(_send(), loop)
        try:
            future.result(timeout=_SEND_TIMEOUT_S)
        except Exception:
            logger.warning("Failed to send audio to browser (session likely ended)")

    def close(self) -> None:
        self.detach()


async def _safe_send_json(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


async def _safe_close(websocket: WebSocket, code: int = 1002) -> None:
    try:
        await websocket.close(code=code)
    except Exception:
        pass


async def handle_connection(
    websocket: WebSocket, pipeline: ConversationPipeline, audio_io: WebSocketAudioIO
) -> None:
    """Runs one browser session: reset shared session state, then run a
    continuous multi-turn conversation (VAD auto-detects each utterance,
    same as the CLI) until the browser disconnects.

    Deliberately doesn't reuse ConversationPipeline.run_forever() — its
    open_mic()-then-loop-until-KeyboardInterrupt shape doesn't fit a
    WebSocket's disconnect-driven lifecycle.
    """
    await websocket.accept()

    try:
        raw_start = await websocket.receive_text()
        session_start = json.loads(raw_start)
        input_sample_rate = int(session_start["input_sample_rate"])
    except (WebSocketDisconnect, KeyError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Web session failed to start (bad or missing session_start message)")
        await _safe_close(websocket)
        return

    logger.info("Web session starting (browser sample rate: %dHz)", input_sample_rate)
    pipeline.vad.reset()
    pipeline.short_term.clear()
    audio_io.attach(websocket, input_sample_rate)

    stop_event = asyncio.Event()

    async def receive_loop() -> None:
        try:
            while True:
                raw = await websocket.receive_bytes()
                chunk = np.frombuffer(raw, dtype=np.float32)
                audio_io.push_chunk(_resample_to_16k(chunk, input_sample_rate))
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()
            audio_io.detach()

    async def turn_loop() -> None:
        while not stop_event.is_set():
            audio_io.begin_turn()
            try:
                await websocket.send_json({"type": "status", "state": "listening"})
                await pipeline.run_turn()
                durations = pipeline.monitor.stage_durations()
                await websocket.send_json(
                    {"type": "latency", "stages": durations, "total_ms": sum(durations.values())}
                )
            except WebSocketSessionEnded:
                break
            except Exception as exc:
                logger.exception("Turn failed in web session")
                await _safe_send_json(websocket, {"type": "error", "message": str(exc)})

    await asyncio.gather(receive_loop(), turn_loop())
    logger.info("Web session ended")
