"""
Silero VAD wrapper for real-time, streaming voice activity detection.

How it works:
  - Silero VAD expects exactly 512 samples per inference call (at 16kHz = 32ms of audio).
  - This class accepts chunks of ANY size from AudioManager and internally splits them
    into 512-sample windows via a ring buffer.
  - It runs a state machine: SILENCE → SPEECH → (silence timeout) → emit utterance.
  - Returns a complete float32 audio array (the full utterance) when the user stops talking.
  - Returns None on every other call.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SILERO_WINDOW = 512  # samples per inference call at 16kHz (non-negotiable for Silero)


class SileroVAD:
    """
    Stateful streaming VAD. Feed audio chunks via process_chunk().

    State machine:
        SILENCE: no buffering, waiting for speech.
        SPEECH:  buffering every window; counting silence windows.
        On silence timeout → emit utterance buffer and return to SILENCE.
        On too-short burst (< min_speech_duration_ms) → discard silently.
    """

    def __init__(self, config: dict) -> None:
        vad_cfg = config["vad"]
        self._sample_rate: int = config["audio"]["sample_rate"]
        self._threshold: float = vad_cfg["threshold"]

        # Convert ms thresholds → sample counts (for comparison against window counts)
        self._min_speech_samples: int = int(
            self._sample_rate * vad_cfg["min_speech_duration_ms"] / 1000
        )
        self._min_silence_samples: int = int(
            self._sample_rate * vad_cfg["min_silence_duration_ms"] / 1000
        )

        logger.info("Loading Silero VAD model via torch.hub...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
            trust_repo=True,
        )
        self._model.eval()
        logger.info("Silero VAD ready.")

        # Ring buffer: accumulates incoming samples until we have 512 to process
        self._ring: list[float] = []

        # State machine variables
        self._in_speech: bool = False
        self._utterance_buf: list[np.ndarray] = []  # windows collected during speech
        self._speech_samples: int = 0               # how many speech samples buffered
        self._silence_samples: int = 0              # consecutive silence samples since last speech

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed any-size float32 audio chunk.

        Returns:
            np.ndarray: complete utterance (float32, 16kHz) when user stops speaking.
            None:       in all other cases.
        """
        # Push incoming samples into the ring buffer
        self._ring.extend(chunk.tolist())

        # Drain the ring in 512-sample windows
        result: Optional[np.ndarray] = None
        while len(self._ring) >= _SILERO_WINDOW:
            window = np.array(self._ring[:_SILERO_WINDOW], dtype=np.float32)
            del self._ring[:_SILERO_WINDOW]
            fired = self._process_window(window)
            if fired is not None:
                result = fired  # keep the utterance; keep draining any leftover windows

        return result

    def reset(self) -> None:
        """Hard reset — call this when restarting the pipeline between sessions."""
        self._ring.clear()
        self._reset_state()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _process_window(self, window: np.ndarray) -> Optional[np.ndarray]:
        """Run one 512-sample window through Silero and advance the state machine."""
        score = self._score(window)
        is_speech = score >= self._threshold

        if is_speech:
            if not self._in_speech:
                logger.debug("VAD ▶ speech start (score=%.2f)", score)
                self._in_speech = True
            self._utterance_buf.append(window)
            self._speech_samples += _SILERO_WINDOW
            self._silence_samples = 0  # reset silence counter whenever speech resumes

        elif self._in_speech:
            # We're in SPEECH state but this window is silent
            # Buffer it anyway (catches the natural tail/decay of speech)
            self._utterance_buf.append(window)
            self._silence_samples += _SILERO_WINDOW

            if self._silence_samples >= self._min_silence_samples:
                # Enough silence has passed — the user finished their utterance
                logger.debug(
                    "VAD ■ speech end — %dms speech, %dms silence",
                    self._speech_samples * 1000 // self._sample_rate,
                    self._silence_samples * 1000 // self._sample_rate,
                )
                if self._speech_samples >= self._min_speech_samples:
                    utterance = np.concatenate(self._utterance_buf)
                    self._reset_state()
                    return utterance
                else:
                    # Too short — noise burst, click, cough. Discard.
                    logger.debug("VAD ✗ discarding short burst (%dms)",
                                 self._speech_samples * 1000 // self._sample_rate)
                    self._reset_state()

        return None

    def _score(self, window: np.ndarray) -> float:
        """Run Silero VAD on exactly 512 samples. Returns P(speech) in [0.0, 1.0]."""
        tensor = torch.from_numpy(window).unsqueeze(0)  # [1, 512]
        with torch.no_grad():
            return float(self._model(tensor, self._sample_rate).item())

    def _reset_state(self) -> None:
        self._in_speech = False
        self._utterance_buf = []
        self._speech_samples = 0
        self._silence_samples = 0
        try:
            self._model.reset_states()  # clear Silero's internal GRU hidden state
        except AttributeError:
            pass  # older versions don't expose reset_states
