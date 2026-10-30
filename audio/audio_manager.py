"""
Raw PCM audio I/O — microphone capture (Part 1) and speaker playback (Part 4).
All audio flowing through this class is float32, mono, 16kHz, normalized to [-1.0, 1.0].
"""
from __future__ import annotations

import logging
from typing import Generator

import numpy as np
import pyaudio

logger = logging.getLogger(__name__)


class AudioManager:
    """
    Thin wrapper around PyAudio for mic capture and speaker output.
    Decouples hardware I/O from the VAD/TTS processing logic above it.

    Usage (mic):
        with AudioManager(config) as am:
            for chunk in am.stream_mic():
                process(chunk)
    """

    def __init__(self, config: dict) -> None:
        cfg = config["audio"]
        self._sample_rate: int = cfg["sample_rate"]          # 16000 Hz
        self._channels: int = cfg["channels"]                # 1 (mono)
        self._chunk_size: int = cfg["chunk_size"]            # 1024 frames per read
        self._input_device: int | None = cfg.get("input_device_index")   # None = system default
        self._output_device: int | None = cfg.get("output_device_index") # None = system default

        self._pa = pyaudio.PyAudio()
        self._mic_stream: pyaudio.Stream | None = None
        self._out_stream: pyaudio.Stream | None = None
        self._out_sample_rate: int | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # ── Mic capture ────────────────────────────────────────────────────────────

    def open_mic(self) -> None:
        """Open the microphone input stream. Idempotent."""
        if self._mic_stream is not None:
            return
        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16,       # 16-bit signed integers from hardware
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._input_device,
            frames_per_buffer=self._chunk_size,
        )
        logger.info(
            "Mic opened — device=%s  rate=%dHz  chunk=%d frames",
            self._input_device or "default",
            self._sample_rate,
            self._chunk_size,
        )

    def read_chunk(self) -> np.ndarray:
        """
        Read one chunk of audio from the mic.
        Converts int16 PCM from hardware → float32 normalized to [-1.0, 1.0].
        Shape: (chunk_size,)
        """
        raw: bytes = self._mic_stream.read(
            self._chunk_size, exception_on_overflow=False
        )
        # int16 range is [-32768, 32767] — dividing by 32768 gives [-1.0, ~1.0]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def stream_mic(self) -> Generator[np.ndarray, None, None]:
        """
        Infinite generator: opens the mic and yields float32 chunks continuously.
        Closes the mic cleanly on KeyboardInterrupt or when the caller breaks.
        """
        self.open_mic()
        try:
            while True:
                yield self.read_chunk()
        except KeyboardInterrupt:
            pass
        finally:
            self.close_mic()

    def close_mic(self) -> None:
        if self._mic_stream is not None:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None
            logger.info("Mic stream closed.")

    # ── Speaker playback ─────────────────────────────────────────────────────

    def open_speaker(self, sample_rate: int) -> None:
        """Open the speaker output stream at sample_rate. Idempotent for a matching
        rate; reopens if a different rate is requested (TTS engines have a fixed
        native rate, e.g. Kokoro's 24kHz, independent of the 16kHz mic/STT rate).
        """
        if self._out_stream is not None:
            if self._out_sample_rate == sample_rate:
                return
            self.close_speaker()
        self._out_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._channels,
            rate=sample_rate,
            output=True,
            output_device_index=self._output_device,
        )
        self._out_sample_rate = sample_rate
        logger.info(
            "Speaker opened — device=%s  rate=%dHz",
            self._output_device or "default",
            sample_rate,
        )

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Blocking write of float32 PCM audio ([-1.0, 1.0], mono) to the speaker."""
        self.open_speaker(sample_rate)
        self._out_stream.write(pcm.astype(np.float32).tobytes())

    def close_speaker(self) -> None:
        if self._out_stream is not None:
            self._out_stream.stop_stream()
            self._out_stream.close()
            self._out_stream = None
            self._out_sample_rate = None
            logger.info("Speaker stream closed.")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release all audio resources."""
        self.close_mic()
        self.close_speaker()
        self._pa.terminate()

    def __enter__(self) -> AudioManager:
        return self

    def __exit__(self, *_) -> None:
        self.close()
