"""
faster-whisper STT wrapper.

Accepts a float32 PCM audio array (mono, 16kHz, normalized [-1.0, 1.0]).
Returns a plain transcript string.

Key choices:
  - beam_size=1: greedy decoding — fastest, marginally less accurate than beam search.
  - vad_filter=False: Silero VAD already handled upstream; double-filtering wastes time.
  - without_timestamps=True: we don't need word timings, skipping them saves compute.
  - Logs elapsed time on every call so latency_monitor.py can read it from logs in Part 6.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    # CTranslate2's CUDA backend loads cuBLAS/cuDNN via LoadLibrary calls that only
    # honor the process PATH, not os.add_dll_directory()'s search list. The pip
    # nvidia-*-cu12 packages ship the DLLs but don't register their bin/ dirs, so we
    # prepend them to PATH before ctranslate2 (imported by faster_whisper) loads.
    for _pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        _spec = importlib.util.find_spec(_pkg)
        if _spec and _spec.submodule_search_locations:
            _bin_dir = os.path.join(_spec.submodule_search_locations[0], "bin")
            if os.path.isdir(_bin_dir):
                os.add_dll_directory(_bin_dir)
                os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel


class WhisperSTT:
    """
    Thin wrapper around faster-whisper. Model loaded once at init and reused.
    Thread-safe: WhisperModel.transcribe() is stateless — safe to call from different threads.
    """

    def __init__(self, config: dict) -> None:
        stt_cfg = config["stt"]
        model_size: str = stt_cfg["model_size"]     # "small"
        device: str = stt_cfg["device"]             # "cuda"
        compute_type: str = stt_cfg["compute_type"] # "int8"
        self._language: str = stt_cfg["language"]   # "en"
        self._beam_size: int = stt_cfg["beam_size"] # 1

        logger.info(
            "Loading faster-whisper '%s' on %s (%s)...",
            model_size, device, compute_type,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper loaded in %.2fs", time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 PCM audio array.

        Args:
            audio: float32 ndarray, mono, 16kHz, normalized to [-1.0, 1.0].
                   (Directly from SileroVAD.process_chunk output.)

        Returns:
            Transcript string. Empty string if no speech detected.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        t0 = time.perf_counter()

        # transcribe() returns a generator of Segment objects — we must consume it
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,        # VAD already done by SileroVAD upstream
            without_timestamps=True, # skip word-level timestamps — not needed, saves time
        )

        # Join all segment texts (most short utterances produce just one segment)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("STT [%.0fms]: '%s'", elapsed_ms, transcript[:80])

        return transcript
