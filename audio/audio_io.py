# AudioIO — the minimal protocol ConversationPipeline needs from an audio backend.
# audio_manager.py's AudioManager (real PyAudio hardware) satisfies this structurally
# already; demo/web/ws_handler.py's WebSocketAudioIO (Part 8) is the second
# implementation, sourcing/sinking audio over a WebSocket instead of local hardware.
# Implementation: Part 8
from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioIO(Protocol):
    def open_mic(self) -> None: ...

    def read_chunk(self) -> np.ndarray: ...

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None: ...

    def close(self) -> None: ...
