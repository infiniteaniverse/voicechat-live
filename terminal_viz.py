"""Live terminal dashboard for VoiceChat: animates every pipeline step.

Local add-on (not part of the upstream repo), same spirit as run_live.py. It
changes no pipeline code: it wraps the already-built pipeline's methods at
runtime (VAD, STT, guardrails, memory, LLM stream, sentence chunker, TTS,
playback, fallback, latency monitor) and renders what they do with `rich`:

    * the 10-step pipeline as boxes, with data packets moving along the arrows
    * live graphs: mic level, VAD speech probability, end-of-turn timer, GPU
      utilisation / VRAM, the generated speech waveform with a playback cursor
    * the conversation as it happens: transcript, recalled memories, LLM tokens
      streaming in, each sentence's guardrail / TTS / playback status
    * latency: a live "first audio out" clock against the 2 s budget, a stacked
      per-stage bar, and a per-turn history chart
    * a timestamped timeline of every step, plus a startup checklist

Entry point: run_viz.py.
"""
from __future__ import annotations

import io
import logging
import math
import re
import subprocess
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

FPS = 12
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
BARS = "▁▂▃▄▅▆▇█"
BUDGET_MS = 2000.0

# Palette
C_IDLE = "grey39"
C_ACTIVE_A, C_ACTIVE_B = "#ffd166", "#ff9f1c"
C_DONE = "#06d6a0"
C_BLOCK = "#ef476f"
C_DIM = "grey50"
C_FRAME = "#3a4a6b"

STAGES = [
    ("mic", "01 MIC"),
    ("vad", "02 VAD"),
    ("stt", "03 STT"),
    ("in_guard", "04 INPUT GUARD"),
    ("memory", "05 MEMORY"),
    ("llm", "06 LLM"),
    ("chunker", "07 CHUNKER"),
    ("out_guard", "08 OUTPUT GUARD"),
    ("tts", "09 TTS"),
    ("speaker", "10 SPEAKER"),
]

KIND_STYLE = {
    "SYS": "grey62", "VAD": "#ef476f", "STT": "#4cc9f0", "GUARD": "#f72585",
    "MEM": "#b388ff", "LLM": "#ffd166", "CHUNK": "#90be6d", "TTS": "#06d6a0",
    "PLAY": "#43aa8b", "FALLBK": "#f9c74f", "CLOCK": "#8ecae6", "TURN": "bold white",
    "WARN": "#ff6b6b",
}

# (monitor stage key, short label, colour) for the "time to first audio" bar
LATENCY_SEGMENTS = [
    ("stt", "STT", "#4cc9f0"),
    ("input_guardrail", "In-guard", "#f72585"),
    ("llm_first_token", "LLM 1st", "#ffd166"),
    ("tts_first_chunk", "Out+TTS", "#06d6a0"),
]

BOOT_ITEMS = [
    ("imports", "Import libraries (torch, LangGraph, NeMo...)", None),
    ("vad", "Silero VAD model", ("Silero VAD ready",)),
    ("stt", "faster-whisper STT (small, int8, CUDA)", ("Whisper loaded",)),
    ("tts", "Kokoro-82M text-to-speech", ("Kokoro pipeline ready",)),
    ("embed", "MiniLM embedding model", ("Embedding model loaded",)),
    ("faiss", "FAISS + SQLite long-term memory", ("Loading FAISS index", "creating new one")),
    ("guard", "NeMo Guardrails (input + output rails)", ("GuardrailsManager loaded",)),
    ("prewarm", "Pre-warm the LLM in Ollama (cold start is slow)", ("pre-warm complete",)),
]


# ── Small drawing helpers ──────────────────────────────────────────────────────

def T(*parts, no_wrap: bool = True) -> Text:
    """Build a Text from plain strings and (string, style) tuples."""
    t = Text(no_wrap=no_wrap, overflow="ellipsis")
    for p in parts:
        if isinstance(p, str):
            t.append(p)
        else:
            t.append(p[0], style=p[1])
    return t


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "--"
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


def level_norm(peak: float) -> float:
    """Peak amplitude (0..1) -> 0..1 on a -60..0 dB scale."""
    if peak <= 0:
        return 0.0
    db = 20 * math.log10(max(peak, 1e-4))
    return min(1.0, max(0.0, (db + 60) / 60))


def spark(values, width: int, lo: float = 0.0, hi: float = 1.0, color_fn=None) -> Text:
    """One-line bar graph of the last `width` values, newest at the right."""
    vals = list(values)[-width:]
    t = Text(" " * (width - len(vals)), no_wrap=True)
    for v in vals:
        n = 0.0 if hi <= lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))
        t.append(BARS[min(7, int(n * 8))], style=color_fn(n) if color_fn else "#4cc9f0")
    return t


def level_color(n: float) -> str:
    return C_DONE if n < 0.55 else ("#ffd166" if n < 0.8 else C_BLOCK)


def bar(frac: float, width: int, color: str) -> Text:
    frac = min(1.0, max(0.0, frac))
    filled = int(round(frac * width))
    t = Text(no_wrap=True)
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="grey23")
    return t


def envelope(pcm: np.ndarray, n: int = 120) -> np.ndarray:
    """Peak envelope of `pcm` in n buckets, normalised to 0..1."""
    x = np.abs(np.asarray(pcm, dtype=np.float32))
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    edges = np.linspace(0, x.size, n + 1).astype(int)
    env = np.array([x[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])], dtype=np.float32)
    m = float(env.max())
    return env / m if m > 0 else env


def waveform_line(env: np.ndarray | None, width: int, progress: float | None) -> Text:
    """Envelope drawn as one row of blocks; the part already 'played' is bright."""
    if env is None or len(env) == 0:
        return Text("▁" * width, style="grey23", no_wrap=True)
    xs = np.interp(np.linspace(0, len(env) - 1, width), np.arange(len(env)), env)
    cut = -1 if progress is None else int(progress * width)
    t = Text(no_wrap=True)
    for i, v in enumerate(xs):
        style = "#4cc9f0" if progress is None or i <= cut else "#1b4f63"
        if progress is not None and i == cut:
            style = "bold white"
        t.append(BARS[min(7, int(v * 8))], style=style)
    return t


def vchart(values: list[float], height: int, vmax: float, thr: float, width: int) -> list[Text]:
    """Vertical bar chart (one column group per value) with a dashed threshold line."""
    col, gap = 3, 1
    ncols = max(1, (width - 7) // (col + gap))
    vals = values[-ncols:]
    lines: list[Text] = []
    for r in range(height):
        top = (height - r) / height * vmax
        bot = (height - r - 1) / height * vmax
        t = Text(no_wrap=True)
        has_thr = bot < thr <= top
        for v in vals:
            style = C_DONE if v <= thr else C_BLOCK
            if v >= top:
                t.append("█" * col, style=style)
            elif v > bot:
                t.append(BARS[min(7, int((v - bot) / (top - bot) * 8))] * col, style=style)
            elif has_thr:
                t.append("┄" * col, style="#8a7a2a")
            else:
                t.append(" " * col)
            t.append(" ")
        if has_thr:
            t.append(f" {thr / 1000:.1f}s", style="#ffd166")
        lines.append(t)
    return lines


# ── State records ──────────────────────────────────────────────────────────────

@dataclass
class Stage:
    state: str = "idle"        # idle | active | done | blocked
    detail: str = ""
    started: float = 0.0
    inflight: int = 0
    count: int = 0
    blocked: int = 0
    last_ms: float | None = None
    total_ms: float = 0.0


@dataclass
class Sentence:
    text: str
    kind: str = "llm"          # llm | filler | refusal
    guard: str | None = None   # None | run | ok | block
    guard_ms: float | None = None
    synth: str | None = None   # None | run | done
    tts_ms: float | None = None
    audio_s: float | None = None
    play: str | None = None    # None | scheduled | playing | done
    t_play: float = 0.0


class _LogBridge(logging.Handler):
    def __init__(self, dash: "Dashboard") -> None:
        super().__init__(level=logging.INFO)
        self.dash = dash

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.dash._on_log(record.name, record.levelno, record.getMessage())
        except Exception:
            pass


class _GraphProxy:
    """Stands in for the compiled LangGraph so the token stream can be observed."""

    def __init__(self, graph, dash: "Dashboard") -> None:
        self._graph = graph
        self._dash = dash

    def __getattr__(self, name):
        return getattr(self._graph, name)

    async def astream(self, *args, **kwargs):
        self._dash._llm_begin()
        try:
            async for token in self._graph.astream(*args, **kwargs):
                self._dash._llm_token(token)
                yield token
        finally:
            self._dash._llm_end()


# ── The dashboard ──────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, mode: str = "web", url: str = "", out_dir: str | Path = "data/viz_logs",
                 console: Console | None = None, use_screen: bool = True) -> None:
        self.mode = mode
        self.url = url
        self.console = console or Console()
        self.use_screen = use_screen and self.console.is_terminal
        self.t_start = time.monotonic()
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._live: Live | None = None
        self._stop = threading.Event()
        self._last_frame = Text("starting...")
        self._last_tick = 0.0

        self.pipeline = None
        self.ready = False
        self.stages: dict[str, Stage] = {k: Stage() for k, _ in STAGES}
        self.events: deque = deque(maxlen=500)
        self.all_events: list = []
        self.turns: list[dict] = []
        self.turn_no = 0
        self.turn_active = False
        self.turn_started_once = False

        # boot checklist: key -> (finished_at monotonic, seconds the step took)
        self.boot_done: dict[str, tuple[float, float]] = {}
        self._boot_last = self.t_start

        # config-derived (set in attach)
        self.fallback_ms = 800
        self.vad_threshold = 0.5
        self.min_silence_ms = 700
        self.filler_phrases: list[str] = []
        self.sample_rate = 16000

        # live signals
        self.mic_hist: deque = deque(maxlen=400)
        self.vad_hist: deque = deque(maxlen=400)
        self.gpu_hist: deque = deque(maxlen=400)
        self.vram_hist: deque = deque(maxlen=400)
        self.gpu_text = "n/a"
        self.vram_text = ""
        self._gpu_ok = True
        self._mic_peak = 0.0
        self._mic_last = 0.0
        self._vad_prob = 0.0
        self._vad_in_speech = False
        self._vad_silence_ms = 0.0
        self.speech_started_at = 0.0

        # per-turn content
        self.transcript = ""
        self.transcript_ms: float | None = None
        self.recalled: list[str] = []
        self.stored: list[str] = []
        self.tokens = ""
        self.n_tokens = 0
        self.token_times: list[float] = []
        self.llm_t0: float | None = None
        self.llm_first_ms: float | None = None
        self.sents: list[Sentence] = []
        self.fb_start: float | None = None
        self.fb_fired = False
        self.wave: np.ndarray | None = None
        self.wave_play: tuple[np.ndarray, float, float] | None = None  # env, t_start, dur
        self.play_until = 0.0
        self._turn_has_transcript = False
        self.block_in_turn = False
        self.refusal_ms: float | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def install_logging(self) -> None:
        """Route all logging to a file + this dashboard (nothing hits the terminal)."""
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        fh = logging.FileHandler(self.out_dir / f"viz_{self.stamp}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
        root.addHandler(fh)
        root.addHandler(_LogBridge(self))
        root.setLevel(logging.INFO)

    def start(self, live: bool = True) -> None:
        threading.Thread(target=self._poll_gpu, daemon=True, name="viz-gpu").start()
        if not live:
            return
        self._live = Live(
            get_renderable=self._frame,
            console=self.console,
            refresh_per_second=FPS,
            screen=self.use_screen,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()

    def start_frame_recorder(self, interval: float) -> Path:
        """Headless capture (no TTY): sample the signals and save a text frame every `interval` s."""
        folder = self.out_dir / f"frames_{self.stamp}"
        folder.mkdir(parents=True, exist_ok=True)

        def loop() -> None:
            n, last = 0, 0.0
            while not self._stop.is_set():
                self._tick()
                if time.monotonic() - last >= interval:
                    last, n = time.monotonic(), n + 1
                    try:
                        (folder / f"f{n:04d}.txt").write_text(
                            self.render_static().export_text(styles=False), encoding="utf-8")
                    except Exception:
                        pass
                self._stop.wait(1 / FPS)

        threading.Thread(target=loop, daemon=True, name="viz-frames").start()
        return folder

    def stop(self) -> None:
        self._stop.set()
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

    def boot_mark(self, key: str) -> None:
        if key in self.boot_done:
            return
        now = time.monotonic()
        self.boot_done[key] = (now, now - self._boot_last)
        self._boot_last = now
        label = next((l for k, l, _ in BOOT_ITEMS if k == key), key)
        self.event("SYS", f"ready: {label}  ({self.boot_done[key][1]:.1f}s)")

    # ── events ─────────────────────────────────────────────────────────────────

    def event(self, kind: str, text: str) -> None:
        rec = (time.time(), kind, text)
        self.events.append(rec)
        if len(self.all_events) < 20000:
            self.all_events.append(rec)

    def _on_log(self, name: str, level: int, msg: str) -> None:
        for key, _label, pats in BOOT_ITEMS:
            if pats and key not in self.boot_done and any(p in msg for p in pats):
                self.boot_mark(key)
        if level >= logging.WARNING:
            self.event("WARN", f"{name.split('.')[-1]}: {msg[:160]}")
        elif "Discarding stalling first sentence" in msg:
            self.event("CHUNK", "stalling first sentence dropped: " + msg.split(":", 1)[-1].strip()[:90])
        elif msg.startswith("Stored fact"):
            self.event("MEM", "stored: " + msg.split(":", 1)[-1].strip()[:110])
            self.stored.append(msg.split(":", 1)[-1].strip().strip("'"))
        elif msg.startswith("Duplicate of fact"):
            self.event("MEM", "duplicate fact, not stored again")
        elif msg.startswith("LLM first token"):
            m = re.search(r"\[(\d+)ms\]", msg)
            if m:
                self.llm_first_ms = float(m.group(1))

    # ── stage helpers ──────────────────────────────────────────────────────────

    def _begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            s = self.stages[key]
            s.inflight += 1
            s.state = "active"
            s.detail = detail
            if s.inflight == 1:
                s.started = time.monotonic()

    def _end(self, key: str, ms: float | None = None, detail: str | None = None, blocked: bool = False) -> None:
        with self._lock:
            s = self.stages[key]
            s.inflight = max(0, s.inflight - 1)
            s.count += 1
            if ms is not None:
                s.last_ms = ms
                s.total_ms += ms
            if blocked:
                s.blocked += 1
            if detail is not None:
                s.detail = detail
            if s.inflight == 0:
                s.state = "blocked" if s.blocked else "done"

    def _reset_turn_view(self) -> None:
        with self._lock:
            for k, _ in STAGES:
                if k not in ("mic", "vad"):
                    self.stages[k] = Stage()
            self.sents = []
            self.tokens = ""
            self.n_tokens = 0
            self.token_times = []
            self.llm_t0 = None
            self.llm_first_ms = None
            self.transcript = ""
            self.transcript_ms = None
            self.recalled = []
            self.fb_start = None
            self.fb_fired = False
            self.block_in_turn = False
            self.refusal_ms = None
            self.stored = []
            self.wave_play = None

    # ── attach: wrap the live pipeline ─────────────────────────────────────────

    def attach(self, pipeline) -> None:
        cfg = pipeline._config
        self.pipeline = pipeline
        self.fallback_ms = cfg["fallback"]["llm_first_token_timeout_ms"]
        self.filler_phrases = list(cfg["fallback"]["filler_phrases"])
        self.vad_threshold = cfg["vad"]["threshold"]
        self.min_silence_ms = cfg["vad"]["min_silence_duration_ms"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        self._wrap_audio_in(pipeline)
        self._wrap_vad(pipeline)
        self._wrap_stt(pipeline)
        self._wrap_guardrails(pipeline)
        self._wrap_memory_and_chunker()
        pipeline.graph = _GraphProxy(pipeline.graph, self)
        self._wrap_tts_and_playback(pipeline)
        self._wrap_fallback(pipeline)
        self._wrap_monitor(pipeline)
        self._wrap_run_turn(pipeline)

        for key, _l, _p in BOOT_ITEMS:
            self.boot_mark(key)
        self.ready = True
        self.event("SYS", "pipeline ready - waiting for you to speak"
                   + (f" (open {self.url} and click the mic)" if self.mode == "web" and self.url else ""))

    def _wrap_audio_in(self, p) -> None:
        io = p.audio_manager
        orig = io.read_chunk

        def read_chunk():
            chunk = orig()
            if len(chunk):
                self._mic_peak = max(self._mic_peak, float(np.max(np.abs(chunk))))
            self._mic_last = time.monotonic()
            return chunk

        io.read_chunk = read_chunk

    def _wrap_vad(self, p) -> None:
        vad = p.vad
        orig_score = vad._score
        orig_chunk = vad.process_chunk

        def score(window):
            prob = orig_score(window)
            self._vad_prob = prob
            return prob

        in_utterance = False

        def process_chunk(chunk):
            nonlocal in_utterance
            was = vad._in_speech
            out = orig_chunk(chunk)
            now_in = vad._in_speech
            if now_in and not was:
                in_utterance = True
                self._on_speech_start()
            if out is not None:
                if not in_utterance:  # burst started and ended inside one call
                    self._on_speech_start()
                in_utterance = False
                self._on_speech_end(len(out) / self.sample_rate)
            elif was and not now_in:
                in_utterance = False
                self._on_speech_discarded()
            self._vad_in_speech = vad._in_speech
            self._vad_silence_ms = vad._silence_samples * 1000.0 / self.sample_rate
            return out

        vad._score = score
        vad.process_chunk = process_chunk

    def _on_speech_start(self) -> None:
        self._reset_turn_view()
        self.speech_started_at = time.monotonic()
        s = self.stages["vad"]
        s.state, s.detail, s.started = "active", "hearing you", time.monotonic()
        self.event("VAD", f"speech started (P={self._vad_prob:.2f} >= {self.vad_threshold})")

    def _on_speech_end(self, seconds: float) -> None:
        self.stages["vad"].state = "done"
        self.stages["vad"].detail = f"{seconds:.1f}s utterance"
        self.stages["vad"].last_ms = seconds * 1000
        self.stages["mic"].state = "done"
        self.stages["mic"].detail = f"{seconds:.1f}s captured"
        self.event("VAD", f"end of speech after {self.min_silence_ms}ms of silence - {seconds:.2f}s of audio captured")

    def _on_speech_discarded(self) -> None:
        self.stages["vad"].state = "active"
        self.stages["vad"].detail = "too short, ignored"
        self.event("VAD", "short noise burst discarded")

    def _wrap_stt(self, p) -> None:
        stt = p.stt
        orig = stt.transcribe

        def transcribe(audio):
            self._begin("stt", f"{len(audio) / self.sample_rate:.1f}s of audio")
            t0 = time.perf_counter()
            text = ""
            try:
                text = orig(audio)
                return text
            finally:
                ms = (time.perf_counter() - t0) * 1000
                self.transcript = text or ""
                self.transcript_ms = ms
                self._turn_has_transcript = bool(text)
                self._end("stt", ms, detail=f"{len(text.split())} words" if text else "nothing heard")
                self.event("STT", f'"{text}"  ({fmt_ms(ms)})' if text else f"heard nothing ({fmt_ms(ms)})")

        stt.transcribe = transcribe

    def _wrap_guardrails(self, p) -> None:
        g = p.guardrails
        orig_in, orig_out = g.check_input, g.check_output

        async def check_input(text):
            self._begin("in_guard", "self_check_input")
            t0 = time.perf_counter()
            try:
                res = await orig_in(text)
            except BaseException:
                self._end("in_guard", (time.perf_counter() - t0) * 1000, detail="error (fail-open)")
                raise
            ms = (time.perf_counter() - t0) * 1000
            self.block_in_turn = not res.allowed
            self._end("in_guard", ms, detail="allowed" if res.allowed else "blocked", blocked=not res.allowed)
            self.event("GUARD", f"input check: {'ALLOWED' if res.allowed else 'BLOCKED -> refusal will be spoken'} ({fmt_ms(ms)})")
            return res

        async def check_output(text):
            sent = self._sentence_for(text, create_kind="llm")
            sent.guard = "run"
            self._begin("out_guard", f"{sum(1 for s in self.sents if s.guard == 'run')} checking")
            t0 = time.perf_counter()
            try:
                res = await orig_out(text)
            except BaseException:
                sent.guard = "ok"
                self._end("out_guard", (time.perf_counter() - t0) * 1000)
                raise
            ms = (time.perf_counter() - t0) * 1000
            sent.guard = "ok" if res.allowed else "block"
            sent.guard_ms = ms
            self._end("out_guard", ms, detail=f"{self.stages['out_guard'].count + 1}x checked",
                      blocked=not res.allowed)
            self.event("GUARD", f"output check: {'allowed' if res.allowed else 'BLOCKED'} ({fmt_ms(ms)})  '{text[:60]}'")
            return res

        g.check_input = check_input
        g.check_output = check_output

    def _wrap_memory_and_chunker(self) -> None:
        from audio.tts import SentenceChunker
        from memory.long_term import LongTermMemory

        if getattr(LongTermMemory, "_viz_patched", False):
            return
        orig_ret = LongTermMemory.retrieve_similar

        def retrieve_similar(lt, query, top_k=None):
            self._begin("memory", "embed + FAISS search")
            t0 = time.perf_counter()
            try:
                facts = orig_ret(lt, query, top_k)
            except BaseException:
                self._end("memory", (time.perf_counter() - t0) * 1000, detail="error")
                raise
            ms = (time.perf_counter() - t0) * 1000
            self.recalled = list(facts)
            self._end("memory", ms, detail=f"{len(facts)} fact(s) recalled")
            self.event("MEM", f"searched long-term memory ({fmt_ms(ms)}): "
                       + (f"{len(facts)} fact(s) recalled" if facts else "nothing relevant"))
            for f in facts:
                self.event("MEM", f"  recalled: {f[:110]}")
            if self.stages["llm"].state == "idle":
                self._begin("llm", "thinking...")
            return facts

        orig_feed, orig_flush = SentenceChunker.feed, SentenceChunker.flush

        def feed(ch, token):
            out = orig_feed(ch, token)
            for s in out:
                self._on_sentence(s)
            return out

        def flush(ch):
            out = orig_flush(ch)
            if out:
                self._on_sentence(out)
            return out

        LongTermMemory.retrieve_similar = retrieve_similar
        SentenceChunker.feed, SentenceChunker.flush = feed, flush
        LongTermMemory._viz_patched = True

    # LLM stream (called by _GraphProxy)
    def _llm_begin(self) -> None:
        self.tokens, self.n_tokens, self.token_times = "", 0, []
        self.llm_t0 = time.perf_counter()
        self.event("LLM", "graph started: retrieve_memory -> generate -> extract_memory")

    def _llm_token(self, token: str) -> None:
        now = time.perf_counter()
        if self.n_tokens == 0:
            ms = (now - (self.llm_t0 or now)) * 1000
            self.llm_first_ms = ms
            self.event("LLM", f"first token after {fmt_ms(ms)}")
            if self.stages["llm"].state != "active":
                self._begin("llm", "")
            self._begin("chunker", "splitting on . ! ?")
        self.n_tokens += 1
        self.token_times.append(now)
        self.tokens += token

    def _llm_end(self) -> None:
        if self.llm_t0 is None:
            return
        total = (time.perf_counter() - self.llm_t0) * 1000
        if self.stages["llm"].state == "idle":
            return
        rate = self._tok_rate()
        self._end("llm", total, detail=f"{self.n_tokens} tok - {rate:.0f}/s")
        if self.stages["chunker"].inflight:
            self._end("chunker", None, detail=f"{len(self.sents)} sentence(s)")
        self.event("LLM", f"generation done: {self.n_tokens} tokens in {fmt_ms(total)}")

    def _tok_rate(self) -> float:
        if len(self.token_times) < 2:
            return 0.0
        span = self.token_times[-1] - self.token_times[0]
        return (len(self.token_times) - 1) / span if span > 0 else 0.0

    def _on_sentence(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.sents.append(Sentence(text=text))
        self.event("CHUNK", f"sentence {len(self.sents)} complete: '{text[:90]}'")

    def _sentence_for(self, text: str, create_kind: str) -> Sentence:
        """The record for `text`: prefer a chunker sentence that hasn't been
        checked/synthesized yet, else one still in flight, else a new record."""
        key = text.strip()
        for s in self.sents:
            if s.text == key and s.synth is None and s.guard is None:
                return s
        for s in self.sents:
            if s.text == key and s.guard in (None, "run") and s.synth in (None, "run"):
                return s
        s = Sentence(text=key, kind=create_kind)
        self.sents.append(s)
        return s

    def _wrap_tts_and_playback(self, p) -> None:
        tts = p.tts
        orig_syn = tts.synthesize
        io = p.audio_manager
        orig_play = io.play_audio

        def synthesize(text):
            # A sentence the chunker produced already has a record. Anything else
            # reaching TTS is pipeline-generated: a fallback filler or a refusal line.
            kind = "filler" if text in self.filler_phrases else "refusal"
            sent = self._sentence_for(text, create_kind=kind)
            sent.synth = "run"
            self._begin("tts", f"'{text[:24]}'")
            t0 = time.perf_counter()
            try:
                pcm = orig_syn(text)
            except BaseException:
                sent.synth = "done"
                self._end("tts", (time.perf_counter() - t0) * 1000, detail="error")
                raise
            ms = (time.perf_counter() - t0) * 1000
            audio_s = len(pcm) / tts.sample_rate
            sent.synth, sent.tts_ms, sent.audio_s = "done", ms, audio_s
            self.wave = envelope(pcm)
            mon = self.pipeline.monitor
            if self.block_in_turn and self.refusal_ms is None and mon._t0 is not None:
                self.refusal_ms = (time.perf_counter() - mon._t0) * 1000
            self._end("tts", ms, detail=f"{self.stages['tts'].count + 1}x - last {fmt_ms(ms)}")
            self.event("TTS", f"synthesized {audio_s:.1f}s of speech in {fmt_ms(ms)}: '{text[:60]}'")
            return pcm

        def play_audio(pcm, sample_rate):
            dur = len(pcm) / float(sample_rate)
            now = time.monotonic()
            start = max(now, self.play_until)
            self.play_until = start + dur
            nxt = next((s for s in self.sents if s.synth == "done" and s.play is None and s.guard != "block"), None)
            if nxt is not None:
                nxt.play, nxt.t_play = "scheduled", start
            self.wave_play = (envelope(pcm), start, dur)
            self.stages["speaker"].count += 1
            self.event("PLAY", f"{dur:.1f}s of audio sent to the speaker"
                       + (f" ('{nxt.text[:50]}')" if nxt else ""))
            return orig_play(pcm, sample_rate)

        tts.synthesize = synthesize
        io.play_audio = play_audio

    def _wrap_fallback(self, p) -> None:
        fb = p.fallback
        orig_watch, orig_speak = fb.watch, fb._speak

        async def watch(ev):
            self.fb_start = time.monotonic()
            self.fb_fired = False
            try:
                fired = await orig_watch(ev)
            finally:
                self.fb_start = None
            self.fb_fired = bool(fired)
            if not fired:
                self.event("FALLBK", f"LLM answered inside {self.fallback_ms}ms - no filler needed")
            return fired

        async def speak(phrase):
            self.event("FALLBK", f"LLM slower than {self.fallback_ms}ms -> speaking filler: '{phrase}'")
            return await orig_speak(phrase)

        fb.watch, fb._speak = watch, speak

    def _wrap_monitor(self, p) -> None:
        m = p.monitor
        orig_start, orig_mark = m.start_turn, m.mark

        def start_turn():
            orig_start()
            self.turn_active = True
            self.event("CLOCK", "latency clock started (VAD confirmed end of speech)")

        def mark(stage):
            ms = orig_mark(stage)
            if stage == "tts_first_chunk":
                verdict = "within" if ms <= BUDGET_MS else "OVER"
                self.event("CLOCK", f"FIRST AUDIO OUT at {ms:.0f}ms - {verdict} the {BUDGET_MS:.0f}ms budget")
            elif stage != "tts_complete":
                self.event("CLOCK", f"{stage} done at {ms:.0f}ms")
            return ms

        m.start_turn, m.mark = start_turn, mark

    def _wrap_run_turn(self, p) -> None:
        orig = p.run_turn

        async def run_turn():
            self._on_turn_wait()
            try:
                return await orig()
            finally:
                self._on_turn_end()

        p.run_turn = run_turn

    def _on_turn_wait(self) -> None:
        self.turn_started_once = True
        self._turn_has_transcript = False
        self.turn_active = False
        self.stages["mic"] = Stage(state="active", detail="listening", started=time.monotonic())
        self.stages["vad"] = Stage(state="active", detail="waiting for speech", started=time.monotonic())
        self._vad_in_speech = False
        self.event("TURN", f"--- turn {self.turn_no + 1}: listening ---")

    def _on_turn_end(self) -> None:
        self.turn_active = False
        if not self._turn_has_transcript:
            return
        self.turn_no += 1
        m = self.pipeline.monitor
        durs = m.stage_durations()
        spoken = " ".join(s.text for s in self.sents if s.guard != "block" and s.kind != "filler")
        rec = {
            "n": self.turn_no,
            "you": self.transcript,
            "reply": spoken,
            "first_audio_ms": m.elapsed_ms("tts_first_chunk"),
            "durations": durs,
            "blocked_in": self.block_in_turn,
            "blocked_out": sum(1 for s in self.sents if s.guard == "block"),
            "filler": self.fb_fired,
            "recalled": len(self.recalled),
            "tokens": self.n_tokens,
        }
        self.turns.append(rec)
        fa = rec["first_audio_ms"]
        self.event("TURN", f"turn {self.turn_no} complete - first audio "
                   + (f"{fa:.0f}ms" if fa is not None else "n/a (input refused)"))

    # ── background sampler ─────────────────────────────────────────────────────

    def _poll_gpu(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4, creationflags=flags,
                ).stdout.strip().splitlines()[0]
                util, used, total = [float(x) for x in out.split(",")]
                self.gpu_hist.append(util / 100)
                self.vram_hist.append(used / total)
                self.gpu_text = f"{util:.0f}%"
                self.vram_text = f"{used / 1024:.1f}/{total / 1024:.1f}GB"
            except Exception:
                self._gpu_ok = False
                self.gpu_text = "n/a"
            self._stop.wait(1.0)

    def _tick(self) -> None:
        """Sample live signals, advance playback state. Several consumers can call
        this (terminal Live + browser panes); it only advances once per frame period."""
        now = time.monotonic()
        if now - self._last_tick < 0.8 / FPS:
            return
        self._last_tick = now
        fresh = now - self._mic_last < 0.4
        peak, self._mic_peak = self._mic_peak, 0.0
        self.mic_hist.append(level_norm(peak) if fresh else 0.0)
        self.vad_hist.append(self._vad_prob if fresh else 0.0)
        for s in self.sents:
            if s.play == "scheduled" and now >= s.t_play:
                s.play = "playing"
            if s.play == "playing" and s.audio_s is not None and now >= s.t_play + s.audio_s:
                s.play = "done"
        sp = self.stages["speaker"]
        if any(s.play in ("scheduled", "playing") for s in self.sents):
            sp.state = "active"
        elif sp.count:
            sp.state = "done"

    # ── rendering ──────────────────────────────────────────────────────────────

    def _frame(self):
        try:
            self._tick()
            w, h = self.console.size
            self._last_frame = self.render(w, h)
        except Exception as exc:  # never let a drawing bug kill the display thread
            self._last_frame = Text(f"render error: {exc!r}", style="red")
        return self._last_frame

    def _phase(self) -> tuple[str, str]:
        st = self.stages
        if not self.ready:
            return "BOOTING", C_ACTIVE_A
        if st["speaker"].state == "active" or st["tts"].state == "active":
            return "SPEAKING", "#4cc9f0"
        if any(st[k].state == "active" for k in ("stt", "in_guard", "memory", "llm", "chunker", "out_guard")):
            return "THINKING", "#b388ff"
        if self._vad_in_speech:
            return "HEARING YOU", C_BLOCK
        if st["mic"].state == "active":
            return "LISTENING", C_DONE
        return "IDLE", C_DIM

    def render(self, width: int, height: int):
        width = max(width, 80)
        flow_h = 10
        rest = max(14, height - 3 - flow_h)
        mid_h = max(8, int(rest * 0.42))
        bot_h = max(6, rest - mid_h)
        lw = int(width * 0.46)
        rw = width - lw

        def row(left, right):
            g = Table.grid(padding=0)
            g.add_column(width=lw)
            g.add_column(width=rw)
            g.add_row(left, right)
            return g

        if self.ready:
            mid = row(
                self._panel(self._signals(lw - 4, mid_h - 2), "LIVE SIGNALS", mid_h),
                self._panel(self._conversation(rw - 4, mid_h - 2), "CONVERSATION", mid_h),
            )
        else:
            mid = row(
                self._panel(self._boot(lw - 4, mid_h - 2), "STARTING UP", mid_h),
                self._panel(self._signals(rw - 4, mid_h - 2, boot=True), "SYSTEM LOAD", mid_h),
            )
        bot = row(
            self._panel(self._latency(lw - 4, bot_h - 2), "LATENCY  (budget 2.0s to first audio)", bot_h),
            self._panel(self._timeline(rw - 4, bot_h - 2), "EVERY STEP, AS IT HAPPENS", bot_h),
        )
        return Group(
            self._header(width),
            self._panel(self._flow(width - 4), "PIPELINE", flow_h),
            mid,
            bot,
        )

    @staticmethod
    def _panel(lines, title: str, height: int, border: str = C_FRAME) -> Panel:
        return Panel(Group(*lines) if isinstance(lines, list) else lines,
                     title=Text(f" {title} ", style="bold white"), title_align="left", height=height,
                     box=box.ROUNDED, border_style=border, padding=(0, 1))

    def _header(self, width: int) -> Panel:
        phase, color = self._phase()
        frame = int(time.monotonic() * FPS)
        spin = SPIN[frame % len(SPIN)] if phase not in ("IDLE",) else " "
        up = int(time.monotonic() - self.t_start)
        g = Table.grid(expand=True)
        g.add_column(justify="left", ratio=1)
        g.add_column(justify="center", ratio=1)
        g.add_column(justify="right", ratio=1)
        g.add_row(
            T(("VOICECHAT", "bold #4cc9f0"), (f"  live pipeline  ({self.mode} mode)", C_DIM)),
            T((f" {spin} {phase} ", f"bold black on {color}")),
            T((f"turns {self.turn_no}   up {up // 60:02d}:{up % 60:02d}", C_DIM)),
        )
        return Panel(g, box=box.ROUNDED, border_style=C_FRAME, height=3, padding=(0, 1))

    # -- pipeline flow --

    def _flow(self, inner: int) -> list[Text]:
        n = 5
        conn = 5 if inner >= 100 else 3
        node_w = max(11, (inner - conn * (n - 1)) // n)
        frame = int(time.monotonic() * FPS)
        lines: list[Text] = []
        for row_items in (STAGES[:5], STAGES[5:]):
            for ln in range(4):
                line = Text(no_wrap=True)
                for i, (key, label) in enumerate(row_items):
                    line.append_text(self._node_line(key, label, ln, node_w, frame))
                    if i < len(row_items) - 1:
                        line.append_text(self._connector(key, row_items[i + 1][0], ln, conn, frame))
                lines.append(line)
        return lines

    def _node_style(self, s: Stage, frame: int) -> str:
        if s.state == "active":
            return "bold " + (C_ACTIVE_A if (frame // 3) % 2 == 0 else C_ACTIVE_B)
        return {"done": C_DONE, "blocked": "bold " + C_BLOCK}.get(s.state, C_IDLE)

    def _node_line(self, key: str, label: str, ln: int, w: int, frame: int) -> Text:
        s = self.stages[key]
        style = self._node_style(s, frame)
        inner = w - 2
        t = Text(no_wrap=True)
        if ln == 0:
            if w < 21:
                label = label.replace("INPUT GUARD", "IN GUARD").replace("OUTPUT GUARD", "OUT GUARD")
            head = f"─ {label} "[:inner]
            t.append("╭" + head + "─" * max(0, inner - len(head)) + "╮", style=style)
        elif ln == 3:
            t.append("╰" + "─" * inner + "╯", style=style)
        else:
            if ln == 1:
                txt, tstyle = self._node_status(key, s, frame)
            else:
                txt, tstyle = self._node_detail(key, s, frame)
            txt = txt[:inner].center(inner)
            t.append("│", style=style)
            t.append(txt, style=tstyle)
            t.append("│", style=style)
        return t

    def _node_status(self, key: str, s: Stage, frame: int) -> tuple[str, str]:
        spin = SPIN[frame % len(SPIN)]
        if s.state == "idle":
            return "idle", C_IDLE
        if s.state == "active":
            if key == "speaker":
                return f"{spin} playing", "bold #4cc9f0"
            el = (time.monotonic() - s.started) * 1000
            return f"{spin} {fmt_ms(el)}", "bold " + C_ACTIVE_A
        if s.state == "blocked":
            return "✗ BLOCKED", "bold " + C_BLOCK
        ms = s.last_ms if key not in ("mic", "speaker") else None
        return ("✓ " + fmt_ms(ms)) if ms is not None else "✓ done", C_DONE

    def _node_detail(self, key: str, s: Stage, frame: int) -> tuple[str, str]:
        if key == "speaker" and s.state == "active":
            now = time.monotonic()
            left = max(0.0, self.play_until - now)
            return f"▶ {left:.1f}s left", "#4cc9f0"
        if key == "llm" and s.state == "active" and self.n_tokens:
            return f"{self.n_tokens} tok {self._tok_rate():.0f}/s", C_ACTIVE_A
        if key == "llm" and s.state == "active" and self.fb_start:
            left = self.fallback_ms - (time.monotonic() - self.fb_start) * 1000
            return (f"filler in {left:.0f}ms" if left > 0 else "filler fired"), "#f9c74f"
        if key == "vad" and self._vad_in_speech and self.stages["vad"].state == "active":
            return f"P={self._vad_prob:.2f}", C_BLOCK
        return s.detail, C_DIM if s.state == "idle" else "grey70"

    def _connector(self, a: str, b: str, ln: int, w: int, frame: int) -> Text:
        if ln != 1:
            return Text(" " * w)
        sa, sb = self.stages[a], self.stages[b]
        if sb.state == "active" and sa.state in ("done", "active", "blocked"):
            pos = (frame // 2) % (w - 1)
            t = Text(no_wrap=True)
            for i in range(w):
                if i == pos:
                    t.append("●", style="bold #ffd166")
                elif i == pos - 1:
                    t.append("•", style="#a58a2a")
                else:
                    t.append("─", style="#4a4a2a")
            return t
        if sa.state in ("done", "blocked") and sb.state in ("done", "blocked"):
            return T(("━" * (w - 1) + "▶", C_DONE))
        if sa.state == "done":
            return T(("─" * (w - 1) + "▶", "#1d6b55"))
        return T(("─" * w, "grey23"))

    # -- boot / signals --

    def _boot(self, iw: int, rows: int) -> list[Text]:
        frame = int(time.monotonic() * FPS)
        done = len(self.boot_done)
        total = len(BOOT_ITEMS)
        lines = [T(("loading models, this takes a while the first time  ", C_DIM)), bar(done / total, max(10, iw - 12), C_DONE)]
        lines[1].append(f" {done}/{total}", style="bold")
        first_pending = next((k for k, _l, _p in BOOT_ITEMS if k not in self.boot_done), None)
        for key, label, _p in BOOT_ITEMS:
            if key in self.boot_done:
                lines.append(T(("✔ ", C_DONE), (label, "grey85"), (f"  {self.boot_done[key][1]:.1f}s", C_DIM)))
            elif key == first_pending:
                el = time.monotonic() - self._boot_last
                lines.append(T((f"{SPIN[frame % len(SPIN)]} ", C_ACTIVE_A), (label, "bold " + C_ACTIVE_A), (f"  {el:.0f}s", C_DIM)))
            else:
                lines.append(T(("○ ", C_IDLE), (label, C_IDLE)))
        return lines[:rows]

    def _signals(self, iw: int, rows: int, boot: bool = False) -> list[Text]:
        lab, val = 10, 15
        gw = max(10, iw - lab - val)
        now = time.monotonic()
        fresh = now - self._mic_last < 0.4
        lines: list[Text] = []

        def line(label, graph, value="", vstyle=C_DIM):
            t = T((f"{label:<{lab}}", "grey70"))
            t.append_text(graph)
            t.append(" " + value, style=vstyle)
            return t

        if not boot:
            last = self.mic_hist[-1] if self.mic_hist else 0.0
            lines.append(line("MIC", spark(self.mic_hist, gw, color_fn=level_color),
                              f"{last * 60 - 60:.0f} dB" if fresh else "not listening"))
            vcol = lambda n: C_BLOCK if n >= self.vad_threshold else "#3d5a80"
            lines.append(line("VAD P()", spark(self.vad_hist, gw, color_fn=vcol),
                              f"{self._vad_prob:.2f}  {'SPEECH' if self._vad_in_speech else 'quiet'}",
                              C_BLOCK if self._vad_in_speech else C_DIM))
            if self._vad_in_speech and self._vad_silence_ms > 0:
                lines.append(line("TURN END", bar(self._vad_silence_ms / self.min_silence_ms, gw, "#ffd166"),
                                  f"{self._vad_silence_ms:.0f}/{self.min_silence_ms}ms", "#ffd166"))
            elif self._vad_in_speech:
                lines.append(line("TURN END", bar(0, gw, "#ffd166"), "talking...", C_BLOCK))
            else:
                lines.append(line("TURN END", bar(0, gw, "#ffd166"), "silence timer idle"))
        lines.append(line("GPU", spark(self.gpu_hist, gw, color_fn=level_color), self.gpu_text, "grey85"))
        lines.append(line("VRAM", spark(self.vram_hist, gw, color_fn=lambda n: "#b388ff"), self.vram_text, "grey85"))
        if not boot:
            prog = None
            if self.wave_play is not None:
                _env, t0, dur = self.wave_play
                if dur > 0 and t0 <= now < t0 + dur:
                    prog = (now - t0) / dur
            env = self.wave_play[0] if self.wave_play is not None else self.wave
            lines.append(line("AUDIO OUT", waveform_line(env, gw, prog),
                              "playing" if prog is not None else ("generated" if env is not None else "-"),
                              "#4cc9f0" if prog is not None else C_DIM))
            if self.token_times:
                lines.append(line("LLM tok/s", self._token_spark(gw), f"{self._tok_rate():.0f}/s", "#ffd166"))
        return lines[:rows]

    def _token_spark(self, width: int) -> Text:
        """Tokens-per-bucket over the generation (a burst chart of the stream)."""
        times = self.token_times
        t0, t1 = times[0], max(times[-1], times[0] + 1e-3)
        buckets = np.zeros(width)
        idx = np.minimum(((np.array(times) - t0) / (t1 - t0) * (width - 1)).astype(int), width - 1)
        for i in idx:
            buckets[i] += 1
        return spark(list(buckets), width, 0, max(1.0, buckets.max()), color_fn=lambda n: "#ffd166")

    # -- conversation --

    def _conversation(self, iw: int, rows: int) -> list[Text]:
        frame = int(time.monotonic() * FPS)
        lab = 8
        tw = max(10, iw - lab)
        out: list[Text] = []
        if not self.turn_started_once:
            msg = (f"Open {self.url} in your browser and click the mic to talk."
                   if self.mode == "web" else "Start talking - the mic is open." if self.mode == "local"
                   else "Demo is starting...")
            return [T((msg, "grey70")), T(("Everything you say and everything it does will be animated here.", C_DIM))]

        def add(label, body_lines, style, lstyle="grey70"):
            for i, b in enumerate(body_lines or [""]):
                out.append(T((f"{label if i == 0 else '':<{lab}}", lstyle), (b, style)))

        if self.transcript:
            extra = f"   [STT {fmt_ms(self.transcript_ms)}]" if self.transcript_ms else ""
            add("YOU", textwrap.wrap(f'"{self.transcript}"', tw - len(extra))[:2], "bold #4cc9f0")
            out[-1].append(extra, style=C_DIM)
        elif self.stages["stt"].state == "active":
            add("YOU", [f"{SPIN[frame % len(SPIN)]} transcribing..."], C_ACTIVE_A)
        elif self._vad_in_speech:
            add("YOU", ["(speaking...)"], C_BLOCK)
        else:
            add("YOU", ["-"], C_DIM)

        if self.block_in_turn:
            add("GUARD", ["input BLOCKED by the safety rail - a refusal is spoken instead of calling the LLM"], "bold " + C_BLOCK, C_BLOCK)
        if self.recalled:
            add("RECALL", [f"• {f}"[:tw] for f in self.recalled[:2]], "#b388ff", "#b388ff")
        elif self.stages["memory"].state == "done":
            add("RECALL", ["no relevant stored facts"], C_DIM)

        if self.tokens or self.stages["llm"].state == "active":
            cursor = "▍" if (frame // 4) % 2 == 0 and self.stages["llm"].state == "active" else " "
            body = textwrap.wrap(self.tokens.strip() + cursor, tw) if self.tokens else [f"{SPIN[frame % len(SPIN)]} thinking..."]
            add("LLM", body[-2:], "#90be6d", "#90be6d")
        if self.stored:
            add("MEMORY+", [f"stored: {self.stored[-1]}"[:tw]], "#b388ff", "#b388ff")

        if self.sents:
            out.append(T(("SENTENCES", "bold grey70"), ("  guard + synth run in parallel, then playback", C_DIM)))
            room = max(1, rows - len(out))
            for i, s in enumerate(self.sents[-room:], start=max(0, len(self.sents) - room) + 1):
                out.append(self._sentence_line(i, s, iw, frame))
        return out[:rows]

    def _sentence_line(self, i: int, s: Sentence, iw: int, frame: int) -> Text:
        spin = SPIN[frame % len(SPIN)]
        now = time.monotonic()
        if s.guard == "block":
            glyph, gs, note = "✗", C_BLOCK, "BLOCKED"
        elif s.play == "done":
            glyph, gs, note = "✓", C_DONE, f"spoken {s.audio_s:.1f}s"
        elif s.play == "playing":
            frac = min(1.0, (now - s.t_play) / max(s.audio_s or 1, 1e-3))
            glyph, gs, note = "▶", "#4cc9f0", "playing " + "▰" * int(frac * 6) + "▱" * (6 - int(frac * 6))
        elif s.play == "scheduled":
            glyph, gs, note = "●", "#4cc9f0", "queued for speaker"
        elif s.guard == "run" or s.synth == "run":
            glyph, gs = spin, C_ACTIVE_A
            note = "guard " + ("…" if s.guard == "run" else "ok") + " | tts " + ("…" if s.synth == "run" else "ok")
        elif s.synth == "done":
            glyph, gs, note = "●", C_DONE, "ready"
        else:
            glyph, gs, note = "·", C_DIM, "queued"
        tag = {"filler": "[filler] ", "refusal": "[refusal] "}.get(s.kind, "")
        stats = ""
        if s.guard_ms is not None:
            stats += f" guard {fmt_ms(s.guard_ms)}"
        if s.tts_ms is not None:
            stats += f" tts {fmt_ms(s.tts_ms)}"
        right = f"{note}{stats}"
        if iw - len(right) - 7 < 40:
            right = note
        left_w = max(8, iw - len(right) - 7)
        body = (tag + s.text)[:left_w].ljust(left_w)
        return T((f" {glyph} ", "bold " + gs), (f"{i}. ", C_DIM), (body, "grey85" if s.guard != "block" else C_BLOCK), (" " + right, C_DIM))

    # -- latency --

    def _latency(self, iw: int, rows: int) -> list[Text]:
        lines: list[Text] = []
        m = self.pipeline.monitor if self.pipeline else None
        durs = m.stage_durations() if m else {}
        first = m.elapsed_ms("tts_first_chunk") if m else None
        running = self.turn_active and first is None and m is not None and m._t0 is not None
        refused = self.block_in_turn and self.refusal_ms is not None
        if refused:
            first, running = None, False
        clock = self.refusal_ms if refused else ((time.perf_counter() - m._t0) * 1000 if running else first)
        if clock is None:
            lines.append(T(("FIRST AUDIO OUT  ", "grey70"), ("waiting for a turn", C_DIM)))
        else:
            ok = clock <= BUDGET_MS
            col = C_DONE if ok else C_BLOCK
            tag = ("running..." if running else
                   "refusal audio ready (LLM never called)" if refused else
                   "within budget" if ok else "OVER budget")
            lines.append(T(("REFUSAL AUDIO    " if refused else "FIRST AUDIO OUT  ", "grey70"), (f"{clock:,.0f} ms", "bold " + col), (f"   {tag}", col)))
            lines.append(T(("clock ", C_DIM)))
            lines[-1].append_text(bar(clock / (BUDGET_MS * 1.25), max(10, iw - 24), col))
            lines[-1].append(f" {clock / BUDGET_MS * 100:.0f}% of 2s", style=C_DIM)
        segs = [(lbl, durs[k], c) for k, lbl, c in LATENCY_SEGMENTS if k in durs]
        if refused and clock is not None:
            segs.append(("Refusal TTS", max(1.0, clock - sum(d for _l, d, _c in segs)), "#ef476f"))
        if segs:
            total = sum(d for _l, d, _c in segs)
            vmax = max(BUDGET_MS * 1.3, total * 1.05)
            bw = max(20, iw - 2)
            t = Text(no_wrap=True)
            used = 0
            for lbl, d, c in segs:
                n = max(1, int(round(d / vmax * bw)))
                t.append("█" * n, style=c)
                used += n
            t.append("░" * max(0, bw - used), style="grey23")
            lines.append(t)
            pos = min(bw - 1, int(BUDGET_MS / vmax * bw))
            lines.append(T((" " * pos + "▲ 2.0s budget", "#ffd166")))
            leg = Text(no_wrap=True)
            for lbl, d, c in segs:
                leg.append("■ ", style=c)
                leg.append(f"{lbl} {fmt_ms(d)}  ", style="grey85")
            lines.append(leg)
        hist = [t["first_audio_ms"] for t in self.turns if t["first_audio_ms"] is not None]
        if hist and rows - len(lines) >= 4:
            ch = max(3, min(6, rows - len(lines) - 2))
            lines.append(T((f"history: first-audio ms per turn (median {np.median(hist):.0f}ms)", C_DIM)))
            lines.extend(vchart(hist, ch, max(BUDGET_MS * 1.3, max(hist) * 1.05), BUDGET_MS, iw))
        elif not hist and clock is None:
            lines.append(T(("history chart appears after the first answered turn", C_DIM)))
        return lines[:rows]

    # -- timeline --

    def _timeline(self, iw: int, rows: int) -> list[Text]:
        evs = list(self.events)[-rows:]
        out = []
        for ts, kind, text in evs:
            clock = datetime.fromtimestamp(ts).strftime("%H:%M:%S.") + f"{int(ts * 1000) % 1000:03d}"
            out.append(T((clock + " ", C_DIM), (f"{kind:<6} ", KIND_STYLE.get(kind, "white")), (text[: max(10, iw - 18)], "grey85")))
        return out

    # ── export / summary ───────────────────────────────────────────────────────

    def render_static(self, width: int = 150, height: int = 46) -> Console:
        rec = Console(record=True, width=width, height=height, file=io.StringIO(),
                      force_terminal=True, color_system="truecolor")
        rec.print(self.render(width, height))
        return rec

    def render_ansi(self, cols: int, rows: int) -> str:
        """One dashboard frame as raw ANSI for xterm.js (cursor home, exactly `rows` lines)."""
        buf = io.StringIO()
        con = Console(file=buf, force_terminal=True, color_system="truecolor", width=cols, height=rows,
                      legacy_windows=False)
        con.print(self.render(cols, rows), end="")
        lines = buf.getvalue().split("\n")[:rows]
        return "\x1b[?25l\x1b[H" + "\r\n".join(lines) + "\x1b[0m"

    def export(self, tag: str = "final") -> list[Path]:
        paths: list[Path] = []
        try:
            rec = self.render_static()
            svg = self.out_dir / f"viz_{self.stamp}_{tag}.svg"
            rec.save_svg(str(svg), title="VoiceChat live pipeline", clear=False)
            paths.append(svg)
            txt = self.out_dir / f"viz_{self.stamp}_{tag}.txt"
            txt.write_text(rec.export_text(styles=False), encoding="utf-8")
            paths.append(txt)
        except Exception:
            logging.getLogger(__name__).exception("frame export failed")
        return paths

    def export_timeline(self) -> Path:
        path = self.out_dir / f"viz_{self.stamp}_timeline.txt"
        with open(path, "w", encoding="utf-8") as f:
            for ts, kind, text in self.all_events:
                f.write(f"{datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]}  {kind:<6} {text}\n")
        return path

    def print_summary(self) -> None:
        c = self.console
        c.print()
        c.print(self.render(c.width, min(c.height, 46)))
        c.print()
        if not self.turns:
            c.print("[grey62]No completed turns this session.[/]")
            return
        tbl = Table(title="Session summary - latency per turn", box=box.SIMPLE_HEAVY, header_style="bold #4cc9f0", expand=False)
        for col in ("#", "First audio", "STT", "In-guard", "LLM 1st", "Out+TTS", "Notes"):
            tbl.add_column(col, no_wrap=True)
        for t in self.turns:
            d = t["durations"]
            fa = t["first_audio_ms"]
            notes = []
            if t["blocked_in"]:
                notes.append("input refused")
            if t["blocked_out"]:
                notes.append(f"{t['blocked_out']} sentence blocked")
            if t["filler"]:
                notes.append("filler spoken")
            if t["recalled"]:
                notes.append(f"{t['recalled']} fact recalled")
            col = "green" if fa is not None and fa <= BUDGET_MS else "red"
            tbl.add_row(
                str(t["n"]),
                f"[{col}]{fmt_ms(fa)}[/]" if fa is not None else "n/a",
                fmt_ms(d.get("stt")), fmt_ms(d.get("input_guardrail")), fmt_ms(d.get("llm_first_token")),
                fmt_ms(d.get("tts_first_chunk")), ", ".join(notes) or "-",
            )
        c.print(tbl)
        for t in self.turns:
            c.print(f"[bold]Turn {t['n']}[/]  [#4cc9f0]You:[/] {t['you']}")
            c.print(f"        [#90be6d]Assistant:[/] {t['reply'] or '-'}")
        hist = [t["first_audio_ms"] for t in self.turns if t["first_audio_ms"] is not None]
        if hist:
            c.print(f"First-audio latency over {len(hist)} turn(s): median [bold]{np.median(hist):.0f}ms[/], "
                    f"best {min(hist):.0f}ms, worst {max(hist):.0f}ms (budget {BUDGET_MS:.0f}ms)")


# ── Demo audio source (no microphone needed) ───────────────────────────────────

class DemoAudioIO:
    """AudioIO that 'speaks' scripted questions into the real pipeline and 'plays'
    the replies in real time. Lets the dashboard be shown without a microphone."""

    CHUNK = 512

    def __init__(self, sample_rate: int = 16000, lead_in_s: float = 3.0, tail_s: float = 1.4, sound=None) -> None:
        self._rate = sample_rate
        self._lead = lead_in_s
        self._tail = tail_s
        self._sound = sound
        self._gen = None

    def prepare(self, audio_16k: np.ndarray) -> None:
        self._gen = self._chunks(audio_16k)

    def _chunks(self, audio: np.ndarray):
        sig = np.concatenate([
            np.zeros(int(self._lead * self._rate), np.float32),
            audio.astype(np.float32),
            np.zeros(int(self._tail * self._rate), np.float32),
        ])
        for i in range(0, len(sig), self.CHUNK):
            c = sig[i:i + self.CHUNK]
            if len(c) < self.CHUNK:
                c = np.pad(c, (0, self.CHUNK - len(c)))
            time.sleep(self.CHUNK / self._rate)
            yield c
        while True:
            time.sleep(self.CHUNK / self._rate)
            yield np.zeros(self.CHUNK, np.float32)

    def open_mic(self) -> None:
        pass

    def read_chunk(self) -> np.ndarray:
        return next(self._gen)

    def play_audio(self, pcm: np.ndarray, sample_rate: int) -> None:
        if self._sound is not None:
            self._sound.play_audio(pcm, sample_rate)
        else:
            time.sleep(len(pcm) / float(sample_rate))

    def close(self) -> None:
        pass
