# VoiceChat — Implementation Plan

## Pre-work (Before Any Code)
- [x] Repo created and pushed
- [ ] Hardware check: run `nvidia-smi` → determines model size and `device` config
- [ ] First commit already done (Sprint 0 scaffold)

---

## Part 1 — Audio Foundation: VAD + STT
**Files**: `audio/vad.py`, `audio/stt.py`, `audio/audio_manager.py` (mic capture only), `scripts/benchmark.py` (STT section), `scripts/download_models.py`

**What happens**:
- `download_models.py` pulls faster-whisper base and Silero VAD weights to disk (one-time)
- `audio_manager.py` opens a PyAudio mic stream and yields 512-sample chunks in a loop
- `vad.py` wraps Silero VAD — takes a chunk, returns True/False for "speech detected", manages the 700ms silence window to know when the user stopped talking, emits the complete audio buffer
- `stt.py` wraps faster-whisper — takes the audio buffer from VAD, returns a transcript string
- `benchmark.py` runs STT on a fixed 5-second test clip and prints median latency over 20 runs

**Why this order**: VAD and STT are independent of the LLM and TTS. Isolating them first lets you confirm your latency budget for the audio input side before touching the more complex parts. If STT alone is already at 900ms, you know to switch to a smaller model or enable CUDA before wiring everything together.

**Commit**: `feat: audio layer — VAD (Silero) + STT (faster-whisper) with latency benchmarks`

---

## Part 2 — LLM Core: LangGraph + Ollama Streaming
**Files**: `agent/state.py`, `agent/graph.py`, `agent/nodes.py` (generate_node only), `agent/prompts.py`, `scripts/download_models.py` (Ollama pull section)

**What happens**:
- `state.py` defines `AgentState` TypedDict: `messages`, `transcript`, `retrieved_facts`, `response_chunks`
- `prompts.py` has the system prompt (conversational assistant, concise answers for TTS) and placeholder for extraction prompt
- `nodes.py` implements `generate_node` — takes state, calls Ollama streaming API, yields tokens
- `graph.py` wires a minimal graph: just `generate_node` as the only node for now (memory nodes added in Part 3)
- Ollama pre-warm on startup (dummy prompt to avoid cold-start on first real query)

**Key thing about Ollama streaming**: The Python client gives you an iterator of token chunks. The `generate_node` must yield these chunks (not collect them all), because the sentence chunker in Part 4 needs to receive tokens as they arrive — not in one big string at the end.

A standalone test (hardcoded string → graph → stream output to console) confirms this works before touching audio.

**Commit**: `feat: LLM core — LangGraph agent with streaming Ollama generation`

---

## Part 3 — Memory System
**Files**: `memory/embeddings.py`, `memory/short_term.py`, `memory/long_term.py`, `agent/nodes.py` (add retrieve_memory_node + extract_memory_node), `agent/graph.py` (update edges), `tests/test_memory.py`

**What happens**:
- `embeddings.py` loads `all-MiniLM-L6-v2` once as a singleton, exposes `embed(text) -> np.array`
- `short_term.py` is a `collections.deque` (maxlen from config, default 10 turns) — in-memory only, no persistence, clears on session end.
- `long_term.py` manages FAISS index (`data/memory/faiss_index`) + SQLite (`data/memory/facts.sqlite`). Functions: `store_fact(text)`, `retrieve_similar(query, top_k=3)`, `delete_fact(id)`
- `retrieve_memory_node` — embeds transcript, queries FAISS, injects top-k into `state.retrieved_facts`
- `extract_memory_node` — async/parallel, cheap LLM call: "Is there a fact worth remembering? Extract it." Writes to FAISS+SQLite if yes
- `agent/graph.py` updated: `retrieve_memory → generate → (async) extract_memory`

**Design decision — short-term memory is in-memory deque for latency reasons**: Short-term memory is read on every single turn to build the LLM context window. SQLite adds ~1-5ms disk I/O per read (even with WAL mode). Postgres adds a TCP roundtrip even on localhost. A Python deque read is microseconds — literally 1000x faster. At 10 reads per conversation turn inside a 2-second budget, that overhead compounds. This is a deliberate latency optimization, not just simpler code. The trade-off accepted: no cross-session short-term context (that's what long-term FAISS is for). If multi-user support is added later, this is the point where you'd swap to a session-keyed store.

**The async pattern**: `extract_memory_node` runs in a background task (`asyncio.create_task`). Facts from the current turn aren't retrievable until the next turn — acceptable, because retrieval is most useful for facts from past sessions.

**Commit**: `feat: dual memory system — short-term buffer + long-term FAISS fact store`

---

## Part 4 — TTS: Sentence Chunking + Audio Playback
**Files**: `audio/tts.py`, `audio/audio_manager.py` (add speaker playback), `scripts/download_models.py` (Kokoro/Piper section), `tests/test_tts.py`

**What happens**:
- `audio_manager.py` gains a `play_audio(pcm_bytes)` method using PyAudio output stream
- `tts.py` has two parts:
  - **Sentence chunker**: buffers streaming tokens, emits complete sentences on `.`, `?`, `!` boundaries
  - **TTS engine wrapper**: sentence string → Kokoro (or Piper) → PCM audio bytes
- Playback loop: each sentence → TTS → play immediately, overlapped with LLM still generating

**The overlapping pattern**:
```
asyncio tasks running concurrently:
  Task A: LangGraph streams tokens → sentence chunker
  Task B: For each complete sentence → TTS → play audio
```
Task B starts as soon as the first sentence boundary hits. This overlap is what makes the pipeline feel fast.

**Commit**: `feat: TTS layer — sentence-chunk streaming with overlapped LLM+TTS playback`

---

## Part 5 — Guardrails
**Files**: `guardrails/config.yml`, `guardrails/rails/input_rails.co`, `guardrails/rails/output_rails.co`, `tests/test_guardrails.py`

**What happens**:
- Write Colang rail definitions for input (blocks jailbreak/harmful) and output (safety net per sentence chunk)
- Guardrails wired at pipeline level, not inside LangGraph graph — keeps graph clean, avoids adding latency inside LLM generation path
- Test with adversarial prompt list written before implementing

**Commit**: `feat: guardrails — NeMo input/output rails integrated`

---

## Part 6 — Fallback System + Latency Monitor
**Files**: `pipeline/latency_monitor.py`, `pipeline/fallback.py`

**What happens**:
- `latency_monitor.py`: context manager / timer. Records: VAD end, STT end, LLM first token, TTS first chunk, TTS complete. Logs per-stage breakdown if `log_latency: true`
- `fallback.py`: `FallbackManager` class. At `start_turn()`, starts 800ms timer. If LLM hasn't emitted first token by then → pick random filler phrase from config → fire through TTS immediately. Fires on *slowness*, not on error — that's the key UX insight.

**Commit**: `feat: fallback system — filler phrase queue with latency-threshold trigger`

---

## Part 7 — Full Pipeline Integration
**Files**: `pipeline/pipeline.py`, `main.py`

**What happens**:
- `pipeline.py` runs the full conversation loop:
  1. Start mic stream
  2. VAD → wait for speech end → emit audio buffer
  3. STT → transcript string
  4. Input guardrail on transcript
  5. Start fallback timer
  6. LangGraph agent (streaming) → sentence chunker → TTS → speaker (overlapped)
  7. extract_memory in background
  8. Repeat
- `main.py` loads config, initializes all components, pre-warms LLM, hands off to pipeline

Expect most debugging here — timing issues, async edge cases, audio glitches surface at integration time.

**Commit**: `feat: full pipeline integration — end-to-end voice conversation working`

---

## Part 8 — Demo Interface
**Files**: `demo/cli_demo.py` (primary), `demo/web/` (stretch)

**CLI demo**:
- Push-to-talk: Enter to start, Enter to stop
- Latency breakdown printed after each turn: `[STT: 280ms | LLM: 620ms | TTS: 140ms | Total: 1040ms]`

**Web demo (stretch — cut this if time is tight)**:
- FastAPI + WebSocket: browser streams mic in, server streams TTS audio back
- Much better demo video, but costs ~4 hours

**Commit**: `feat: demo interface — CLI push-to-talk + optional web UI`

---

## Part 9 — Tests + Benchmarks
**Files**: All `tests/` files, `scripts/benchmark.py` (complete)

- Fill in all test files, run `pytest`
- Run `scripts/benchmark.py` → get per-component latency numbers
- Fill in `docs/latency_report.md` with real numbers from your hardware

**Commit**: `test: unit tests + e2e latency benchmarks`

---

## Part 10 — Docs + Submission
**Files**: `docs/architecture.md`, `docs/tradeoffs.md`, `docs/ai_usage.md`, `docs/latency_report.md`, `docs/submission/links.md`, `README.md`

- Architecture diagram (Excalidraw / draw.io → export PNG to `docs/`)
- Finalize tradeoffs, AI usage disclosure
- Write README (setup, how to run)
- Record demo video (2-3 min screen record)
- Upload to Google Drive, fill in submission links, convert to PDF

**Commit**: `docs: architecture, tradeoffs, AI-usage disclosure, submission prep`

---

## Timeline

| Day | Morning | Afternoon |
|---|---|---|
| Day 0 (now) | Hardware check, pre-work | — |
| Day 1 | Part 1: VAD + STT | Part 2: LangGraph + Ollama |
| Day 2 | Part 3: Memory | Part 4: TTS + sentence chunking |
| Day 3 | Part 5: Guardrails + Part 6: Fallback | Part 7: Full pipeline integration |
| Day 4 | Part 8: Demo + Part 9: Tests | Part 10: Docs + record video |

**If something slips**: Cut web UI in Part 8 first — CLI demo is fully submission-worthy.
