# VoiceChat

A real-time, fully offline audio-in/audio-out conversational AI assistant — speak to it, it listens, thinks, and talks back. Built as an AI Engineering assignment focused on a hard constraint: **sub-2-second voice-to-voice latency**, with every design decision measured, not assumed.

No cloud APIs are called at inference time. Speech recognition, the LLM, memory retrieval, safety checks, and speech synthesis all run locally.

## Demo

- **Video**: [https://drive.google.com/file/d/14tk-MqsvEQfYThMHLtu1S12pDDP_rA6P/view?usp=sharing](https://drive.google.com/file/d/14tk-MqsvEQfYThMHLtu1S12pDDP_rA6P/view?usp=sharing)
- **Live architecture doc**: [`docs/architecture.md`](docs/architecture.md)

## What it does

- Listens continuously through voice activity detection (no push-to-talk button needed)
- Transcribes speech locally with faster-whisper
- Reasons with a local 3B LLM (Qwen2.5, via Ollama) orchestrated through LangGraph
- Remembers you: a rolling short-term buffer plus a FAISS-backed long-term fact store that recalls things you've told it across turns
- Speaks back with Kokoro TTS, streaming audio out sentence-by-sentence as the LLM is still generating — not waiting for the full response
- Self-checks both what it hears and what it says via NeMo Guardrails, without adding a second full LLM turn
- Never goes silent on a slow response — a fallback manager fires a natural filler phrase if the LLM takes too long, instead of a dead-air pause
- Runs from the terminal (continuous conversation loop) or from a browser (WebSocket mic/speaker UI with a live audio-reactive visualizer)

## Architecture

```
Mic Input
    ↓
[VAD] Silero VAD — detects speech start/end
    ↓
[STT] faster-whisper (small, int8, CUDA) — transcript
    ↓
[Input Guardrail] NeMo self-check — blocks unsafe input before it reaches the LLM
    ↓
[LangGraph Agent]
    ├── retrieve_memory  — embed query → FAISS search → inject top-k facts
    ├── generate          — streaming LLM call via Ollama (Qwen2.5-3B)
    │       ↓ tokens streamed as they're generated
    │   [Sentence Chunker] → [Output Guardrail] → [TTS] → speaker
    │   (bounded producer/consumer queue — generation and speech overlap,
    │    not stacked sequentially)
    └── extract_memory    — async, fire-and-forget — stores durable facts
    ↓
Speaker Output

Racing in parallel: FallbackManager — fires a spoken filler phrase if the
LLM hasn't produced a first token within 800ms, so the assistant never
just goes quiet.
```

Full component-by-component breakdown, every non-obvious implementation decision, and the reasoning behind each one: [`docs/architecture.md`](docs/architecture.md) and [`docs/tradeoffs.md`](docs/tradeoffs.md) (~38 documented design decisions, written as they were made, not reconstructed after the fact).

## Tech stack

| Layer | Tech | Why |
|---|---|---|
| VAD | Silero VAD | More robust than webrtcvad in noisy environments |
| STT | faster-whisper (small, int8, CUDA) | CTranslate2 backend — fast, offline, GPU-accelerated |
| LLM | Qwen2.5-3B-Instruct via Ollama | Small, quantized, fast enough to hit the latency budget locally |
| Orchestration | LangGraph | Clean node/edge mapping to the pipeline's actual data flow |
| Short-term memory | Rolling in-memory deque | Last N turns, no persistence needed |
| Long-term memory | FAISS + all-MiniLM-L6-v2 + SQLite | Offline similarity search over facts the user has shared |
| Guardrails | NeMo Guardrails (input + output self-check only) | Dialogue rails would cost ~200-400ms/turn the budget can't afford |
| TTS | Kokoro-82M | High-quality local voice synthesis |
| Fallback | Custom filler-phrase manager | Proactive — fires on *slowness*, not on error |
| Demo UI | CLI (continuous loop) + FastAPI/WebSocket browser UI | Terminal for development, browser for a clean demo recording |

## Latency

Full breakdown, methodology, and every regression found along the way: [`docs/latency_report.md`](docs/latency_report.md).

| Stage | Warm, isolated | In the full pipeline |
|---|---|---|
| VAD | ~0.2-0.3ms | — |
| STT | 68ms | ~345-380ms |
| Input guardrail | 328ms | ~610ms |
| LLM first token | 587ms | ~520ms |
| Output guardrail + first-sentence TTS | ~444ms | ~800-1070ms (post-optimization) |
| **Time to first audio out** | **~1130ms estimated** | **2229ms median** (down from ~2.6-2.8s pre-optimization) |

**Honest result: this lands at ~2.2s, about 10-15% over the 2s target — not under it.** The gap isn't a bug in any one component; it's measured, root-caused, and documented: five models (Whisper, Silero, Kokoro, the embedder, and Ollama's LLM) share a single 8GB GPU, and consumer GPUs don't isolate concurrent CUDA contexts the way datacenter cards do — under contention, every stage runs 2-6x slower than its isolated benchmark. The mitigations that would close the rest of the gap (skip the output guardrail after the first sentence, move a model to CPU) are identified and written up, not silently applied, because each is a real accuracy/safety tradeoff, not a free win. See the [Analysis section of the latency report](docs/latency_report.md#analysis) for the full reasoning.

## Getting started

**Prerequisites**: Python 3.12+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and (strongly recommended) an NVIDIA GPU — this project runs meaningfully slower on CPU-only.

```bash
# 1. Install dependencies
uv sync

# 2. Pull the LLM
ollama pull qwen2.5:3b

# 3. Download all other models (STT, VAD, embeddings, TTS) — one-time, offline after this
uv run python scripts/download_models.py --all

# 4. Make sure Ollama is running
ollama serve
```

### Run it

**Terminal** (continuous conversation loop — just start talking):
```bash
uv run python main.py
```

**Browser** (click to start a session, VAD auto-detects your turns from there):
```bash
uv run python demo/web/app.py
# open http://127.0.0.1:8000
```

### Configure it

Every model name, path, and threshold lives in [`config.yaml`](config.yaml) — nothing is hardcoded in pipeline code. Notable knobs:
- `fallback.llm_first_token_timeout_ms` — how long before a filler phrase fires (default 800ms)
- `llm.keep_alive` — how long Ollama keeps the model warm in VRAM between turns (default 30m)
- `pipeline.sentence_queue_maxsize` — backpressure bound between LLM generation and TTS playback
- `guardrails.check_input` / `check_output` — toggle safety checks independently

### Run the tests

```bash
uv run pytest tests/ --ignore=tests/test_pipeline_e2e.py   # fast suite, no live audio
uv run pytest tests/                                        # full suite, incl. real end-to-end audio (~47 min)
```

## Project structure

```
agent/          LangGraph nodes, graph wiring, prompts, state
audio/          VAD, STT, TTS, mic/speaker I/O
guardrails/     NeMo Guardrails manager + input/output rail configs
memory/         Short-term buffer + FAISS/SQLite long-term store
pipeline/       Turn orchestration, fallback manager, latency monitor
demo/           CLI entry point + FastAPI/WebSocket web UI
scripts/        Model downloader, latency benchmark runner
tests/          Component tests + full end-to-end latency tests
docs/           Architecture, tradeoffs, latency report, AI-usage disclosure
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — full pipeline breakdown, component by component
- [`docs/tradeoffs.md`](docs/tradeoffs.md) — every non-obvious design decision and the reasoning behind it
- [`docs/latency_report.md`](docs/latency_report.md) — full benchmark methodology, numbers, and root-cause analysis
- [`docs/ai_usage.md`](docs/ai_usage.md) — AI-usage disclosure

## Submission links

- GitHub Repository: [https://github.com/infiniteaniverse/voicechat-live](https://github.com/infiniteaniverse/voicechat-live)
- Demo video: [https://drive.google.com/file/d/14tk-MqsvEQfYThMHLtu1S12pDDP_rA6P/view?usp=sharing](https://drive.google.com/file/d/14tk-MqsvEQfYThMHLtu1S12pDDP_rA6P/view?usp=sharing)

---

