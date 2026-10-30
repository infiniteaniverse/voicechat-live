# Per-component latency benchmarks used for the submission latency report.
# Tests each stage in isolation: VAD, retrieve_memory, LLM first-token.
# STT and TTS already have their own dedicated latency tests (tests/test_stt.py,
# tests/test_tts.py); guardrail latency lives in tests/test_guardrails.py. This
# file covers the remaining per-component gaps flagged in docs/latency_report.md.
# Implementation: Part 9
from __future__ import annotations

import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _ollama_reachable(base_url: str) -> bool:
    try:
        httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


# ── VAD ──────────────────────────────────────────────────────────────────────

def test_vad_window_latency_within_budget(config):
    """Median Silero VAD inference latency for one 512-sample window must be
    well under the ~50ms per-turn budget slice (CLAUDE.md's original latency
    budget table). Measured ~0.2-0.3ms warm on an RTX 4060 — budgeted
    generously here to tolerate slower hardware while still catching a real
    regression (e.g. VAD accidentally falling back to a much slower path).
    """
    from audio.vad import SileroVAD

    vad = SileroVAD(config)
    window = np.zeros(512, dtype=np.float32)

    vad.process_chunk(window)  # warm-up
    vad.reset()

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        vad.process_chunk(window)
        latencies.append((time.perf_counter() - t0) * 1000)
        vad.reset()  # isolate single-window inference cost, not state-machine bookkeeping

    median = sorted(latencies)[len(latencies) // 2]
    assert median < 50, f"VAD median latency {median:.1f}ms exceeds 50ms budget"


# ── retrieve_memory ──────────────────────────────────────────────────────────

def test_retrieve_memory_latency_within_budget(config, tmp_path):
    """retrieve_similar() runs on the critical path (before generate_node),
    so it must stay negligible against the 2s budget. Measured ~25ms in
    Part 3's own testing.
    """
    from memory.embeddings import Embedder
    from memory.long_term import LongTermMemory

    mem_config = dict(config)
    mem_config["memory"] = dict(config["memory"])
    mem_config["memory"]["long_term"] = dict(config["memory"]["long_term"])
    mem_config["memory"]["long_term"]["store_path"] = str(tmp_path / "faiss_index")
    mem_config["memory"]["long_term"]["facts_db_path"] = str(tmp_path / "facts.sqlite")

    embedder = Embedder(mem_config)
    long_term = LongTermMemory(mem_config, embedder)
    for fact in [
        "The user's name is Jay.",
        "The user works as a machine learning engineer.",
        "The user is allergic to peanuts.",
    ]:
        long_term.store_fact(fact)

    long_term.retrieve_similar("What do I do for work?")  # warm-up

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        long_term.retrieve_similar("What do I do for work?")
        latencies.append((time.perf_counter() - t0) * 1000)

    median = sorted(latencies)[len(latencies) // 2]
    assert median < 100, f"retrieve_memory median latency {median:.1f}ms exceeds 100ms budget"


# ── LLM first token ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _skip_llm_tests_if_ollama_unreachable(config):
    if not _ollama_reachable(config["llm"]["base_url"]):
        pytest.skip(
            f"Ollama not reachable at {config['llm']['base_url']} — "
            "run `ollama serve` to test LLM first-token latency."
        )


@pytest.mark.asyncio
async def test_llm_first_token_latency_within_budget(config, tmp_path):
    """Warm LLM first-token latency must stay comfortably under the 800ms
    fallback threshold — if this regresses, the fallback would start firing
    on nearly every turn instead of only genuinely slow ones (see
    docs/tradeoffs.md decision 19 for why that specific number matters).
    Measured ~500-650ms warm throughout Parts 2-7's testing.
    """
    graph_config = dict(config)
    graph_config["memory"] = dict(config["memory"])
    graph_config["memory"]["long_term"] = dict(config["memory"]["long_term"])
    graph_config["memory"]["long_term"]["store_path"] = str(tmp_path / "faiss_index")
    graph_config["memory"]["long_term"]["facts_db_path"] = str(tmp_path / "facts.sqlite")

    from agent.graph import build_graph

    graph = build_graph(graph_config)

    async def _first_token_latency_ms(prompt: str) -> float:
        state = {"messages": [], "transcript": prompt, "retrieved_facts": [], "response_chunks": []}
        t0 = time.perf_counter()
        async for _ in graph.astream(state, stream_mode="custom"):
            return (time.perf_counter() - t0) * 1000
        raise AssertionError("graph produced no tokens at all")

    await _first_token_latency_ms("Hello")  # warm-up (loads the model into VRAM)

    latencies = [await _first_token_latency_ms("Say hi in one short sentence.") for _ in range(3)]

    median = sorted(latencies)[len(latencies) // 2]
    assert median < 800, (
        f"LLM first-token median latency {median:.0f}ms exceeds the 800ms fallback threshold — "
        f"the fallback would fire on nearly every turn. Check Ollama's keep_alive/model state."
    )
