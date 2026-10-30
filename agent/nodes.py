# LangGraph node implementations:
#   generate_node       — streaming LLM call via Ollama, emits sentence chunks
#   retrieve_memory_node — embed query, search FAISS, inject top-k facts into context
#   extract_memory_node  — classify if turn contains storable fact, write to store (async)
# Implementation: Part 2 (generate_node) → Part 3 (memory nodes)
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import ollama
from langgraph.config import get_stream_writer

from agent.prompts import build_extraction_prompt, build_system_prompt, EXTRACTION_SYSTEM_PROMPT
from agent.state import AgentState
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

# Keeps references to extract_memory_node's fire-and-forget background tasks so they
# aren't garbage-collected mid-run (asyncio only holds a weak reference otherwise).
_background_tasks: set[asyncio.Task] = set()


async def wait_for_background_tasks(timeout: float = 5.0) -> None:
    """Wait for any in-flight extract_memory_node tasks to finish before the
    process exits. Without this, stopping the server (Ctrl+C, --reload
    restart, closing the web app) shortly after a turn kills the fact write
    before it commits — the fact is silently lost even though extraction and
    storage both work correctly. Bounded by timeout so a genuinely stuck
    task can't hang shutdown forever. Called from ConversationPipeline.close().
    """
    if not _background_tasks:
        return
    logger.info("Waiting for %d in-flight memory-extraction task(s) before shutdown...", len(_background_tasks))
    _, pending = await asyncio.wait(_background_tasks, timeout=timeout)
    if pending:
        logger.warning(
            "%d memory-extraction task(s) still running after %.0fs — proceeding with shutdown anyway",
            len(pending), timeout,
        )


def make_generate_node(config: dict) -> Callable[[AgentState], Awaitable[dict]]:
    """Build generate_node, closing over one reused AsyncClient + the configured
    model/options so they're created once at graph-build time, not per turn.
    """
    llm_cfg = config["llm"]
    client = ollama.AsyncClient(host=llm_cfg["base_url"])
    model: str = llm_cfg["model"]
    keep_alive = llm_cfg.get("keep_alive")
    options = {
        "temperature": llm_cfg["temperature"],
        "num_predict": llm_cfg["max_tokens"],
    }

    async def generate_node(state: AgentState) -> dict:
        """Stream a response from Ollama for state["transcript"].

        Emits each raw token via LangGraph's custom stream writer as it arrives
        (consumed with graph.astream(..., stream_mode="custom")) so the sentence
        chunker (Part 4) can start TTS before generation finishes, instead of
        waiting for the full response.
        """
        writer = get_stream_writer()

        history = state.get("messages", [])
        user_turn = {"role": "user", "content": state["transcript"]}
        request_messages = [
            {"role": "system", "content": build_system_prompt(state.get("retrieved_facts"))},
            *history,
            user_turn,
        ]

        t0 = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []

        stream = await client.chat(
            model=model,
            messages=request_messages,
            stream=True,
            options=options,
            keep_alive=keep_alive,
        )
        async for part in stream:
            token = part["message"]["content"]
            if not token:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                logger.info("LLM first token [%.0fms]", (first_token_at - t0) * 1000)
            chunks.append(token)
            writer(token)

        full_response = "".join(chunks)
        logger.info(
            "LLM generation complete [%.0fms total, %d chars]",
            (time.perf_counter() - t0) * 1000,
            len(full_response),
        )

        assistant_turn = {"role": "assistant", "content": full_response}
        return {
            "messages": history + [user_turn, assistant_turn],
            "response_chunks": chunks,
        }

    return generate_node


def make_retrieve_memory_node(long_term: LongTermMemory) -> Callable[[AgentState], Awaitable[dict]]:
    """Build retrieve_memory_node, closing over the shared LongTermMemory instance."""

    async def retrieve_memory_node(state: AgentState) -> dict:
        """Embed state["transcript"], search long-term memory, inject top-k facts."""
        facts = await asyncio.to_thread(long_term.retrieve_similar, state["transcript"])
        if facts:
            logger.info("Retrieved %d fact(s) from long-term memory", len(facts))
        return {"retrieved_facts": facts}

    return retrieve_memory_node


def make_extract_memory_node(
    config: dict, long_term: LongTermMemory
) -> Callable[[AgentState], Awaitable[dict]]:
    """Build extract_memory_node, closing over a dedicated Ollama client + LongTermMemory.

    Fires the actual classify-and-store work as a background asyncio task and returns
    immediately — the graph never waits on it. Facts from this turn are therefore not
    retrievable until the *next* turn; acceptable since retrieval is most useful for
    facts from past turns/sessions, not the one just generated.
    """
    llm_cfg = config["llm"]
    client = ollama.AsyncClient(host=llm_cfg["base_url"])
    model: str = llm_cfg["model"]
    keep_alive = llm_cfg.get("keep_alive")

    async def _classify_and_store(user_text: str) -> None:
        try:
            response = await client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": build_extraction_prompt(user_text)},
                ],
                stream=False,
                # num_predict raised from 60->120: a single fact fits in 60 tokens, but
                # a turn with two or three distinct facts (one per line, per the
                # multi-fact prompt revision) needs the extra headroom.
                options={"temperature": 0.0, "num_predict": 120},
                keep_alive=keep_alive,
            )
            raw = response["message"]["content"].strip()
            facts = [
                line.strip()
                for line in raw.splitlines()
                if line.strip() and line.strip().upper().rstrip(".") != "NONE"
            ]
            if facts:
                for fact in facts:
                    await asyncio.to_thread(long_term.store_fact, fact)
                    logger.info("Memory extracted: '%s'", fact)
            else:
                logger.debug("Memory extraction: nothing worth storing this turn")
        except Exception:
            logger.exception("extract_memory_node background task failed")

    async def extract_memory_node(state: AgentState) -> dict:
        messages = state["messages"]
        if len(messages) < 2:
            return {}
        user_text = messages[-2]["content"]

        task = asyncio.create_task(_classify_and_store(user_text))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        return {}

    return extract_memory_node
