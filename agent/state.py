# AgentState TypedDict — single source of truth for what flows through the graph.
# Keeps state definition separate from graph wiring (graph.py) and node logic (nodes.py).
# Implementation: Part 2
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """State passed between LangGraph nodes for a single conversation turn.

    messages: prior conversation history as Ollama-style {"role", "content"} dicts.
        Populated turn-over-turn by the caller (Part 7 pipeline); generate_node
        appends this turn's user + assistant messages before returning.
    transcript: this turn's user input (from STT).
    retrieved_facts: top-k long-term memory facts for this turn (Part 3 populates
        this via retrieve_memory_node; empty list until then).
    response_chunks: raw token chunks streamed from the LLM this turn, in order.
    """

    messages: list[dict[str, str]]
    transcript: str
    retrieved_facts: list[str]
    response_chunks: list[str]
