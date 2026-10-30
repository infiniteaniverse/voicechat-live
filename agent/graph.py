# LangGraph graph definition — nodes, edges, entry point, conditional routing.
# Node order: retrieve_memory → generate → (async) extract_memory
# Implementation: Part 2 (generate only) → extended in Part 3 (memory nodes)
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import make_extract_memory_node, make_generate_node, make_retrieve_memory_node
from agent.state import AgentState
from memory.embeddings import Embedder
from memory.long_term import LongTermMemory


def build_graph(config: dict) -> CompiledStateGraph:
    """Build and compile the conversation graph: retrieve_memory -> generate -> extract_memory.

    extract_memory is sequenced last in the graph but does its real work in a
    background asyncio task (see make_extract_memory_node) — it returns near-
    instantly, so it doesn't add to this turn's latency.
    """
    embedder = Embedder(config)
    long_term = LongTermMemory(config, embedder)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_memory", make_retrieve_memory_node(long_term))
    graph.add_node("generate", make_generate_node(config))
    graph.add_node("extract_memory", make_extract_memory_node(config, long_term))

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "generate")
    graph.add_edge("generate", "extract_memory")
    graph.add_edge("extract_memory", END)

    return graph.compile()
