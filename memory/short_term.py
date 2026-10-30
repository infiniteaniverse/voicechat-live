# Short-term memory — rolling deque of last N conversation turns.
# Held in-memory for the session only (no persistence).
# Provides context window for the generate node.
# Implementation: Part 3
from __future__ import annotations

from collections import deque


class ShortTermMemory:
    """Rolling window of the last N conversation turns, in-memory only.

    Owned by the pipeline (Part 7), not the LangGraph state graph — the graph
    only ever sees a snapshot of this via AgentState["messages"], passed in
    before invoking the graph and written back with add_turn() after.
    See CLAUDE.md Key Engineering Decisions for why this is a deque and not
    SQLite/Postgres (latency: microsecond reads on the hot path).
    """

    def __init__(self, config: dict) -> None:
        max_turns: int = config["memory"]["short_term"]["max_turns"]
        # 2 messages (user + assistant) per turn.
        self._buffer: deque[dict[str, str]] = deque(maxlen=max_turns * 2)

    def get_messages(self) -> list[dict[str, str]]:
        """Return the current buffer as an Ollama-style messages list, oldest first."""
        return list(self._buffer)

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """Append one user+assistant exchange. Oldest turn is dropped if over max_turns."""
        self._buffer.append({"role": "user", "content": user_content})
        self._buffer.append({"role": "assistant", "content": assistant_content})

    def clear(self) -> None:
        """Reset the buffer. Call between sessions."""
        self._buffer.clear()
