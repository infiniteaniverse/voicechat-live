# Tests for memory: store/retrieve facts, similarity threshold, deduplication, persistence.
# Implementation: Part 3
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def base_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def embedder(base_config):
    from memory.embeddings import Embedder
    return Embedder(base_config)


@pytest.fixture
def long_term(base_config, embedder, tmp_path):
    """Fresh LongTermMemory per test, backed by a throwaway tmp_path store."""
    from memory.long_term import LongTermMemory

    config = copy.deepcopy(base_config)
    config["memory"]["long_term"]["store_path"] = str(tmp_path / "faiss_index")
    config["memory"]["long_term"]["facts_db_path"] = str(tmp_path / "facts.sqlite")
    return LongTermMemory(config, embedder)


# ── Embeddings ───────────────────────────────────────────────────────────────────

def test_embed_returns_correct_shape_and_dtype(embedder):
    vector = embedder.embed("hello world")
    assert vector.shape == (embedder.dimension,)
    assert vector.dtype == np.float32


def test_embed_is_normalized(embedder):
    vector = embedder.embed("some arbitrary text")
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4


def test_embed_similar_texts_score_higher_than_unrelated(embedder):
    a = embedder.embed("I love pizza")
    b = embedder.embed("Pizza is my favorite food")
    c = embedder.embed("The stock market crashed today")
    assert float(np.dot(a, b)) > float(np.dot(a, c))


# ── Short-term memory ────────────────────────────────────────────────────────────

def test_short_term_empty_initially(base_config):
    from memory.short_term import ShortTermMemory
    assert ShortTermMemory(base_config).get_messages() == []


def test_short_term_add_turn_appends_user_then_assistant(base_config):
    from memory.short_term import ShortTermMemory

    stm = ShortTermMemory(base_config)
    stm.add_turn("hi", "hello there")
    assert stm.get_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_short_term_respects_max_turns(base_config):
    from memory.short_term import ShortTermMemory

    config = copy.deepcopy(base_config)
    config["memory"]["short_term"]["max_turns"] = 2
    stm = ShortTermMemory(config)
    for i in range(5):
        stm.add_turn(f"u{i}", f"a{i}")

    messages = stm.get_messages()
    assert len(messages) == 4  # 2 turns * 2 messages/turn
    assert messages[0] == {"role": "user", "content": "u3"}  # oldest surviving turn


def test_short_term_clear(base_config):
    from memory.short_term import ShortTermMemory

    stm = ShortTermMemory(base_config)
    stm.add_turn("hi", "hello")
    stm.clear()
    assert stm.get_messages() == []


# ── Long-term memory ─────────────────────────────────────────────────────────────

def test_store_and_retrieve_fact(long_term):
    long_term.store_fact("The user's name is Jay.")
    results = long_term.retrieve_similar("What is the user's name?")
    assert any("Jay" in r for r in results)


def test_retrieve_on_empty_store_returns_empty_list(long_term):
    assert long_term.retrieve_similar("anything at all") == []


def test_retrieve_respects_similarity_threshold(long_term):
    """An unrelated query must not surface a stored fact just because the store is non-empty."""
    long_term.store_fact("The user loves hiking in the mountains.")
    results = long_term.retrieve_similar("What's the weather in Tokyo tomorrow?")
    assert results == []


def test_retrieve_top_k_limits_results(long_term):
    for fact in [
        "The user's favorite pizza topping is mushroom.",
        "The user's favorite pizza topping is pepperoni.",
        "The user's favorite pizza topping is olives.",
        "The user's favorite pizza topping is basil.",
    ]:
        long_term.store_fact(fact)
    results = long_term.retrieve_similar("What pizza toppings does the user like?", top_k=2)
    assert len(results) <= 2


def test_deduplication_skips_near_identical_facts(long_term):
    id1 = long_term.store_fact("The user's favorite color is blue.")
    id2 = long_term.store_fact("The user's favorite color is blue.")
    assert id1 == id2

    results = long_term.retrieve_similar("What color does the user like?")
    assert sum("blue" in r for r in results) == 1


def test_delete_fact_removes_it(long_term):
    fact_id = long_term.store_fact("The user owns a cat named Whiskers.")
    query = "What is the name of the user's cat?"
    assert any("Whiskers" in r for r in long_term.retrieve_similar(query))

    long_term.delete_fact(fact_id)
    assert not any("Whiskers" in r for r in long_term.retrieve_similar(query))


def test_persistence_across_instances(base_config, embedder, tmp_path):
    """A new LongTermMemory pointed at the same paths must see facts from a prior instance."""
    from memory.long_term import LongTermMemory

    config = copy.deepcopy(base_config)
    config["memory"]["long_term"]["store_path"] = str(tmp_path / "faiss_index")
    config["memory"]["long_term"]["facts_db_path"] = str(tmp_path / "facts.sqlite")

    first = LongTermMemory(config, embedder)
    first.store_fact("The user works as a software engineer.")

    second = LongTermMemory(config, embedder)  # reloads index + db from disk
    results = second.retrieve_similar("What is the user's job?")
    assert any("engineer" in r for r in results)
