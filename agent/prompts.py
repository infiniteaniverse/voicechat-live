# All prompt strings in one place — system prompt, memory extraction prompt, fallback lines.
# No logic here, just string constants and template formatters.
# Implementation: Part 2
from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful, friendly voice assistant having a spoken conversation. \
The user is talking to you out loud, and everything you say will be converted to speech, so:
- Write in plain, natural spoken sentences. Never use markdown, bullet points, numbered lists, \
code blocks, headers, or emojis — say things the way a person would say them out loud.
- Keep answers short: 1-3 sentences for most questions. Only go longer if the user clearly asks \
for detail or a step-by-step explanation.
- If you don't know something, say so briefly instead of guessing at length.
- Stay warm and conversational, not robotic or overly formal."""


def build_system_prompt(retrieved_facts: list[str] | None = None) -> str:
    """SYSTEM_PROMPT, optionally extended with facts retrieved from long-term memory."""
    if not retrieved_facts:
        return SYSTEM_PROMPT
    facts_block = "\n".join(f"- {fact}" for fact in retrieved_facts)
    return f"{SYSTEM_PROMPT}\n\nRelevant things you remember about the user:\n{facts_block}"


# EXTRACTION_PROMPT — the "is this worth remembering?" prompt used by extract_memory_node.
# Implementation: Part 3, revised in Part 8 follow-up (see docs/tradeoffs.md decision 31)
#
# Tuned empirically against qwen2.5:3b (see docs/tradeoffs.md for the iteration notes).
# Three findings that shaped this: (1) few-shot examples with NONE cases listed before
# positive cases classify much more reliably than a terse zero-shot instruction; (2)
# including the assistant's reply alongside the user's text made the model noticeably
# less reliable (a differently-phrased acknowledgment could flip a correct extraction to
# NONE) — classifying on the user's utterance alone scored 9/10 on a manual test set vs.
# 5-7/9 with the assistant reply included, so the assistant reply is intentionally left
# out; (3) the original "extract only the single most important fact" rule was a real bug,
# not a simplification — measured directly against qwen2.5:3b, "Hi, my name is Jay and I
# work as a machine learning engineer" deterministically (temperature=0) extracted only the
# occupation, never the name, 3/3 times; swapping the sentence order flipped which fact won
# ("...and my name is Jay" extracted the name instead) — confirming a recency/positional
# bias, not a genuine importance judgment. Any single-sentence introduction naturally
# containing two facts (a very common way people actually talk) was silently losing one of
# them. Now extracts every distinct fact present, one per line, instead of picking one.
# Small local models are inconsistent classifiers; this is "good enough for a cheap async
# call", not a guarantee — occasional misses are still expected.
EXTRACTION_SYSTEM_PROMPT = """You extract durable facts about the user from one thing they said, \
for a long-term memory store.

A fact is worth extracting if it is about the user specifically and would still be true and useful \
in a future conversation: their name, occupation, preferences, health/dietary needs, ongoing goals, \
or relationships.
A fact is NOT worth extracting if it is small talk, a one-off question, or a request with no \
lasting info about the user.

Examples of NOT worth extracting:
- "What time is it?" -> NONE
- "Can you recommend a good pizza place?" -> NONE
- "Thanks, that helps." -> NONE

Examples of worth extracting (always rewrite in third person, starting with "The user"):
- "My name is Jay." -> The user's name is Jay.
- "I am allergic to peanuts, just so you know." -> The user is allergic to peanuts.
- "I really prefer when you keep answers short." -> The user prefers short answers.
- "My sister Maya is visiting me next week." -> The user's sister Maya is visiting them next week.

Example with more than one fact — extract EVERY distinct fact, one per line:
- "Hi, my name is Jay and I work as a machine learning engineer." ->
The user's name is Jay.
The user works as a machine learning engineer.

Rules:
- ALWAYS rewrite worth-extracting facts in third person starting with "The user" — never copy \
first-person phrasing like "I am" or "my" directly.
- If multiple distinct facts are present, extract every one of them. Do not pick just one.
- Respond with one short third-person sentence per fact, each on its own line, or exactly NONE \
if there is nothing worth extracting. No preamble, no numbering, no quotes, nothing after the \
last fact."""


def build_extraction_prompt(user_text: str) -> str:
    """User utterance to classify — pairs with EXTRACTION_SYSTEM_PROMPT.

    Deliberately takes only the user's text, not the assistant's reply — see the
    module-level comment above EXTRACTION_SYSTEM_PROMPT for why.
    """
    return f'User said: "{user_text}"\n\nExtract the fact or respond NONE.'
