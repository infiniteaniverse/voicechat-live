# GuardrailsManager — loads the NeMo Guardrails config once, exposes
# check_input(text) / check_output(text), each a single self-check LLM call.
# Implementation: Part 5
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult

logger = logging.getLogger(__name__)

# Suppress two DeprecationWarnings emitted by NeMo Guardrails internals during
# RailsConfig.from_path() and LLMRails.__init__() — both originate inside
# nemoguardrails library code we can't change.
#
# 1. 'nim_url' field rename warning from nemoguardrails/rails/llm/config.py.
# 2. 'There is no current event loop' from nemoguardrails/utils.py — NeMo calls
#    asyncio.get_event_loop() at construction time without a running loop; this
#    is an internal NeMo asyncio shim that's harmless in practice (the manager
#    is always used in an async context at call time).
warnings.filterwarnings(
    "ignore",
    message=r"Use 'nim_base_url' instead",
    category=DeprecationWarning,
    module=r"nemoguardrails\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"There is no current event loop",
    category=DeprecationWarning,
    module=r"nemoguardrails\..*",
)

_INPUT_REFUSAL = "I'm not able to help with that. Let's talk about something else."
_OUTPUT_REFUSAL = "I'd rather not say that. Let's move on."


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of a single check_input()/check_output() call.

    message is the voice-friendly refusal line to speak instead of the
    checked text, and is None whenever allowed is True.
    """

    allowed: bool
    message: str | None = None


class GuardrailsManager:
    """Wraps NeMo Guardrails' self-check input/output actions.

    Deliberately calls the self_check_input/self_check_output actions
    directly via LLMRails.runtime.action_dispatcher instead of going through
    rails.generate()/the Colang flow engine. rails.generate() drives NeMo's
    own dialog loop, which — with nothing stopping it — falls through to
    generating a full response with the *main* model; that's redundant work
    we don't want, since agent/nodes.py's generate_node already owns real
    response generation. Direct action dispatch gets one focused LLM call per
    check and nothing else. See docs/tradeoffs.md for the full reasoning and
    the alternative approaches that were tried and rejected.
    """

    def __init__(self, config: dict) -> None:
        gr_cfg = config["guardrails"]
        self._enabled: bool = gr_cfg["enabled"]
        self._check_input_enabled: bool = gr_cfg["check_input"]
        self._check_output_enabled: bool = gr_cfg["check_output"]
        self._rails: LLMRails | None = None

        if not self._enabled:
            logger.info("Guardrails disabled via config — all checks will pass through")
            return

        config_dir = Path(gr_cfg["config_path"]).parent
        rails_config = RailsConfig.from_path(str(config_dir))
        self._rails = LLMRails(rails_config)
        logger.info("GuardrailsManager loaded config from %s", config_dir)

    async def check_input(self, text: str) -> GuardrailResult:
        """Check a user transcript before it reaches generate_node."""
        if not self._enabled or not self._check_input_enabled:
            return GuardrailResult(allowed=True)

        allowed = await self._run_check("self_check_input", {"user_message": text})
        if not allowed:
            logger.info("Input rail blocked a message")
        return GuardrailResult(allowed=allowed, message=None if allowed else _INPUT_REFUSAL)

    async def check_output(self, text: str) -> GuardrailResult:
        """Check one sentence chunk before it reaches TTS."""
        if not self._enabled or not self._check_output_enabled:
            return GuardrailResult(allowed=True)

        allowed = await self._run_check("self_check_output", {"bot_message": text})
        if not allowed:
            logger.info("Output rail blocked a sentence: %r", text)
        return GuardrailResult(allowed=allowed, message=None if allowed else _OUTPUT_REFUSAL)

    async def _run_check(self, action_name: str, context: dict) -> bool:
        assert self._rails is not None
        t0 = time.perf_counter()
        kwargs = {
            "context": context,
            "config": self._rails.config,
            "llm_task_manager": self._rails.runtime.llm_task_manager,
            "llm": self._rails.llm,
        }
        result, status = await self._rails.runtime.action_dispatcher.execute_action(action_name, kwargs)
        logger.info("%s [%.0fms]", action_name, (time.perf_counter() - t0) * 1000)

        if status != "success":
            # Fail open: a broken check shouldn't make the assistant go silent.
            # See docs/tradeoffs.md for why this is the right default here.
            logger.warning("%s failed (status=%s) — failing open (allowed)", action_name, status)
            return True

        return bool(result.return_value) if isinstance(result, ActionResult) else bool(result)
