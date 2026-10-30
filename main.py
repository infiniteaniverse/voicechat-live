# Entry point — wires all pipeline components and starts the voice loop.
# Implementation: Part 7
from __future__ import annotations

import asyncio
import logging

import yaml

from pipeline.pipeline import ConversationPipeline


def _configure_logging(config: dict) -> None:
    log_cfg = config["logging"]
    logging.basicConfig(
        level=getattr(logging, log_cfg["level"]),
        format="%(asctime)s %(name)s: %(message)s",
        filename=log_cfg["log_file"],
    )


async def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    _configure_logging(config)

    pipeline = ConversationPipeline(config)
    await pipeline.prewarm()
    await pipeline.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
