# FastAPI + WebSocket backend for the web demo UI.
# Streams mic audio in from browser via WebSocket, returns TTS audio chunks out.
# Implementation: Part 8 (stretch goal — only if time allows)
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demo.web.ws_handler import WebSocketAudioIO, handle_connection
from pipeline.pipeline import ConversationPipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _configure_logging(config: dict) -> None:
    log_cfg = config["logging"]
    logging.basicConfig(
        level=getattr(logging, log_cfg["level"]),
        format="%(asctime)s %(name)s: %(message)s",
        filename=log_cfg["log_file"],
    )


_config = _load_config()
_configure_logging(_config)
_static_dir = ROOT / _config["web"]["static_dir"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    audio_io = WebSocketAudioIO()
    pipeline = ConversationPipeline(_config, audio_manager=audio_io)
    logger.info("Web demo: pre-warming pipeline...")
    await pipeline.prewarm()
    logger.info("Web demo: ready.")

    app.state.pipeline = pipeline
    app.state.audio_io = audio_io
    app.state.session_active = False

    yield

    await pipeline.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_static_dir / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """One conversation session per connection. Only one concurrent session is
    allowed (config.yaml's web.max_concurrent_sessions) — this is a
    single-user demo, not a multi-tenant service; see docs/tradeoffs.md.
    """
    if websocket.app.state.session_active:
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "message": "A conversation is already in progress. Try again shortly."}
        )
        await websocket.close(code=1008)
        return

    websocket.app.state.session_active = True
    try:
        await handle_connection(websocket, websocket.app.state.pipeline, websocket.app.state.audio_io)
    finally:
        websocket.app.state.session_active = False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=_config["web"]["host"], port=_config["web"]["port"])
