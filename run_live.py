"""VoiceChat web demo + a live feed of what the pipeline is doing.

Local add-on (not part of the upstream repo). Starts the same FastAPI app as
demo/web/app.py and adds:
    /live         the voice UI and the live feed side by side
    /feed         the feed on its own
    /feed/events  Server-Sent Events stream behind it

The feed shows the pipeline's own log lines (speech detected, transcript,
guardrail checks, LLM timing, TTS, latency summary) plus a mic-level meter fed
by the audio the server actually receives from the browser.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi.responses import HTMLResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import demo.web.app as webapp  # noqa: E402
from demo.web.ws_handler import WebSocketAudioIO  # noqa: E402

app = webapp.app

_OWN_LOGGERS = ("demo.", "pipeline.", "audio.", "agent.", "memory.", "guardrails.", "run_live")
_backlog: collections.deque[str] = collections.deque(maxlen=300)
_clients: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _publish(event: dict) -> None:
    """Thread-safe: queue one event for every connected feed page."""
    data = json.dumps(event)
    if event["type"] == "log":
        _backlog.append(data)
    if _loop is None:
        return

    def fan_out() -> None:
        for q in list(_clients):
            if q.qsize() < 500:
                q.put_nowait(data)

    _loop.call_soon_threadsafe(fan_out)


class _FeedHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith(_OWN_LOGGERS):
            return  # skip library noise (NeMo prompt dumps, httpx, uvicorn)
        _publish({
            "type": "log",
            "t": time.strftime("%H:%M:%S", time.localtime(record.created)),
            "src": record.name.split(".")[-1],
            "msg": record.getMessage(),
        })


logging.getLogger().addHandler(_FeedHandler())
logging.getLogger("audio.vad").setLevel(logging.DEBUG)  # speech start/end lines
log = logging.getLogger("run_live")

# ── Mic-level meter: observe the audio the browser is really sending ─────────
_orig_push = WebSocketAudioIO.push_chunk
_level = {"peak": 0.0, "last": 0.0}


def _push_with_meter(self: WebSocketAudioIO, chunk: np.ndarray) -> None:
    if len(chunk):
        _level["peak"] = max(_level["peak"], float(np.max(np.abs(chunk))))
    now = time.monotonic()
    if now - _level["last"] >= 0.25:
        _publish({"type": "level", "peak": _level["peak"], "muted": self._discard_incoming})
        _level["peak"], _level["last"] = 0.0, now
    _orig_push(self, chunk)


WebSocketAudioIO.push_chunk = _push_with_meter


@app.get("/feed/events")
async def feed_events() -> StreamingResponse:
    global _loop
    _loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    for item in _backlog:
        q.put_nowait(item)
    _clients.add(q)

    async def stream():
        try:
            while True:
                try:
                    yield f"data: {await asyncio.wait_for(q.get(), timeout=15)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            _clients.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


_FEED_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>VoiceChat live feed</title>
<style>
 body{margin:0;background:#0b0d12;color:#cbd5e1;font:13px/1.45 Consolas,monospace;display:flex;flex-direction:column;height:100vh}
 header{padding:10px 14px;border-bottom:1px solid #1e2430;font-family:Segoe UI,sans-serif}
 h1{font-size:13px;margin:0 0 8px;color:#e2e8f0;letter-spacing:.08em;text-transform:uppercase}
 #bar{height:10px;background:#1e2430;border-radius:5px;overflow:hidden}
 #fill{height:100%;width:0;background:#e0263f;transition:width .15s}
 #mic{margin-top:6px;font-size:12px;color:#94a3b8}
 #log{flex:1;overflow:auto;padding:8px 14px}
 .row{white-space:pre-wrap;word-break:break-word;padding:1px 0}
 .t{color:#64748b}.src{color:#8b5cf6}
 .stt{color:#fbbf24}.vad{color:#f87171}.nodes{color:#60a5fa}.tts{color:#34d399}.latency_monitor{color:#f472b6}
</style></head><body>
<header><h1>Live feed</h1><div id="bar"><div id="fill"></div></div>
<div id="mic">Waiting for a session — click the mic to start.</div></header>
<div id="log"></div>
<script>
const logEl=document.getElementById('log'),fill=document.getElementById('fill'),mic=document.getElementById('mic');
let lastLevel=0;
function add(e){const d=document.createElement('div');d.className='row';
 d.innerHTML='<span class="t"></span> <span class="src"></span> <span></span>';
 d.children[0].textContent=e.t;d.children[1].textContent='['+e.src+']';
 d.children[2].textContent=e.msg;d.children[2].className=e.src;
 const stick=logEl.scrollTop+logEl.clientHeight>=logEl.scrollHeight-40;
 logEl.appendChild(d);if(stick)logEl.scrollTop=logEl.scrollHeight;}
const es=new EventSource('/feed/events');
es.onmessage=m=>{const e=JSON.parse(m.data);
 if(e.type==='log')add(e);
 else if(e.type==='level'){lastLevel=Date.now();
  const db=e.peak>0?20*Math.log10(e.peak):-100;
  fill.style.width=Math.max(0,Math.min(100,(db+60)/60*100))+'%';
  fill.style.background=e.muted?'#3b82f6':'#e0263f';
  mic.textContent=e.muted?'Assistant is speaking (mic ignored)':
   'Mic level: '+db.toFixed(0)+' dB'+(db<-45?'  —  very quiet / silent: check the mic chosen in the browser':'');}};
setInterval(()=>{if(lastLevel&&Date.now()-lastLevel>2000){fill.style.width='0';
 mic.textContent='No mic audio is reaching the server (session stopped, or the browser is not sending audio).';}},1000);
</script></body></html>"""

_LIVE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>VoiceChat</title>
<style>html,body{margin:0;height:100%;background:#0b0d12}
 .wrap{display:flex;height:100%}iframe{border:0;height:100%}
 #ui{flex:1 1 55%}#feed{flex:1 1 45%;border-left:1px solid #1e2430}</style></head>
<body><div class="wrap"><iframe id="ui" src="/" allow="microphone"></iframe>
<iframe id="feed" src="/feed"></iframe></div></body></html>"""


@app.get("/feed", response_class=HTMLResponse)
async def feed_page() -> str:
    return _FEED_HTML


@app.get("/live", response_class=HTMLResponse)
async def live_page() -> str:
    return _LIVE_HTML


if __name__ == "__main__":
    web = webapp._config["web"]
    log.info("Live view: http://%s:%d/live", web["host"], web["port"])
    uvicorn.run(app, host=web["host"], port=web["port"])
