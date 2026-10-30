"""Browser side of the live dashboard (local add-on, used by run_viz.py in web mode).

Adds to the existing FastAPI app, without editing it:
    /          split page: the normal voice UI on the left, the live terminal on the right
    /voice     the original voice UI on its own
    /term      WebSocket that streams dashboard frames (ANSI) to xterm.js
    /viz/...   vendored xterm.js (no CDN, works offline)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from terminal_viz import FPS, Dashboard

ROOT = Path(__file__).resolve().parent

SPLIT_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceChat live</title>
<link rel="stylesheet" href="/viz/xterm.css">
<style>
  :root { --bg:#0b0d12; --line:#1e2430; --muted:#8b95a7; --ok:#06d6a0; --warn:#ffd166; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; overflow:hidden; background:var(--bg); color:#cbd5e1;
               font:13px/1.4 "Segoe UI", system-ui, sans-serif; }
  .wrap { display:flex; height:100%; }
  #voice { flex:0 0 34%; min-width:320px; border:0; background:var(--bg); }
  #side { flex:1 1 0; min-width:0; display:flex; flex-direction:column; border-left:1px solid var(--line); }
  #bar { display:flex; align-items:center; gap:10px; padding:6px 12px; border-bottom:1px solid var(--line);
         color:var(--muted); letter-spacing:.08em; text-transform:uppercase; font-size:11px; }
  #dot { width:8px; height:8px; border-radius:50%; background:var(--warn); }
  #dot.live { background:var(--ok); box-shadow:0 0 6px var(--ok); }
  #term { flex:1; min-height:0; padding:4px 8px 0; overflow:hidden; }
  #term .xterm-viewport { overflow-y:hidden !important; }  /* the dashboard redraws in place; no scrolling */
  @media (max-width: 900px) {
    .wrap { flex-direction:column; }
    #voice { flex:0 0 46%; min-width:0; }
    #side { border-left:0; border-top:1px solid var(--line); }
  }
</style></head>
<body>
<div class="wrap">
  <iframe id="voice" src="/voice" allow="microphone; autoplay" title="Voice chat"></iframe>
  <div id="side">
    <div id="bar"><span id="dot"></span><span id="status">connecting to the live pipeline...</span></div>
    <div id="term"></div>
  </div>
</div>
<script src="/viz/xterm.js"></script>
<script src="/viz/xterm-addon-fit.js"></script>
<script>
  const el = document.getElementById("term");
  const dot = document.getElementById("dot");
  const status = document.getElementById("status");
  const term = new Terminal({
    cursorBlink: false, disableStdin: true, scrollback: 0, fontSize: 13, allowProposedApi: true,
    fontFamily: '"Cascadia Mono", Consolas, "Courier New", monospace',
    theme: { background: "#0b0d12", foreground: "#d0d6e0" },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(el);
  let ws = null;
  let sent = "";

  function sendSize(force) {
    const msg = JSON.stringify({ cols: term.cols, rows: term.rows });
    if (ws && ws.readyState === WebSocket.OPEN && (force || msg !== sent)) {
      ws.send(msg);
      sent = msg;
      status.textContent = `live pipeline - every step, animated  (${term.cols}x${term.rows})`;
    }
  }
  // xterm measures the font before it has loaded, so fit again once fonts are ready and keep checking.
  function refit(force) { try { fit.fit(); } catch (e) {} sendSize(force === true); }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/term`);
    ws.onopen = () => { dot.className = "live"; sent = ""; refit(true); };
    ws.onmessage = (e) => term.write(e.data);
    ws.onclose = () => { dot.className = ""; status.textContent = "reconnecting..."; setTimeout(connect, 1000); };
  }

  new ResizeObserver(() => refit()).observe(el);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => refit());
  window.addEventListener("resize", () => refit());
  setInterval(() => refit(), 1000);
  refit();
  connect();
</script>
</body></html>
"""


def install(app: FastAPI, dash: Dashboard, voice_index: Path) -> None:
    """Replace '/' with the split page and add /voice, /term and /viz."""
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
    app.mount("/viz", StaticFiles(directory=str(ROOT / "viz_static")), name="viz")

    @app.get("/voice")
    async def voice() -> FileResponse:
        return FileResponse(str(voice_index))

    def page() -> Response:
        # no-store: a browser must never reuse an older copy of the page that lacks the terminal
        return HTMLResponse(SPLIT_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/")
    async def split() -> Response:
        return page()

    @app.get("/live")
    async def live() -> Response:
        return page()

    @app.websocket("/term")
    async def term(ws: WebSocket) -> None:
        await ws.accept()
        size = {"cols": 110, "rows": 40}

        async def read_size() -> None:
            try:
                while True:
                    msg = json.loads(await ws.receive_text())
                    size["cols"] = max(60, min(300, int(msg["cols"])))
                    size["rows"] = max(24, min(100, int(msg["rows"])))
            except Exception:
                pass  # disconnect or bad message: the send loop below notices and stops

        reader = asyncio.create_task(read_size())
        last_size = None
        try:
            while not reader.done():
                dash._tick()
                cols, rows = size["cols"], size["rows"]
                frame = await asyncio.to_thread(dash.render_ansi, cols, rows)
                if (cols, rows) != last_size:
                    frame = "\x1b[2J" + frame  # clear once after a resize
                    last_size = (cols, rows)
                await ws.send_text(frame)
                await asyncio.sleep(1 / FPS)
        except Exception:
            pass
        finally:
            reader.cancel()
