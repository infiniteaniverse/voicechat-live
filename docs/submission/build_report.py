# Builds docs/submission/report.html (+ figures) from the saved experiment results in
# docs/submission/results/*.json. Print report.html to PDF with a Chromium-based browser:
#   msedge --headless=new --no-pdf-header-footer --print-to-pdf=BCSE306L_TeamName_FinalReport.pdf report.html
# Every number in the experiments section is read from the JSON files written by
# scripts/eval_retrieval.py, scripts/eval_extraction.py and scripts/eval_guardrails.py.
from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

R = json.loads((RES / "retrieval.json").read_text(encoding="utf-8"))
E = json.loads((RES / "extraction.json").read_text(encoding="utf-8"))
G = json.loads((RES / "guardrails.json").read_text(encoding="utf-8"))

# ── palette (validated with the dataviz validator: CVD dE 9.2, normal-vision dE 27.6) ──
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#1f2328", "#57606a", "#e3e6ea"
FONT = "Segoe UI, system-ui, sans-serif"

REPO = "https://github.com/infiniteaniverse/voicechat-live"
# REPORT_IMG=1 builds a Word-friendly variant (report_docx.html): figures become PNG files
# and placeholders carry a Word character style, for conversion with pandoc.
IMG_MODE = os.environ.get("REPORT_IMG") == "1"
PH = (lambda s: f'<span custom-style="Placeholder">[{html.escape(s)}]</span>') if IMG_MODE else      (lambda s: f'<span class="ph">[{html.escape(s)}]</span>')  # fill-in placeholder, highlighted


def _edge() -> str:
    for c in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(c):
            return c
    raise SystemExit("msedge.exe not found")


def F(name: str, svg: str, alt: str) -> str:
    """PDF mode: the inline SVG. Word mode: a 2x PNG rendered from it with headless Edge."""
    if not IMG_MODE:
        return svg
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    w, h = int(m.group(1)), int(m.group(2))
    (HERE / "figures").mkdir(exist_ok=True)
    page = HERE / "figures" / f"{name}.svg.html"
    page.write_text(f'<!doctype html><meta charset="utf-8"><style>html,body{{margin:0;overflow:hidden;background:#fff}}svg{{display:block}}</style><body>{svg.replace("<svg ", f"<svg width=\"{w}\" height=\"{h}\" ", 1)}</body>', encoding="utf-8")
    out = HERE / "figures" / f"{name}.png"
    subprocess.run([_edge(), "--headless=new", "--disable-gpu", f"--screenshot={out}", f"--window-size={w},{h}",
                    "--force-device-scale-factor=2", f"--user-data-dir={tempfile.gettempdir()}/edge_fig_prof", page.as_uri()],
                   check=True, capture_output=True)
    page.unlink()
    return f'<img src="figures/{name}.png" alt="{alt}" />'


def esc(s) -> str:
    return html.escape(str(s))


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


# ───────────────────────────── SVG helpers ─────────────────────────────
def svg_text(x, y, s, size=9, anchor="start", weight="normal", fill=INK2, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}" {extra}>{esc(s)}</text>')


def line_panel(x0, y0, w, h, title, series, vlines, xmax=0.8):
    """One panel of line series over tau in [0, xmax], y in [0, 1]. series: (name, color, dash, pts, width)"""
    pl, pr, pt, pb = 34, 8, 30, 26
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda v: x0 + pl + v / xmax * iw
    Y = lambda v: y0 + pt + (1 - v) * ih
    out = [svg_text(x0 + pl, y0 + 12, title, 10, weight="600", fill=INK)]
    for gv in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{X(0):.1f}" x2="{X(xmax):.1f}" y1="{Y(gv):.1f}" y2="{Y(gv):.1f}" stroke="{GRID}" stroke-width="0.8"/>')
        out.append(svg_text(X(0) - 4, Y(gv) + 3, f"{gv:g}", 8, "end"))
    for xv in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(svg_text(X(xv), Y(0) + 12, f"{xv:g}", 8, "middle"))
    out.append(svg_text(X(xmax / 2), Y(0) + 23, "similarity threshold τ", 8, "middle"))
    for xv, label, col in vlines:
        out.append(f'<line x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{Y(1):.1f}" y2="{Y(0):.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')
        if xv < 0.3:
            out.append(svg_text(X(xv) - 3, Y(0.04), label, 7.5, "end", fill=col, extra='stroke="#ffffff" stroke-width="3" paint-order="stroke"'))
        else:
            out.append(svg_text(X(xv) + 3, Y(0.14), label, 7.5, fill=col, extra='stroke="#ffffff" stroke-width="3" paint-order="stroke"'))
    for name, col, dash, pts, sw in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-dasharray="{dash}" stroke-linejoin="round" stroke-linecap="round"/>')
    return "\n".join(out)


def legend(x, y, items, gap=10):
    out, cx = [], x
    for name, col, dash in items:
        out.append(f'<line x1="{cx}" x2="{cx + 18}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
        out.append(svg_text(cx + 22, y + 3, name, 8.5, fill=INK))
        cx += 22 + len(name) * 4.9 + gap + 6
    return "\n".join(out)


def fig_threshold_sweep() -> str:
    sw = R["sweep"]
    ser = lambda name, key: [(r["tau"], r[key]) for r in sw[name]]
    A = [
        ("recall", BLUE, "none", ser("dense_1p", "recall"), 2),
        ("precision", ORANGE, "6 3", ser("dense_1p", "precision"), 2),
        ("false-retrieval rate", AQUA, "2 3", ser("dense_1p", "false_retrieval_rate"), 2.2),
        ("F1", VIOLET, "none", ser("dense_1p", "f1"), 2.6),
    ]
    B = [
        ("dense 1st-person", BLUE, "none", ser("dense_1p", "f1"), 2.2),
        ("dense 3rd-person", ORANGE, "6 3", ser("dense_3p", "f1"), 2.2),
        ("TF-IDF", AQUA, "2 3", ser("tfidf_1p", "f1"), 2.2),
    ]
    vl = [(0.15, "0.15 adopted", INK), (0.60, "0.60 original plan", INK2)]
    parts = [f'<svg viewBox="0 0 700 262" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Threshold sweep for memory retrieval">']
    parts.append(line_panel(0, 22, 346, 238, "(a) Dense retrieval across thresholds", A, vl))
    parts.append(line_panel(354, 22, 346, 238, "(b) F1 by retriever and query phrasing", B, vl))
    parts.append(legend(34, 10, [(n, c, d) for n, c, d, *_ in A], 4))
    parts.append(legend(372, 10, [(n, c, d) for n, c, d, *_ in B], 2))
    parts.append("</svg>")
    return "\n".join(parts)


def fig_score_strips() -> str:
    d1, d3 = R["distributions"]["dense_1p"], R["distributions"]["dense_3p"]
    cols = [
        ("correct fact\n1st-person", d1["positive_gold_scores"], BLUE),
        ("best fact for\nunrelated query", d1["negative_best_scores"], ORANGE),
        ("correct fact\n3rd-person", d3["positive_gold_scores"], BLUE),
        ("best fact for\nunrelated query", d3["negative_best_scores"], ORANGE),
    ]
    W, H, pl, pt, pb = 700, 230, 40, 14, 40
    ih = H - pt - pb
    Y = lambda v: pt + (1 - v / 0.8) * ih
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cosine score distributions">']
    for gv in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line x1="{pl}" x2="{W - 124}" y1="{Y(gv):.1f}" y2="{Y(gv):.1f}" stroke="{GRID}" stroke-width="0.8"/>')
        out.append(svg_text(pl - 5, Y(gv) + 3, f"{gv:.1f}", 8, "end"))
    for tv, lab in ((0.15, "τ = 0.15 (adopted)"), (0.60, "τ = 0.60 (original plan)")):
        out.append(f'<line x1="{pl}" x2="{W - 124}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" stroke="{INK}" stroke-width="1" stroke-dasharray="4 3"/>')
        out.append(svg_text(W - 120, Y(tv) + 3, lab, 8, "start", fill=INK))
    colw = (W - pl - 130) / 4
    for i, (lab, vals, col) in enumerate(cols):
        cx = pl + colw * (i + 0.5)
        for j, v in enumerate(sorted(vals)):
            jit = ((j * 37) % 11 - 5) * 4.2
            out.append(f'<circle cx="{cx + jit:.1f}" cy="{Y(v):.1f}" r="3.3" fill="{col}" fill-opacity="0.85" stroke="#fff" stroke-width="1"/>')
        med = sorted(vals)[len(vals) // 2]
        out.append(f'<line x1="{cx - 34:.1f}" x2="{cx + 34:.1f}" y1="{Y(med):.1f}" y2="{Y(med):.1f}" stroke="{INK}" stroke-width="1.6"/>')
        out.append(svg_text(cx + 38, Y(med) - 4, f"median {med:.2f}", 8, fill=INK, extra='stroke="#ffffff" stroke-width="3" paint-order="stroke"'))
        for k, ln in enumerate(lab.split("\n")):
            out.append(svg_text(cx, H - 22 + k * 11, ln, 8.5, "middle", fill=INK))
    out.append(svg_text(10, pt + ih / 2, "cosine", 8, "middle", extra=f'transform="rotate(-90 10 {pt + ih / 2})"'))
    out.append("</svg>")
    return "\n".join(out)


def fig_extraction() -> str:
    v1, v2 = E[0], E[1]
    groups = [
        ("facts recalled", v1["fact_recall"], v2["fact_recall"], f'{v1["gold_recalled"]}/{v1["gold_facts"]}', f'{v2["gold_recalled"]}/{v2["gold_facts"]}'),
        ("utterances fully captured", v1["utterances_fully_captured"] / v1["n_positive_utterances"], v2["utterances_fully_captured"] / v2["n_positive_utterances"],
         f'{v1["utterances_fully_captured"]}/{v1["n_positive_utterances"]}', f'{v2["utterances_fully_captured"]}/{v2["n_positive_utterances"]}'),
        ("multi-fact utterances fully captured", v1["multi_fact_utterances_fully_captured"] / v1["n_multi_fact_utterances"], v2["multi_fact_utterances_fully_captured"] / v2["n_multi_fact_utterances"],
         f'{v1["multi_fact_utterances_fully_captured"]}/{v1["n_multi_fact_utterances"]}', f'{v2["multi_fact_utterances_fully_captured"]}/{v2["n_multi_fact_utterances"]}'),
    ]
    W, H, pl, pt, pb = 700, 210, 40, 30, 38
    ih = H - pt - pb
    Y = lambda v: pt + (1 - v) * ih
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Extraction prompt comparison">']
    out.append(legend(pl, 12, [("v1: “extract the single most important fact”", ORANGE, "none"), ("v2: “extract every distinct fact” (current)", BLUE, "none")], 14))
    for gv in (0, 0.5, 1):
        out.append(f'<line x1="{pl}" x2="{W - 10}" y1="{Y(gv):.1f}" y2="{Y(gv):.1f}" stroke="{GRID}" stroke-width="0.8"/>')
        out.append(svg_text(pl - 5, Y(gv) + 3, f"{int(gv * 100)}%", 8, "end"))
    gw = (W - pl - 20) / 3
    bw = 62
    for i, (lab, a, b, la, lb) in enumerate(groups):
        cx = pl + gw * (i + 0.5)
        for dx, val, col, lt in ((-bw / 2 - 1, a, ORANGE, la), (bw / 2 + 1, b, BLUE, lb)):
            x = cx + dx - bw / 2
            top, base = Y(val), Y(0)
            hgt = max(base - top, 1.5)
            out.append(f'<path d="M{x:.1f},{base:.1f} V{base - hgt + 3:.1f} Q{x:.1f},{base - hgt:.1f} {x + 3:.1f},{base - hgt:.1f} H{x + bw - 3:.1f} Q{x + bw:.1f},{base - hgt:.1f} {x + bw:.1f},{base - hgt + 3:.1f} V{base:.1f} Z" fill="{col}"/>')
            out.append(svg_text(x + bw / 2, base - hgt - 4, f"{pct(val)}  ({lt})", 8.5, "middle", "600", INK))
        out.append(svg_text(cx, H - 18, lab, 9, "middle", fill=INK))
    out.append("</svg>")
    return "\n".join(out)


# per-turn stage timings (ms) read off the dashboard run of 2026-10-27, data/viz_logs/viz_20261027_104319_timeline.txt
LAT_TURNS = [
    ("Turn 1  (first turn after start-up)", 882, 434, 329, 485),
    ("Turn 2  (“What is the capital of France?”)", 342, 383, 336, 218),
    ("Turn 4  (“What do I do for a living?”, memory recall)", 351, 375, 352, 603),
]
LAT_STAGES = [("speech-to-text", BLUE), ("input guardrail", ORANGE), ("LLM first token", AQUA), ("output guardrail ∥ TTS", VIOLET)]


def fig_latency() -> str:
    W, H, pl, pr, pt = 700, 160, 232, 64, 30
    rowh, xmax = 30, 2400
    X = lambda v: pl + v / xmax * (W - pl - pr)
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Per-turn latency to first audio">']
    out.append(legend(pl - 120, 12, [(n, c, "none") for n, c in LAT_STAGES], 6))
    for gv in (0, 500, 1000, 1500, 2000):
        out.append(f'<line x1="{X(gv):.1f}" x2="{X(gv):.1f}" y1="{pt - 4}" y2="{pt + rowh * 3 + 2}" stroke="{GRID}" stroke-width="0.8"/>')
        out.append(svg_text(X(gv), pt + rowh * 3 + 14, str(gv), 8, "middle"))
    out.append(svg_text(X(1000), pt + rowh * 3 + 26, "milliseconds from end of speech to first audio out", 8, "middle"))
    for i, (lab, *segs) in enumerate(LAT_TURNS):
        y = pt + i * rowh + 4
        out.append(svg_text(pl - 8, y + 13, lab, 8.5, "end", fill=INK))
        x = 0
        for (nm, col), v in zip(LAT_STAGES, segs):
            out.append(f'<rect x="{X(x) + 0.5:.1f}" y="{y}" width="{max(X(x + v) - X(x) - 1.5, 1):.1f}" height="18" fill="{col}" rx="2"/>')
            x += v
        out.append(svg_text(X(x) + 5, y + 13, f"{x} ms", 8.5, fill=INK, weight="600"))
    out.append(f'<line x1="{X(2000):.1f}" x2="{X(2000):.1f}" y1="{pt - 6}" y2="{pt + rowh * 3}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    out.append(svg_text(X(2000) + 3, pt - 8, "2000 ms budget", 8, fill=INK, weight="600"))
    out.append("</svg>")
    return "\n".join(out)


def fig_architecture() -> str:
    bw, bh, gap, x0 = 116, 46, 22, 6
    cx = lambda c: x0 + c * (bw + gap)
    ry = {1: 6, 2: 100, 3: 194}
    kinds = {"model": ("#e8f1fc", BLUE), "logic": ("#fdeee7", ORANGE), "store": ("#e3f6ef", AQUA)}
    out = ['<svg viewBox="0 0 700 262" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="System architecture">',
           f'<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="{INK2}"/></marker></defs>']

    def box(c, r, title, sub, kind):
        fill, stroke = kinds[kind]
        x, y = cx(c), ry[r]
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        out.append(svg_text(x + bw / 2, y + 19, title, 9.5, "middle", "600", INK))
        for k, ln in enumerate(sub.split("\n")):
            out.append(svg_text(x + bw / 2, y + 31 + k * 9.5, ln, 7.6, "middle"))

    def arrow(x1, y1, x2, y2, dash="none"):
        out.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{INK2}" stroke-width="1.2" fill="none" stroke-dasharray="{dash}" marker-end="url(#ah)"/>')

    box(0, 1, "1  Microphone", "browser WebSocket\nor PyAudio, 16 kHz", "logic")
    box(1, 1, "2  VAD", "Silero VAD + 700 ms\nsilence turn-end rule", "model")
    box(2, 1, "3  STT", "faster-whisper small\n(int8, CUDA)", "model")
    box(3, 1, "4  Input guardrail", "few-shot self-check\nQwen2.5-3B via NeMo", "model")
    box(4, 1, "5  Memory retrieval", "MiniLM embed + FAISS\ncosine, top-3, τ = 0.15", "logic")
    box(4, 2, "6  LLM generate", "Qwen2.5-3B (Ollama)\ntoken streaming", "model")
    box(3, 2, "7  Sentence chunker", "regex boundary split,\nstalling-phrase filter", "logic")
    box(2, 2, "8  Out-guard ∥ TTS", "per-sentence check and\nKokoro-82M run together", "model")
    box(1, 2, "9  Speaker", "bounded queue, mic\ngated during playback", "logic")
    box(4, 3, "Fact store", "FAISS (vectors) +\nSQLite (text), dedup 0.92", "store")
    box(3, 3, "Extract memory", "async few-shot LLM call,\none fact per line", "logic")
    box(2, 3, "Fallback timer", "filler phrase if no LLM\ntoken within 800 ms", "logic")
    mid = lambda c, r: (cx(c) + bw / 2, ry[r] + bh / 2)
    for c in range(0, 4):  # row 1, left to right
        arrow(cx(c) + bw, ry[1] + bh / 2, cx(c + 1) - 1, ry[1] + bh / 2)
    arrow(mid(4, 1)[0], ry[1] + bh, mid(4, 2)[0], ry[2] - 1)  # retrieval -> LLM
    for c in (4, 3, 2):  # row 2, right to left
        arrow(cx(c), ry[2] + bh / 2, cx(c - 1) + bw + 1, ry[2] + bh / 2)
    arrow(mid(4, 2)[0] - 30, ry[2] + bh, mid(3, 3)[0] + 30, ry[3] - 1, "4 3")  # LLM -> extract (async)
    arrow(cx(3) + bw, ry[3] + bh / 2, cx(4) - 1, ry[3] + bh / 2)  # extract -> store
    # store -> retrieval, routed round the right edge
    out.append(f'<path d="M{cx(4) + bw},{ry[3] + 14} H690 V{ry[1] + bh / 2} H{cx(4) + bw + 1}" stroke="{INK2}" stroke-width="1.2" fill="none" marker-end="url(#ah)"/>')
    arrow(mid(2, 3)[0], ry[3] - 1, mid(2, 2)[0], ry[2] + bh + 1, "4 3")  # fallback -> TTS
    kx = cx(0) + 2
    out.append(svg_text(kx, ry[3] + 10, "Key", 8.5, weight="600", fill=INK))
    for k, (kd, lab) in enumerate((("model", "pretrained model, not trained by us"), ("logic", "logic written by the team"), ("store", "persistent data store"))):
        fill, stroke = kinds[kd]
        yy = ry[3] + 22 + k * 13
        out.append(f'<rect x="{kx}" y="{yy - 8}" width="14" height="9" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        out.append(svg_text(kx + 19, yy, lab, 8, fill=INK))
    out.append(f'<line x1="{kx}" x2="{kx + 14}" y1="{ry[3] + 55}" y2="{ry[3] + 55}" stroke="{INK2}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    out.append(svg_text(kx + 19, ry[3] + 58, "step off the critical path", 8, fill=INK))
    out.append("</svg>")
    return "\n".join(out)


# ───────────────────────────── tables ─────────────────────────────
def table(head, rows, cls="", widths=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    thead = f"<thead><tr>{h}</tr></thead>" if any(head) else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cg = "" if not widths else "<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>"
    return f'<table class="{cls}">{cg}{thead}<tbody>{body}</tbody></table>'


def op_row(label, r):
    return [label, f'{r["recall"]:.2f}', f'{r["precision"]:.2f}', f'{r["f1"]:.2f}', f'{r["hit_at_1"]:.2f}', pct(r["false_retrieval_rate"])]


def best_f1(rows):
    return max(rows, key=lambda r: (r["f1"], -r["tau"]))


sw = R["sweep"]
op = lambda name, tau: next(r for r in sw[name] if abs(r["tau"] - tau) < 1e-9)
best1p, best3p, bestdf = best_f1(sw["dense_1p"]), best_f1(sw["dense_3p"]), best_f1(sw["tfidf_1p"])
D = {r["tau"]: r for r in R["dedup_sweep"]}
d1, d3 = R["distributions"]["dense_1p"], R["distributions"]["dense_3p"]
neg_max_1p, neg_med_1p = max(d1["negative_best_scores"]), sorted(d1["negative_best_scores"])[len(d1["negative_best_scores"]) // 2]
gs, gz = G[0], G[1]
v1, v2 = E[0], E[1]

retrieval_table = table(
    ["Retriever / queries", "τ", "Recall@3", "Precision", "F1", "Hit@1", "Unrelated queries that wrongly retrieve a fact"],
    [
        ["Dense MiniLM, 1st-person (as spoken)", "0.60 (original plan)"] + op_row("", op("dense_1p", 0.6))[1:],
        ["Dense MiniLM, 1st-person (as spoken)", "0.15 (adopted)"] + op_row("", op("dense_1p", 0.15))[1:],
        ["Dense MiniLM, 1st-person (as spoken)", "0.20"] + op_row("", op("dense_1p", 0.2))[1:],
        ["Dense MiniLM, 1st-person (as spoken)", f"{best1p['tau']:.2f} (best F1)"] + op_row("", best1p)[1:],
        ["Dense MiniLM, 3rd-person rewording", f"{best3p['tau']:.2f} (best F1)"] + op_row("", best3p)[1:],
        ["TF-IDF keyword baseline, 1st-person", f"{bestdf['tau']:.2f} (best F1)"] + op_row("", bestdf)[1:],
    ],
    "num", ["30%", "17%", "9%", "9%", "7%", "7%", "21%"],
)

extraction_table = table(
    ["Measure (qwen2.5:3b, temperature 0)", "v1 single fact", "v2 every fact (current)"],
    [
        ["Gold facts recalled", f'{v1["gold_recalled"]} / {v1["gold_facts"]} ({pct(v1["fact_recall"])})', f'<b>{v2["gold_recalled"]} / {v2["gold_facts"]} ({pct(v2["fact_recall"])})</b>'],
        ["Utterances with every fact captured", f'{v1["utterances_fully_captured"]} / {v1["n_positive_utterances"]}', f'<b>{v2["utterances_fully_captured"]} / {v2["n_positive_utterances"]}</b>'],
        ["Two- and three-fact utterances fully captured", f'{v1["multi_fact_utterances_fully_captured"]} / {v1["n_multi_fact_utterances"]}', f'<b>{v2["multi_fact_utterances_fully_captured"]} / {v2["n_multi_fact_utterances"]}</b>'],
        ["Spurious (wrong) facts written", str(v1["spurious_facts"]), str(v2["spurious_facts"])],
        ["Chit-chat/questions wrongly stored as facts", f'{v1["negative_false_positives"]} / {v1["n_negative"]}', f'{v2["negative_false_positives"]} / {v2["n_negative"]}'],
        ["Median time per extraction call", f'{v1["median_ms"]:.0f} ms', f'{v2["median_ms"]:.0f} ms'],
    ],
    "num", ["52%", "20%", "28%"],
)


def g_row(label, s):
    return [label, pct(s["accuracy"]), f'{s["blocked_harmful"]}/{s["n_harmful"]}', f'{s["wrongly_blocked_benign"]}/{s["n_benign"]}', f'{s["median_ms"]:.0f} ms', f'{s["max_ms"]:.0f} ms']


guard_table = table(
    ["Check and prompt style", "Accuracy", "Harmful blocked", "Benign wrongly blocked", "Median latency", "Worst latency"],
    [g_row("Input rail, few-shot (current)", gs["input"]), g_row("Input rail, zero-shot", gz["input"]),
     g_row("Output rail, few-shot (current)", gs["output"]), g_row("Output rail, zero-shot", gz["output"])],
    "num", ["32%", "11%", "14%", "17%", "13%", "13%"],
)

dedup_table = table(
    ["Duplicate threshold δ", "Paraphrases merged (of 8)", "Different facts wrongly merged (of 8)"],
    [[f"{t:.2f}" + (" (current)" if abs(t - 0.92) < 1e-9 else ""), f'{D[t]["dups_caught"]}', f'{D[t]["distinct_wrongly_merged"]}'] for t in (0.75, 0.80, 0.85, 0.90, 0.92, 0.96)],
    "num", ["34%", "33%", "33%"],
)

dash_png = HERE / "figures" / "dashboard.png"
dash_img = ""
if dash_png.exists():
    dash_img = '<img src="figures/dashboard.png" alt="Live dashboard" />' if IMG_MODE else f'<img class="shot" alt="Live pipeline dashboard" src="data:image/png;base64,{base64.b64encode(dash_png.read_bytes()).decode()}">'

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; @bottom-center { content: counter(page); font: 8.5pt 'Segoe UI', sans-serif; color: #57606a; } @top-right { content: 'BCSE306L  |  Course Project Final Report'; font: 8pt 'Segoe UI', sans-serif; color: #57606a; } }
@page :first { @bottom-center { content: none; } @top-right { content: none; } }
* { box-sizing: border-box; }
html { color: #1f2328; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 10pt; line-height: 1.42; }
body { margin: 0; }
h1 { font-size: 24pt; line-height: 1.15; margin: 0 0 6pt; color: #14315c; }
h2 { font-size: 14pt; color: #14315c; margin: 15pt 0 5pt; padding-bottom: 2pt; border-bottom: 1.2pt solid #14315c; break-after: avoid; }
h3 { font-size: 10.8pt; color: #14315c; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 0 0 6pt; text-align: left; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; } li { margin-bottom: 2pt; }
code { font-family: Consolas, 'Cascadia Mono', monospace; font-size: 8.8pt; background: #f1f3f5; padding: 0 2pt; border-radius: 2pt; }
pre { font-family: Consolas, 'Cascadia Mono', monospace; font-size: 8pt; line-height: 1.32; background: #f6f8fa; border: 0.6pt solid #d8dee4; border-radius: 3pt; padding: 6pt 8pt; margin: 4pt 0 8pt; white-space: pre-wrap; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; margin: 4pt 0 8pt; font-size: 8.8pt; break-inside: avoid; }
th { background: #eaf0f8; color: #14315c; text-align: left; padding: 3.5pt 5pt; border: 0.6pt solid #c3cfe0; font-weight: 600; }
td { padding: 3pt 5pt; border: 0.6pt solid #d8dee4; vertical-align: top; }
table.num td:not(:first-child), table.num th:not(:first-child) { text-align: right; }
figure { margin: 6pt 0 8pt; break-inside: avoid; } figure svg { width: 100%; height: auto; display: block; }
figcaption { font-size: 8.8pt; color: #57606a; margin-top: 2pt; }
.cap-b { color: #1f2328; font-weight: 600; }
.ph { background: #fff3a8; padding: 0 3pt; border-radius: 2pt; font-weight: 600; }
.note { border-left: 2.5pt solid #2a78d6; background: #f3f8fe; padding: 4pt 8pt; margin: 4pt 0 8pt; font-size: 9.2pt; break-inside: avoid; }
.warn { border-left: 2.5pt solid #eb6834; background: #fef6f2; padding: 4pt 8pt; margin: 4pt 0 8pt; font-size: 9.2pt; break-inside: avoid; }
.cover { height: 252mm; display: flex; flex-direction: column; justify-content: space-between; page-break-after: always; }
.cover .tag { letter-spacing: .12em; text-transform: uppercase; font-size: 9pt; color: #57606a; }
.cover .sub { font-size: 12.5pt; color: #3d4852; margin: 0 0 14pt; }
.cover table { font-size: 10pt; } .cover td, .cover th { padding: 5pt 8pt; }
.shot { width: 100%; border: 0.6pt solid #d8dee4; border-radius: 3pt; }
.pb { page-break-before: always; }
.small { font-size: 8.8pt; color: #57606a; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; }
"""


def build() -> str:
    cover = f"""
<section class="cover">
  <div>
    <div class="tag">School of Computer Science and Engineering &nbsp;|&nbsp; BCSE306L Artificial Intelligence &nbsp;|&nbsp; Course Project (Review 3)</div>
    <div style="height:46mm"></div>
    <h1>VoiceChat: a fully local voice assistant with semantic long-term memory and LLM safety guardrails</h1>
    <p class="sub">Final project report: problem, AI formulation, implementation, experiments and conclusions</p>
    <div style="height:10mm"></div>
    {table(["", ""], [
        ["<b>Team name</b>", PH("Team name")],
        ["<b>AI syllabus module and topic</b>", PH("Module number and name, as approved in Review 1") + "<br><span class='small'>Topic covered: information retrieval and knowledge representation (embedding-based semantic memory), few-shot LLM classifiers for safety and fact extraction, and an LLM agent organised as a state graph.</span>"],
        ["<b>Code repository</b>", f'<a href="{REPO}">{REPO}</a>'],
        ["<b>Date</b>", "October 2026"],
    ], "")}
    <div style="height:6mm"></div>
    <h3>Team members</h3>
    {table(["Full name", "Registration number"], [
        ["Shubham " + PH("surname"), PH("Registration number")],
        ["Anuvansh " + PH("surname"), PH("Registration number")],
    ], "")}
  </div>
  <div class="small">Submitted as BCSE306L_TeamName_FinalReport.pdf through VTOP. Highlighted fields are to be completed by the team before upload.</div>
</section>
<div class="pagebreak"></div>
"""

    s1 = f"""
<h2>1. Problem statement and scope</h2>
<p><b>Problem.</b> Cloud voice assistants send a user&rsquo;s speech to remote servers, depend on the network, and forget the user between sessions. We built <b>VoiceChat</b>, a voice assistant in which <i>every</i> model runs on the user&rsquo;s own laptop (speech detection, speech recognition, language model, safety checks, long-term memory and speech synthesis), and which meets a conversational latency target of <b>under 2 seconds from the end of the user&rsquo;s sentence to the first spoken word</b>.</p>
<p><b>Why it matters.</b> Local execution keeps private speech and personal facts on the device. The 2-second budget is roughly the pause people tolerate in conversation. The AI problem is that three things must be achieved together with small models: the assistant must (a) remember durable facts about the user and recall them when relevant, (b) refuse harmful requests and avoid speaking harmful replies, and (c) do both without breaking the latency budget.</p>
<p><b>Inputs and outputs.</b> Input: a microphone stream (browser microphone over a WebSocket, or a local microphone). Output: synthesised speech, plus a text transcript, per-stage latency figures and a live terminal dashboard. Persistent state: a store of facts about the user that survives restarts.</p>
<p><b>Scope and boundaries.</b> In scope: the retrieval, extraction, safety-check and orchestration logic, and evidence of their quality. Out of scope: training or fine-tuning any neural model (all pretrained), barge-in (interrupting the assistant), multiple simultaneous users, and benchmarking the general knowledge accuracy of the 3-billion-parameter model. The project targets one Windows laptop with a CUDA GPU (Intel Core i9-13900HX, NVIDIA RTX 4060 Laptop, 31.7 GB RAM).</p>
"""

    s2 = f"""
<h2>2. AI formulation</h2>
<p>A voice turn is a chain of AI sub-problems. Table 1 states each one formally; sections 2.1&ndash;2.3 give the formulation of the three parts the team designed and evaluated: <b>memory retrieval</b>, <b>fact extraction</b> and <b>safety classification</b>. Section 2.4 states the real-time constraint that links them.</p>
{table(["Sub-problem", "Formulation", "Model that solves it"], [
    ["Voice activity detection", "Binary classification of each 32 ms audio window: speech / not speech, with a 700 ms silence rule to close a turn", "Silero VAD (pretrained)"],
    ["Speech recognition", "Sequence-to-sequence: audio → text", "Whisper small via faster-whisper (pretrained)"],
    ["Input / output safety", "Binary classifier c(x) ∈ {allow, block} on a user message or a spoken sentence", "Qwen2.5-3B with a few-shot prompt (team-written prompt)"],
    ["Memory retrieval", "Top-k nearest-neighbour search over fact embeddings with a score threshold", "MiniLM embeddings + FAISS (team-designed scoring policy)"],
    ["Fact extraction", "Structured generation: utterance → set of third-person facts (or NONE)", "Qwen2.5-3B with a few-shot prompt (team-written prompt and parser)"],
    ["Response generation", "Conditional text generation given system prompt, retrieved facts, history, utterance", "Qwen2.5-3B via Ollama (pretrained)"],
    ["Speech synthesis", "Text → waveform", "Kokoro-82M (pretrained)"],
], "", ["22%", "47%", "31%"])}
<p class="small" style="margin-top:-4pt"><b>Table 1.</b> The sub-problems in one voice turn.</p>

<h3>2.1 Memory retrieval as nearest-neighbour search with a threshold</h3>
<p>Let F = {{f<sub>1</sub>, &hellip;, f<sub>n</sub>}} be the stored facts, each a short third-person sentence (for example &ldquo;The user works as a data engineer.&rdquo;). An embedding function &phi; maps text to a unit vector in R<sup>384</sup>. The score of fact f for query q is the cosine similarity s(q, f) = &phi;(q) &middot; &phi;(f), since vectors are normalised. The retrieved set is</p>
<p style="text-align:center">R(q) = top-k<sub>f&isin;F</sub> &#123; f : s(q, f) &ge; &tau; &#125; &nbsp;&nbsp; with k = 3 and &tau; = 0.15.</p>
<p>The facts in R(q) are inserted into the language model&rsquo;s system prompt. A new fact f&prime; is stored only if it is not a near-duplicate: max<sub>f&isin;F</sub> s(f&prime;, f) &lt; &delta;, with &delta; = 0.92. The search is exact (a flat inner-product index), which is appropriate for a store of hundreds of facts. The design question the team had to answer is the choice of &tau; and &delta;; section 4 measures both.</p>

<h3>2.2 Fact extraction as constrained generation</h3>
<p>For a user utterance u the extractor g(u) returns a set of facts, one per output line, rewritten in third person, or the token NONE. The model is run at temperature 0 with a system prompt of definitions, NONE examples, positive examples, and a multi-fact example (few-shot prompting). The parser splits the output on newlines and writes every non-NONE line to the store. Extraction runs <i>asynchronously</i>, off the critical path, so facts from the current turn become retrievable from the next turn.</p>

<h3>2.3 Safety as binary classification</h3>
<p>Two classifiers use the same model with different prompts: c<sub>in</sub> screens the user&rsquo;s transcript before it reaches the language model; c<sub>out</sub> screens each sentence before it is spoken. Each prompt states a policy, gives three labelled examples (benign, harmful, benign) and asks &ldquo;Should the message be blocked (Yes or No)?&rdquo;. The answer is capped at 3 tokens. If c<sub>in</sub> blocks, a fixed refusal is spoken and the language model is never called. If a check fails technically, the system <i>fails open</i> (allows) so a hiccup never silences the assistant; this trade-off is discussed in section 5.</p>

<h3>2.4 The real-time constraint</h3>
<p>The agent is a state graph with state S = (messages, transcript, retrieved_facts, response_chunks) and three nodes, retrieve &rarr; generate &rarr; extract. Time to first audio is</p>
<p style="text-align:center">T<sub>first</sub> = t<sub>STT</sub> + t<sub>c_in</sub> + t<sub>retrieve</sub> + t<sub>LLM-first-token</sub> + max(t<sub>c_out</sub>, t<sub>TTS</sub>) &nbsp;&le;&nbsp; 2000 ms.</p>
<p>The max term is the result of a design decision: the first sentence&rsquo;s safety check and its speech synthesis run concurrently, because synthesising does not depend on the check&rsquo;s outcome. Extraction is excluded from T<sub>first</sub> because it runs in the background.</p>
"""

    s3 = f"""
<h2>3. Algorithm and implementation</h2>
<figure>{F("fig1_architecture", fig_architecture(), "System architecture")}<figcaption><span class="cap-b">Figure 1.</span> System architecture. Numbers give the order of a voice turn. Blue boxes are pretrained models used as provided; orange boxes are logic written by the team (often wrapping a pretrained model); the green box is the persistent store.</figcaption></figure>

<h3>3.1 One conversational turn</h3>
<pre>procedure RUN_TURN():
    audio &larr; read mic windows through VAD until 700 ms of silence follows speech
    start latency clock
    text &larr; STT(audio);   if text is empty: return
    if BLOCK(c_in(text)): speak a fixed refusal and return        // the LLM is never called
    facts &larr; RETRIEVE(text)                                     // section 2.1
    start FALLBACK_TIMER(800 ms)      // speaks a filler phrase only if no LLM token has arrived
    run PRODUCER and CONSUMER concurrently:
        PRODUCER: for each LLM token: feed SENTENCE_CHUNKER;
                  put each complete sentence on a bounded queue (size 3)
        CONSUMER: for each sentence taken from the queue:
                      if it is the first sentence and matches a stalling pattern: drop it
                      run c_out(sentence) and TTS(sentence) concurrently
                      if allowed: play the audio, else: speak a refusal line instead
    fire and forget: EXTRACT_AND_STORE(text)                      // section 2.2
    append (text, spoken reply) to the 10-turn short-term memory</pre>

<h3>3.2 Retrieval and storage</h3>
<pre>procedure STORE_FACT(f&prime;):
    v &larr; &phi;(f&prime;)
    (s, id) &larr; index.search(v, 1)
    if s &ge; &delta;: return id                          // near-duplicate: do not store again
    id &larr; SQLite.insert(f&prime;);  index.add(v, id);  save index to disk

procedure RETRIEVE(q):
    if index is empty: return []
    v &larr; &phi;(q);  (S, IDS) &larr; index.search(v, min(k, n))
    return [ text(id) for (s, id) in zip(S, IDS) if s &ge; &tau; ]</pre>

<h3>3.3 Data structures and key design decisions</h3>
{table(["Component", "Data structure / technique", "Why"], [
    ["Fact store", "FAISS <code>IndexIDMap(IndexFlatIP)</code> holds vectors keyed by integer id; SQLite table holds id → text", "FAISS has no payload storage; exact search is fast enough at this scale and gives true cosine scores"],
    ["Sentence chunker", "Regex <code>[.!?]\\s</code> re-scanned on each token", "Starts speech before generation ends; the whitespace requirement avoids splitting decimals"],
    ["Producer/consumer queue", "<code>asyncio.Queue(maxsize=3)</code>", "Backpressure: a fast model cannot run unboundedly ahead of slow playback"],
    ["Stalling filter", "13 regular expressions applied to the first sentence only", "The 3B model sometimes opens with &ldquo;Let me think about that&hellip;&rdquo; to hide latency; prompt rules alone were not reliable"],
    ["Fallback timer", "<code>asyncio.wait_for</code> racing an <code>Event</code> set on the first LLM token", "Speaks a filler phrase only on genuinely slow turns; started after the input check so it races the LLM alone"],
    ["Guardrail dispatch", "Direct call of NeMo&rsquo;s <code>self_check_input/output</code> actions", "The documented <code>generate()</code> path silently skipped the output check in testing"],
    ["Mic gate (web)", "Flag that drops incoming audio while the assistant speaks", "Without it the assistant heard itself and replied in an endless loop"],
], "", ["20%", "40%", "40%"])}

<h3>3.4 What the team implemented, and what libraries provide</h3>
{table(["Provided by libraries or pretrained models (not trained or modified by the team)", "Implemented by the team"], [[
    "<ul><li>Silero VAD, Whisper small (faster-whisper/CTranslate2), Kokoro-82M, Qwen2.5-3B (Ollama), MiniLM-L6-v2 weights (sentence-transformers)</li><li>FAISS vector index, SQLite, NeMo Guardrails runtime, LangGraph graph runner, FastAPI/uvicorn, PyAudio</li></ul>",
    "<ul><li>Memory subsystem: embedding wrapper, FAISS+SQLite store, thresholded top-k retrieval, near-duplicate rule, threshold calibration</li><li>Extraction and safety prompts, output parser, fail-open policy, direct-dispatch guardrail manager</li><li>Turn orchestration: producer/consumer streaming with bounded queue, concurrent check+synthesis, fallback race, stalling filter, sentence chunker, latency monitor</li><li>Browser WebSocket audio path with resampling and mic gate; the live terminal dashboard</li><li>Test suite and the three evaluation scripts used in section 4</li></ul>",
]], "", ["44%", "56%"])}
<p>No model was trained. The AI contribution of the project is the <i>formulation, calibration and integration</i> of pretrained components under a latency constraint, and the measurement of the result.</p>
"""

    s4 = f"""
<h2>4. Experiments and results</h2>
<p>All experiments ran on the machine described in section 1, with Ollama serving qwen2.5:3b locally. Test sets were written by the team (section 5 discusses the limits of that). Everything here is reproducible with the commands in Appendix A, and the raw numbers are saved in <code>docs/submission/results/</code>.</p>

<h3>Experiment 1: memory retrieval, comparing retrievers, phrasing and thresholds</h3>
<p><b>Setup.</b> A store of {R["n_facts"]} facts, {R["n_positive"]} answerable queries (each with one gold fact) and {R["n_negative"]} unrelated queries that should retrieve nothing; top-k = {R["top_k"]}. We compare (i) dense MiniLM retrieval, the implemented design, (ii) the same retriever when queries are reworded into third person, and (iii) a from-scratch TF-IDF keyword baseline with no pretrained model. Thresholds from 0 to 0.8 were swept. Recall@3 is the share of answerable queries whose gold fact is returned; precision is the share of returned facts that are gold; the last column is the share of unrelated queries that wrongly retrieve something.</p>
<figure>{F("fig2_threshold_sweep", fig_threshold_sweep(), "Threshold sweep")}<figcaption><span class="cap-b">Figure 2.</span> (a) Trade-off curves for the implemented retriever. (b) F1 across thresholds for three retrievers/phrasings. The dashed lines mark the originally planned threshold (0.60) and the adopted one (0.15).</figcaption></figure>
{retrieval_table}
<p class="small" style="margin-top:-4pt"><b>Table 2.</b> Retrieval results at selected thresholds.</p>
<figure>{F("fig3_score_strips", fig_score_strips(), "Score distributions")}<figcaption><span class="cap-b">Figure 3.</span> Cosine score of the correct fact (blue) and of the best fact for an unrelated query (orange), one dot per query, for first-person and third-person phrasing. A threshold must sit between the two groups; for spoken (first-person) queries the groups overlap.</figcaption></figure>
<p><b>Findings.</b></p>
<ul>
<li><b>The originally planned threshold of 0.60 would have broken memory.</b> At &tau; = 0.60 recall is {op("dense_1p", 0.6)["recall"]:.2f}: only {round(op("dense_1p", 0.6)["recall"] * R["n_positive"])} of {R["n_positive"]} questions retrieve their fact. Facts are stored in third person but people ask in first person; first-person queries score a median of {sorted(d1["positive_gold_scores"])[len(d1["positive_gold_scores"]) // 2]:.2f} against the correct fact, versus {sorted(d3["positive_gold_scores"])[len(d3["positive_gold_scores"]) // 2]:.2f} when the same question is written in third person (Figure 3).</li>
<li><b>Dense retrieval clearly beats keyword retrieval</b> on this task: best F1 {best1p["f1"]:.2f} (&tau; = {best1p["tau"]:.2f}) against {bestdf["f1"]:.2f} for TF-IDF, whose recall never exceeds {max(r["recall"] for r in sw["tfidf_1p"] if r["false_retrieval_rate"] < 0.2):.2f} while keeping false retrievals low. &ldquo;What is my job?&rdquo; shares no content word with &ldquo;The user works as a data engineer&rdquo;, so a keyword method cannot connect them.</li>
<li><b>The adopted &tau; = 0.15 maximises recall, not balance.</b> It retrieves the right fact for every answerable query (recall {op("dense_1p", 0.15)["recall"]:.2f}) but also returns a fact for {pct(op("dense_1p", 0.15)["false_retrieval_rate"])} of unrelated queries (precision {op("dense_1p", 0.15)["precision"]:.2f}). The best F1 on this test set is at &tau; = {best1p["tau"]:.2f} (F1 {best1p["f1"]:.2f}, recall {best1p["recall"]:.2f}, false retrievals {pct(best1p["false_retrieval_rate"])}); &tau; = 0.20 is nearly as good (F1 {op("dense_1p", 0.2)["f1"]:.2f}, recall {op("dense_1p", 0.2)["recall"]:.2f}, false retrievals {pct(op("dense_1p", 0.2)["false_retrieval_rate"])}). The design recorded earlier (unrelated queries scoring 0.0&ndash;0.09) was measured on fewer queries; on this larger set unrelated queries reach {neg_max_1p:.2f}.</li>
</ul>
<div class="note"><b>Why the team kept 0.15 for now.</b> A wrongly retrieved fact is injected as &ldquo;things you remember about the user&rdquo; and is usually ignored by the model, whereas a missed fact makes the assistant appear forgetful. The data above shows that a value near 0.20 would be a better balance and is a one-line change in <code>config.yaml</code>; we list it as the first improvement in section 5.</div>

<h3>Experiment 2: the near-duplicate threshold</h3>
<p>Eight paraphrase pairs that should merge (for example &ldquo;works as a data engineer&rdquo; and &ldquo;is a data engineer&rdquo;) and eight pairs of genuinely different facts (for example &ldquo;lives in Mumbai&rdquo; and &ldquo;lives in Delhi&rdquo;) were scored.</p>
{dedup_table}
<p class="small" style="margin-top:-4pt"><b>Table 3.</b> Duplicate rule at different thresholds.</p>
<p>The current &delta; = 0.92 never merges different facts but misses {D[0.92]["dups_total"] - D[0.92]["dups_caught"]} of {D[0.92]["dups_total"]} paraphrases, so some duplicates are stored. That is harmless for retrieval (extra copies of a true fact). A value of 0.80&ndash;0.85 merges more paraphrases with no false merges on this set, but the closest different pair (&ldquo;dog named Bruno&rdquo; vs &ldquo;cat named Bruno&rdquo;) scores 0.79, so we kept the conservative value.</p>

<h3>Experiment 3: fact-extraction prompt, single fact vs every fact</h3>
<p>Live testing showed the assistant remembered a user&rsquo;s job but never their name when both were said in one sentence. We compared the original prompt (commit 876a046, &ldquo;extract only the single most important fact&rdquo;) with the current one on {v2["n_positive_utterances"]} utterances containing {v2["gold_facts"]} gold facts, plus {v2["n_negative"]} utterances that must yield NONE. Sentences used in the prompt&rsquo;s own examples were excluded.</p>
<figure>{F("fig4_extraction", fig_extraction(), "Extraction prompt comparison")}<figcaption><span class="cap-b">Figure 4.</span> Extraction quality for the two prompts (same model, temperature 0, same test utterances).</figcaption></figure>
{extraction_table}
<p class="small" style="margin-top:-4pt"><b>Table 4.</b> Extraction results.</p>
<p><b>Findings.</b> The change from &ldquo;one fact&rdquo; to &ldquo;every fact&rdquo; raised fact recall from {pct(v1["fact_recall"])} to {pct(v2["fact_recall"])} and fixed {v2["multi_fact_utterances_fully_captured"]} of {v2["n_multi_fact_utterances"]} multi-fact utterances, with no increase in wrong facts and no chit-chat stored. The original prompt also returned NONE for several single-fact statements (a brother&rsquo;s visit, a favourite language, waking time, a new job); the new prompt fixed the brother&rsquo;s visit but still misses {v2["gold_facts"] - v2["gold_recalled"]} of {v2["gold_facts"]} facts (a favourite language, a new job, waking time, and one of two facts in &ldquo;I&rsquo;m a teacher and I love hiking&rdquo;). The extractor is therefore useful but not reliable; this is the main accuracy limit of the memory.</p>

<h3>Experiment 4: safety checks, few-shot vs zero-shot</h3>
<p>The same two prompts with and without their three examples were run on {gs["input"]["n"]} user messages (half harmful: jailbreaks, weapons, malware, system-prompt extraction) and {gs["output"]["n"]} candidate spoken sentences. All test sentences are new, not the prompt examples. Benign cases deliberately include wording that looks dangerous (&ldquo;How do I kill a Python process?&rdquo;).</p>
{guard_table}
<p class="small" style="margin-top:-4pt"><b>Table 5.</b> Safety-check accuracy and latency (warm). The latencies are lower than the 324&ndash;328 ms recorded in <code>docs/latency_report.md</code> in July; the cause (the 3-token answer cap added since, or less GPU contention here) was not isolated.</p>
<p><b>Findings.</b> Both variants block every harmful user message ({gs["input"]["blocked_harmful"]}/{gs["input"]["n_harmful"]}) and both wrongly block the same benign message, &ldquo;How do I kill a Python process that is stuck?&rdquo;, a keyword-style failure of a 3B model. Few-shot examples helped on the output side: the zero-shot prompt let a system-prompt leak through ({gz["output"]["blocked_harmful"]}/{gz["output"]["n_harmful"]} harmful sentences blocked against {gs["output"]["blocked_harmful"]}/{gs["output"]["n_harmful"]}). During development the zero-shot prompt also let an explicit bomb-making sentence through; that case is one of the prompt&rsquo;s own examples now, so it is not part of this test. The difference measured here is small (one sentence in {gs["output"]["n"]}) and should be read as indicative, not conclusive. The existing adversarial suite (<code>tests/test_guardrails.py</code>, 19 cases) passes in full.</p>

<h3>Experiment 5: end-to-end latency</h3>
<p>Per-stage timings come from the live dashboard run of 2026-10-27, which spoke scripted questions into the real pipeline (real VAD, STT, guardrails, LLM, TTS; only the microphone was replaced by synthesised speech).</p>
<figure>{F("fig5_latency", fig_latency(), "Per-turn latency")}<figcaption><span class="cap-b">Figure 5.</span> Time from end of speech to first audio, by stage. Turn 3 (a blocked prompt-injection attempt) has no first audio by design: the refusal is spoken after the input check at 758 ms.</figcaption></figure>
{table(["Measurement", "Result", "Source"], [
    ["Time to first audio, steady state, before overlap optimisation", "2.6&ndash;2.8 s", "docs/latency_report.md, Part 7"],
    ["Time to first audio after running the output check and TTS concurrently", "2229 ms median (min 2115, max 2357; 5 runs)", "docs/latency_report.md, Part 9"],
    ["Dashboard run, turns 2 and 4 (warm, memory recall on turn 4)", "1279 ms and 1681 ms, both within budget", "this report, Figure 5"],
    ["Dashboard run, turn 1 (first turn after start-up)", "2130 ms, over budget (slow STT: 882 ms)", "this report, Figure 5"],
], "", ["46%", "30%", "24%"])}
<p class="small" style="margin-top:-4pt"><b>Table 6.</b> Latency summary.</p>
<p><b>Findings.</b> The two warm turns timed in the October dashboard run met the budget (1.3 s and 1.7 s) and the first turn after start-up did not (2.1 s); the earlier five-run benchmark from July (2.2 s median) also exceeded it. We have not repeated a controlled multi-run benchmark since, so we do not claim the budget is met reliably. Running the output check and speech synthesis concurrently saved about 260&ndash;400 ms per turn at no safety cost. The remaining time is dominated by two guardrail model calls and one generation call sharing a single laptop GPU with several other resident models.</p>

<h3>Correctness testing</h3>
<p>The automated suite uses real models, not mocks. At the end of development the full suite passed (82/82 recorded in <code>docs/ai_usage.md</code>), and the stalling-filter tests were added afterwards. For this report the fast suites (memory, pipeline queue and stalling filter, latency monitor, guardrails) were re-run: <b>46 passed in 25.7 s</b>. End-to-end tests feed real synthesised speech through the unmodified pipeline: a benign question gives a correct spoken answer, and an adversarial question is blocked with no LLM call.</p>
"""

    s5 = f"""
<h2>5. Discussion and conclusion</h2>
<p><b>Interpretation.</b> The experiments show where the design choices matter most. The retrieval threshold decides whether memory works at all: an off-the-shelf value (0.6) gives 4% recall for spoken queries, because the wording of stored facts differs from the wording of questions. Semantic embeddings are necessary here: a keyword baseline fails whenever the question and the fact share no words. The extraction prompt is the weakest link: even after the fix, {v2["gold_facts"] - v2["gold_recalled"]} of {v2["gold_facts"]} facts were missed, so the memory is only as good as that classifier. The safety checks were accurate on our test set, with one repeatable false alarm.</p>
<p><b>Correctness.</b> Each AI component was tested in isolation and then in the full pipeline, and several defects were found only in the full pipeline: the self-check occasionally produced a 5-second paragraph instead of Yes/No (fixed with a 3-token cap), the first version of the streaming loop lost overlap between generation and speech, a lost memory write when the server stopped quickly (fixed by waiting for background tasks), and an echo loop in the browser path (fixed by gating the microphone).</p>
<p><b>Limitations and failure cases.</b></p>
<ul>
<li><b>Calibration.</b> &tau; = 0.15 returns an irrelevant fact for {pct(op("dense_1p", 0.15)["false_retrieval_rate"])} of unrelated queries on our test set. The score ranges of relevant and irrelevant queries overlap (Figure 3), so no threshold is perfect.</li>
<li><b>Fact updates.</b> The store has no notion of contradiction. If the user moves from Mumbai to Delhi both facts are kept (they score only 0.74 as a pair), and the older fact can be retrieved.</li>
<li><b>Extraction reliability.</b> A 3B model misses some personal facts and is sensitive to prompt wording and example order.</li>
<li><b>Safety scope.</b> The classifier flags keywords such as &ldquo;kill&rdquo; out of context, a 3B model can be fooled by determined prompt injection, and the system fails open on technical errors. It is a demo-grade guard, not a moderation system.</li>
<li><b>Answer accuracy.</b> A 3B local model knows less than a large cloud model. We did not benchmark factual accuracy; the safeguard is that short answers (one to three sentences) and the &ldquo;say you don&rsquo;t know&rdquo; instruction limit the room for long wrong answers.</li>
<li><b>Latency.</b> The 2-second budget was met on some warm turns and missed on others (and on the first turn after start-up) on this laptop; a larger controlled benchmark is still needed.</li>
<li><b>Test sets are small and written by the team,</b> and measured on one machine; the figures show direction and magnitude, not statistical significance.</li>
</ul>
<p><b>Possible improvements.</b> (1) Move &tau; to about 0.20 and add a second-stage re-ranker or a hybrid keyword+dense score. (2) Store a timestamp and let a newer fact supersede an older one on the same attribute. (3) Replace the Yes/No guard with a purpose-trained safety classifier or a larger judge model only on borderline cases. (4) Collect a larger labelled set from real conversations and report confidence intervals. (5) Move one lightweight model to CPU to reduce GPU contention.</p>
<p><b>Conclusion.</b> We built and measured a fully local voice assistant whose AI core is a semantic long-term memory, a few-shot extraction and safety layer, and a latency-aware agent graph. The measurements show that embedding-based retrieval is essential, that a naive threshold would have disabled the memory, that listing every fact in the extraction prompt raised recall from 28% to 78%, and that overlapping safety checking with speech synthesis recovers 260&ndash;400 ms per turn. The system met the 2-second target on the two warm turns timed in October but not in the earlier five-run benchmark, so we report the latency goal as partly met.</p>
"""

    s6 = f"""
<h2>6. Team contributions and references</h2>
<h3>6.1 Contributions</h3>
{table(["Member", "Contribution"], [
    ["Shubham", "Set up the project environment (the uv project, CUDA builds of PyTorch and faster-whisper, the Ollama model and <code>config.yaml</code>) and built the input and language side of the assistant: the audio front end (Silero voice activity detection with the 700 ms turn-end rule, faster-whisper speech-to-text, the microphone and audio manager), the LangGraph agent (state, prompts, the retrieve &rarr; generate &rarr; extract graph with token streaming from Qwen2.5-3B) and the rolling short-term memory. Built the long-term memory: the embedding wrapper, the FAISS and SQLite fact store, thresholded top-k retrieval, the near-duplicate rule and the recalibration of the similarity threshold from 0.6 to 0.15, together with the few-shot fact-extraction prompt and parser, the background write that survives shutdown, and the fix for dropping all but one fact per sentence. Wrote the tests and benchmarks for these parts, ran Experiments 1&ndash;3 (retrieval against the TF-IDF baseline, the duplicate threshold, the extraction prompt comparison) and wrote report sections 1, 2.1 and 2.2 and the memory and agent parts of section 3."],
    ["Anuvansh", "Built the output, safety and real-time side and integrated the whole system: Kokoro text-to-speech with sentence chunking and the sentence-streaming player; the NeMo guardrails layer (few-shot input and output checks, the 3-token answer cap, the direct-dispatch manager and the fail-open policy); the fallback filler-phrase timer and the latency monitor; and the full pipeline (producer/consumer stream with a bounded queue, running the output check and speech synthesis together, the stalling-phrase filter and the fix that starts the fallback timer after the input check). Built the FastAPI and WebSocket web demo with its browser microphone path, resampling, the gate that stops the assistant hearing itself, and the interface with its audio visualiser, plus the live animated terminal and browser dashboard, the launchers and the desktop shortcut. Wrote the end-to-end tests, ran Experiments 4 and 5 (safety checks, latency) and the GPU-contention analysis, and wrote report sections 2.3, 2.4, 5 and the pipeline parts of section 3."],
    ["Both", "Wrote the requirements and the Review 1 and 2 material; ran the full test suite and the live voice tests; wrote the README, the design-trade-off and latency documents and the generative-AI disclosure; prepared the live demonstration; and reviewed the final report."],
], "", ["22%", "78%"])}

<h3>6.2 Generative-AI assistance (disclosure)</h3>
<p>Claude (Anthropic), used through Claude Code, assisted throughout this project at the team&rsquo;s request, one part at a time: architecture planning, writing much of the source code and tests, debugging, writing the documentation in <code>docs/</code>, the live terminal dashboard (<code>terminal_viz.py</code>, <code>run_viz.py</code>, <code>viz_web.py</code>), the three evaluation scripts in <code>scripts/</code>, and the drafting of this report and its figures. A per-part log is in <code>docs/ai_usage.md</code>. The experiments were run on the team&rsquo;s own machine and every number in section 4 is produced by those scripts and saved in <code>docs/submission/results/</code>. Test sets, design decisions and conclusions were reviewed by the team.</p>

<h3>6.3 References</h3>
<ol class="small" style="font-size:8.8pt">
<li>Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. <i>EMNLP-IJCNLP</i>.</li>
<li>Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. and Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. <i>NeurIPS</i>. (Model used: sentence-transformers/all-MiniLM-L6-v2.)</li>
<li>Johnson, J., Douze, M. and J&eacute;gou, H. (2019). Billion-scale similarity search with GPUs. <i>IEEE Transactions on Big Data</i> (FAISS).</li>
<li>Radford, A. et al. (2023). Robust speech recognition via large-scale weak supervision. <i>ICML</i> (Whisper). Run through faster-whisper / CTranslate2.</li>
<li>Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.</li>
<li>Brown, T. et al. (2020). Language models are few-shot learners. <i>NeurIPS</i>.</li>
<li>Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>NeurIPS</i>.</li>
<li>Rebedea, T. et al. (2023). NeMo Guardrails: A toolkit for controllable and safe LLM applications with programmable rails. <i>EMNLP 2023, System Demonstrations</i>.</li>
<li>Manning, C. D., Raghavan, P. and Sch&uuml;tze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press (TF-IDF, cosine similarity).</li>
<li>Russell, S. and Norvig, P. (2021). <i>Artificial Intelligence: A Modern Approach</i>, 4th ed. Pearson.</li>
<li>Software: Silero VAD (github.com/snakers4/silero-vad); Kokoro-82M (huggingface.co/hexgrad/Kokoro-82M); Ollama (ollama.com); LangGraph (langchain-ai.github.io/langgraph); FastAPI; PyTorch; SQLite.</li>
</ol>
"""

    appx = f"""
<h2 class="pb">Appendix A. Running the AI core and reproducing the results</h2>
<p>Source code and test data are in the repository <a href="{REPO}">{REPO}</a>; they can also be demonstrated live on the team&rsquo;s laptop. Requirements: Windows, an NVIDIA GPU, <code>ollama serve</code> running with <code>qwen2.5:3b</code> pulled, and the project&rsquo;s virtual environment.</p>
<pre># start the assistant: browser voice UI on the left, live pipeline terminal on the right
.venv\\Scripts\\python.exe run_viz.py            # then open http://127.0.0.1:8000/

# run the AI components' tests (real models, no mocks)
.venv\\Scripts\\python.exe -m pytest tests/test_memory.py tests/test_pipeline.py tests/test_latency_monitor.py tests/test_guardrails.py

# reproduce the experiments in section 4 (each writes JSON to docs/submission/results/)
.venv\\Scripts\\python.exe scripts/eval_retrieval.py
.venv\\Scripts\\python.exe scripts/eval_extraction.py
.venv\\Scripts\\python.exe scripts/eval_guardrails.py

# rebuild this report's HTML from the saved results
python docs/submission/build_report.py</pre>
<h2>Appendix B. Live demonstration view</h2>
<p>The dashboard animates every stage of a turn (microphone, VAD probability, transcript, guardrail verdicts, retrieved facts, token stream, sentence queue, latency bar against the 2 s budget) and is what the team uses to explain the pipeline in the live demonstration. It is a monitoring aid; the evaluated AI components are those in sections 2&ndash;4.</p>
{dash_img}
<p class="small">Figure B1. The browser page: voice UI on the left, live pipeline terminal on the right, captured after a spoken question. This was a cold first turn in a headless browser (4,055 ms), which the dashboard correctly flags as OVER budget; it is shown for the layout, not as a latency result.</p>
<h2>Appendix C. Test material</h2>
<h3>C.1 Extraction test utterances and outputs (current prompt)</h3>
{table(["Utterance", "Gold facts", "Recalled", "Model output"], [[esc(r["utterance"]), r["n_gold"], r["recalled"], esc("; ".join(r["output"]) or "NONE")] for r in v2["positive_rows"]], "", ["34%", "9%", "10%", "47%"])}
<h3>C.2 Retrieval fact store used in Experiment 1</h3>
<p class="small">12 facts: the user&rsquo;s name (Jay), job (data engineer), peanut allergy, preference for short answers, sister Maya visiting, lives in Mumbai, favourite language Rust, vegetarian, plays guitar, studies computer science (master&rsquo;s), dog named Bruno, wakes at six. The 26 answerable and 15 unrelated queries are listed in <code>scripts/eval_retrieval.py</code>.</p>
<h3>C.3 Safety-check test cases</h3>
<p class="small">24 user messages and 10 spoken sentences, listed in <code>scripts/eval_guardrails.py</code>; per-case outcomes for both prompt styles are in <code>docs/submission/results/guardrails.json</code>.</p>
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VoiceChat: a fully local voice assistant with semantic long-term memory and LLM safety guardrails</title><style>{CSS}</style></head><body>
{cover}{s1}{s2}{s3}{s4}{s5}{s6}{appx}</body></html>"""


if __name__ == "__main__":
    out = HERE / ("report_docx.html" if IMG_MODE else "report.html")
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
