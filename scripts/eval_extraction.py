# Fact-extraction evaluation (BCSE306L course-project report, section 5).
#
# Compares two versions of the "is this worth remembering?" few-shot prompt, both
# run against the same local model (qwen2.5:3b via Ollama, temperature 0):
#   v1 - original prompt (git commit 876a046): "extract only the single most important fact"
#   v2 - current agent/prompts.py: "extract every distinct fact, one per line"
# on a labelled set of utterances. See docs/tradeoffs.md decisions 9 and 31 for context.
#
# A gold fact counts as recalled if any output line contains one of its keywords.
# An output line that matches no gold fact is a spurious fact; any fact output for an
# utterance that should yield NONE is a false positive.
#
# Usage:  .venv\Scripts\python.exe scripts\eval_extraction.py     (needs `ollama serve`)
# Writes: docs/submission/results/extraction.json
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import ollama
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.prompts import EXTRACTION_SYSTEM_PROMPT as V2_PROMPT, build_extraction_prompt  # noqa: E402

V1_COMMIT = "876a046"

# (utterance, [gold facts, each a list of acceptable keywords])
POSITIVE = [
    ("Hi, I'm Priya and I work as a pediatric nurse.", [["priya"], ["nurse"]]),
    ("I work as a civil engineer and my name is Omar.", [["engineer"], ["omar"]]),
    ("My name is Sara.", [["sara"]]),
    ("I'm allergic to shellfish and I live in Pune.", [["shellfish"], ["pune"]]),
    ("I'm vegan, I play the violin, and my cat is named Luna.", [["vegan"], ["violin"], ["luna"]]),
    ("Please keep your answers brief from now on, I prefer them short.", [["brief", "short"]]),
    ("My brother Arjun is visiting me next month.", [["arjun"]]),
    ("I'm studying electrical engineering at university.", [["electrical"]]),
    ("I wake up at five every morning to go running.", [["five", " 5"]]),
    ("My favorite programming language is Rust.", [["rust"]]),
    ("I just started a new job at a bank.", [["bank"]]),
    ("I'm a teacher and I love hiking.", [["teacher"], ["hiking"]]),
]
NEGATIVE = [
    "What time is it?",
    "Can you recommend a good pizza place?",
    "Thanks, that helps.",
    "Tell me a joke.",
    "What's the capital of France?",
    "How do magnets work?",
    "Okay, sounds good.",
    "Can you speak a bit louder?",
]


def v1_prompt() -> str:
    src = subprocess.run(
        ["git", "show", f"{V1_COMMIT}:agent/prompts.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    ns: dict = {}
    exec(compile(src, "old_prompts.py", "exec"), ns)  # only defines string constants + a formatter
    return ns["EXTRACTION_SYSTEM_PROMPT"]


def extract(client, model, system_prompt, utterance, keep_alive):
    t0 = time.perf_counter()
    r = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_extraction_prompt(utterance)},
        ],
        stream=False,
        options={"temperature": 0.0, "num_predict": 120, "seed": 7},
        keep_alive=keep_alive,
    )
    raw = r["message"]["content"].strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and ln.strip().upper().rstrip(".") != "NONE"]
    return lines, raw, (time.perf_counter() - t0) * 1000


def score(prompt_name, system_prompt, client, model, keep_alive):
    rows, gold_total, gold_hit, spurious, out_lines = [], 0, 0, 0, 0
    for utt, gold in POSITIVE:
        lines, raw, ms = extract(client, model, system_prompt, utt, keep_alive)
        low = [ln.lower() for ln in lines]
        hits = [any(any(k in ln for k in grp) for ln in low) for grp in gold]
        extra = [ln for ln in low if not any(any(k in ln for k in grp) for grp in gold)]
        gold_total += len(gold)
        gold_hit += sum(hits)
        spurious += len(extra)
        out_lines += len(lines)
        rows.append({"utterance": utt, "n_gold": len(gold), "recalled": sum(hits), "output": lines, "spurious": len(extra), "ms": round(ms)})
    neg_rows, fp = [], 0
    for utt in NEGATIVE:
        lines, raw, ms = extract(client, model, system_prompt, utt, keep_alive)
        fp += int(bool(lines))
        out_lines += len(lines)
        neg_rows.append({"utterance": utt, "output": lines, "false_positive": bool(lines), "ms": round(ms)})
    all_ms = sorted(r["ms"] for r in rows + neg_rows)
    n_all = len(rows) + len(neg_rows)
    full_utts = sum(1 for r in rows if r["recalled"] == r["n_gold"])
    multi = [r for r in rows if r["n_gold"] > 1]
    return {
        "prompt": prompt_name,
        "fact_recall": gold_hit / gold_total,
        "gold_facts": gold_total, "gold_recalled": gold_hit,
        "utterances_fully_captured": full_utts, "n_positive_utterances": len(rows),
        "multi_fact_utterances_fully_captured": sum(1 for r in multi if r["recalled"] == r["n_gold"]),
        "n_multi_fact_utterances": len(multi),
        "spurious_facts": spurious,
        "negative_false_positives": fp, "n_negative": len(NEGATIVE),
        "median_ms": all_ms[n_all // 2],
        "positive_rows": rows, "negative_rows": neg_rows,
    }


def main() -> None:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = cfg["llm"]
    client = ollama.Client(host=llm["base_url"])
    client.chat(model=llm["model"], messages=[{"role": "user", "content": "hi"}], keep_alive=llm.get("keep_alive"))  # warm
    results = [
        score("v1_single_fact", v1_prompt(), client, llm["model"], llm.get("keep_alive")),
        score("v2_multi_fact", V2_PROMPT, client, llm["model"], llm.get("keep_alive")),
    ]
    res = ROOT / "docs" / "submission" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "extraction.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in results:
        print(f"\n{r['prompt']}: fact recall {r['gold_recalled']}/{r['gold_facts']} = {r['fact_recall']:.2f}; "
              f"utterances fully captured {r['utterances_fully_captured']}/{r['n_positive_utterances']}; "
              f"multi-fact fully captured {r['multi_fact_utterances_fully_captured']}/{r['n_multi_fact_utterances']}; "
              f"spurious {r['spurious_facts']}; negative FP {r['negative_false_positives']}/{r['n_negative']}; median {r['median_ms']}ms")
        for row in r["positive_rows"]:
            print(f"   [{row['recalled']}/{row['n_gold']}] {row['utterance']}  ->  {row['output']}")
        for row in r["negative_rows"]:
            if row["false_positive"]:
                print(f"   [FP] {row['utterance']}  ->  {row['output']}")


if __name__ == "__main__":
    main()
