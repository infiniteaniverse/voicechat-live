# Retrieval evaluation for the long-term memory (BCSE306L course-project report, section 5).
#
# Compares two retrieval scorers over the same labelled fact store and query set:
#   dense  - all-MiniLM-L6-v2 embeddings, cosine similarity (what memory/long_term.py uses)
#   tfidf  - a from-scratch TF-IDF + cosine keyword baseline (no pretrained model)
# and sweeps the similarity threshold, including the originally planned 0.6 and the
# calibrated 0.15 (docs/tradeoffs.md decision 10). Also measures the near-duplicate
# threshold used by store_fact() (_DEDUP_THRESHOLD = 0.92).
#
# Usage:  .venv\Scripts\python.exe scripts\eval_retrieval.py
# Writes: docs/submission/results/retrieval.json
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory.embeddings import Embedder  # noqa: E402

TOP_K = 3

# ── Labelled data ────────────────────────────────────────────────────────────────
# Facts as extract_memory_node writes them: third person, starting "The user".
FACTS = [
    "The user's name is Jay.",
    "The user works as a data engineer.",
    "The user is allergic to peanuts.",
    "The user prefers short answers.",
    "The user's sister Maya is visiting them next week.",
    "The user lives in Mumbai.",
    "The user's favorite programming language is Rust.",
    "The user is vegetarian.",
    "The user plays the guitar.",
    "The user is studying for a master's degree in computer science.",
    "The user has a dog named Bruno.",
    "The user wakes up at six every morning.",
]

# (first-person voice query, third-person rewording of the same question, gold fact index)
POSITIVE = [
    ("What's my name?", "What is the user's name?", 0),
    ("Do you remember what I'm called?", "What is the user called?", 0),
    ("Who am I?", "Who is the user?", 0),
    ("What do I do for a living?", "What does the user do for a living?", 1),
    ("What's my job?", "What is the user's job?", 1),
    ("What kind of work do I do?", "What kind of work does the user do?", 1),
    ("What am I allergic to?", "What is the user allergic to?", 2),
    ("Is there any food I should avoid because of allergies?", "Is there any food the user should avoid because of allergies?", 2),
    ("How do I like my answers?", "How does the user like answers?", 3),
    ("Should you keep your replies long or short for me?", "Should replies be long or short for the user?", 3),
    ("Who is coming to visit me?", "Who is coming to visit the user?", 4),
    ("When is my sister arriving?", "When is the user's sister arriving?", 4),
    ("Where do I live?", "Where does the user live?", 5),
    ("Which city am I in?", "Which city is the user in?", 5),
    ("What programming language do I like best?", "What programming language does the user like best?", 6),
    ("Which language do I code in?", "Which language does the user code in?", 6),
    ("What kind of diet do I follow?", "What kind of diet does the user follow?", 7),
    ("Can I eat chicken?", "Can the user eat chicken?", 7),
    ("What instrument do I play?", "What instrument does the user play?", 8),
    ("Do I play any music?", "Does the user play any music?", 8),
    ("What am I studying?", "What is the user studying?", 9),
    ("What degree am I doing?", "What degree is the user doing?", 9),
    ("What's my pet's name?", "What is the user's pet's name?", 10),
    ("Do I have any animals?", "Does the user have any animals?", 10),
    ("When do I get up in the morning?", "When does the user get up in the morning?", 11),
    ("What time do I wake up?", "What time does the user wake up?", 11),
]

# Queries that must retrieve nothing (nothing in the store is relevant).
NEGATIVE = [
    ("What is the capital of France?", "What is the capital of France?"),
    ("Tell me a joke.", "Tell me a joke."),
    ("How many planets are in the solar system?", "How many planets are in the solar system?"),
    ("What's two plus two?", "What is two plus two?"),
    ("Explain how photosynthesis works.", "Explain how photosynthesis works."),
    ("Who wrote Hamlet?", "Who wrote Hamlet?"),
    ("How far away is the moon?", "How far away is the moon?"),
    ("What's the tallest mountain in the world?", "What is the tallest mountain in the world?"),
    ("How do I reset my router?", "How does the user reset a router?"),
    ("Can you help me write an email to my landlord's agent?", "Can you help write an email to a letting agent?"),
    ("How do I change a flat tire?", "How does the user change a flat tire?"),
    ("What's the difference between a virus and a bacterium?", "What is the difference between a virus and a bacterium?"),
    ("Can you recommend a good science fiction movie?", "Can you recommend a good science fiction movie?"),
    ("How do magnets work?", "How do magnets work?"),
    ("Translate 'good morning' into Spanish.", "Translate 'good morning' into Spanish."),
]

# (fact A, fact B, should_be_duplicate)
DEDUP_PAIRS = [
    ("The user's name is Jay.", "The user's name is Jay.", True),
    ("The user works as a data engineer.", "The user is a data engineer.", True),
    ("The user is allergic to peanuts.", "The user has a peanut allergy.", True),
    ("The user prefers short answers.", "The user likes brief answers.", True),
    ("The user lives in Mumbai.", "The user's home is in Mumbai.", True),
    ("The user plays the guitar.", "The user plays guitar.", True),
    ("The user has a dog named Bruno.", "The user owns a dog called Bruno.", True),
    ("The user is vegetarian.", "The user follows a vegetarian diet.", True),
    ("The user works as a data engineer.", "The user works as a machine learning engineer.", False),
    ("The user lives in Mumbai.", "The user lives in Delhi.", False),
    ("The user's name is Jay.", "The user's name is Maya.", False),
    ("The user has a dog named Bruno.", "The user has a cat named Bruno.", False),
    ("The user is allergic to peanuts.", "The user is allergic to shellfish.", False),
    ("The user plays the guitar.", "The user plays the piano.", False),
    ("The user is vegetarian.", "The user is allergic to peanuts.", False),
    ("The user wakes up at six every morning.", "The user is studying for a master's degree in computer science.", False),
]

# ── From-scratch TF-IDF baseline ─────────────────────────────────────────────────
_STOP = set(
    "a an the is are am was were be to of in on at for and or it its this that with as by "
    "do does did i me my you your we our they them their he she his her what whats which who "
    "when where how can could should would will any some about from user".split()
)


def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower().replace("'s", "")) if t not in _STOP]


class TfIdf:
    def __init__(self, docs: list[str]) -> None:
        toks = [_tok(d) for d in docs]
        df = Counter(t for ts in toks for t in set(ts))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf))}
        self.mat = np.stack([self.vec(d) for d in docs])

    def vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for t, c in Counter(_tok(text)).items():
            if t in self.vocab:
                v[self.vocab[t]] = c * self.idf[t]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def scores(self, query: str) -> np.ndarray:
        return self.mat @ self.vec(query)


# ── Metrics ──────────────────────────────────────────────────────────────────────
def evaluate(score_rows_pos, gold, score_rows_neg, tau, k=TOP_K):
    """score_rows_*: (n_queries, n_facts) similarity matrices."""
    tp = retrieved = hit1 = 0
    for s, g in zip(score_rows_pos, gold):
        order = np.argsort(-s)[:k]
        kept = [int(i) for i in order if s[i] >= tau]
        retrieved += len(kept)
        tp += int(g in kept)
        hit1 += int(bool(kept) and kept[0] == g)
    fp_q = sum(1 for s in score_rows_neg if (np.sort(s)[::-1][0] >= tau))
    n_pos, n_neg = len(gold), len(score_rows_neg)
    recall = tp / n_pos
    precision = tp / retrieved if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tau": round(float(tau), 2),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "hit_at_1": hit1 / n_pos,
        "false_retrieval_rate": fp_q / n_neg,
    }


def main() -> None:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    emb = Embedder(cfg)

    fact_vecs = np.stack([emb.embed(t) for t in FACTS])

    def dense(queries):
        return np.stack([fact_vecs @ emb.embed(q) for q in queries])

    tf = TfIdf(FACTS)

    def tfidf(queries):
        return np.stack([tf.scores(q) for q in queries])

    gold = [p[2] for p in POSITIVE]
    pos_1p = [p[0] for p in POSITIVE]
    pos_3p = [p[1] for p in POSITIVE]
    neg_1p = [n[0] for n in NEGATIVE]
    neg_3p = [n[1] for n in NEGATIVE]

    systems = {
        "dense_1p": (dense(pos_1p), dense(neg_1p)),
        "dense_3p": (dense(pos_3p), dense(neg_3p)),
        "tfidf_1p": (tfidf(pos_1p), tfidf(neg_1p)),
    }
    taus = [round(x, 2) for x in np.arange(0.0, 0.801, 0.05)]
    sweep = {name: [evaluate(p, gold, n, t) for t in taus] for name, (p, n) in systems.items()}

    # Score distributions: correct-fact score for positives vs best-fact score for negatives.
    dist = {}
    for name, (p, n) in systems.items():
        dist[name] = {
            "positive_gold_scores": [float(p[i][g]) for i, g in enumerate(gold)],
            "negative_best_scores": [float(s.max()) for s in n],
        }

    # Dedup threshold
    dedup_scores = []
    for a, b, dup in DEDUP_PAIRS:
        dedup_scores.append({"a": a, "b": b, "duplicate": dup, "cosine": float(emb.embed(a) @ emb.embed(b))})
    dedup_sweep = []
    for t in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        pred = [d["cosine"] >= t for d in dedup_scores]
        truth = [d["duplicate"] for d in dedup_scores]
        tp = sum(p and y for p, y in zip(pred, truth))
        fp = sum(p and not y for p, y in zip(pred, truth))
        fn = sum((not p) and y for p, y in zip(pred, truth))
        dedup_sweep.append({"tau": t, "dups_caught": tp, "dups_total": sum(truth),
                            "distinct_wrongly_merged": fp, "distinct_total": len(truth) - sum(truth), "missed": fn})

    out = {
        "n_facts": len(FACTS), "n_positive": len(POSITIVE), "n_negative": len(NEGATIVE), "top_k": TOP_K,
        "sweep": sweep, "distributions": dist, "dedup_pairs": dedup_scores, "dedup_sweep": dedup_sweep,
        "operating_points": {
            name: {"tau_0.60": next(r for r in rows if r["tau"] == 0.6),
                   "tau_0.15": next(r for r in rows if r["tau"] == 0.15)}
            for name, rows in sweep.items()
        },
    }
    res = ROOT / "docs" / "submission" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "retrieval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{len(FACTS)} facts, {len(POSITIVE)} positive + {len(NEGATIVE)} negative queries, top_k={TOP_K}\n")
    for name, rows in sweep.items():
        print(name)
        print("  tau   recall  prec    F1    hit@1  false-retrieval")
        for r in rows:
            print(f"  {r['tau']:.2f}  {r['recall']:.3f}  {r['precision']:.3f}  {r['f1']:.3f}  {r['hit_at_1']:.3f}  {r['false_retrieval_rate']:.3f}")
    print("\nscore distributions (min/median/max)")
    for name, d in dist.items():
        for key, vals in d.items():
            print(f"  {name:9s} {key:22s} {min(vals):.3f} / {float(np.median(vals)):.3f} / {max(vals):.3f}")
    print("\ndedup pair cosine scores")
    for d in dedup_scores:
        print(f"  {d['cosine']:.3f}  dup={d['duplicate']!s:5}  {d['a']}  |  {d['b']}")
    print("\ndedup sweep")
    for r in dedup_sweep:
        print(" ", r)


if __name__ == "__main__":
    main()
