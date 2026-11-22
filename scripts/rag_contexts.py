"""Build the RAG contexts for the validation turns and score retrieval (output: results/local/, git-ignored).

For each held-out assistant turn the query is the last user message. Retrievers: bm25, dense (embeddinggemma-2),
hybrid (RRF) and `oracle` = the passage whose content words best cover the reference answer (upper bound: a good chunk).
Retrieval metrics against the oracle passage: hit@1/3/5, MRR, and context recall (share of the reference's content words
present in the 4 retrieved passages). `--sweep` repeats the scoring for several chunk sizes.

    .venv/bin/python scripts/rag_contexts.py [--sweep]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp"))
sys.path.insert(0, str(Path(__file__).parent))
import rag  # noqa: E402
from eval_metrics import assistant_turns  # noqa: E402

METHODS = ("bm25", "dense", "hybrid")


def content_words(s: str) -> set[str]:
    return {t for t in rag.tokens(s) if len(t) > 3}


def oracle_id(ref: str) -> int:
    """Passage covering most of the reference's content words (ties: shorter passage)."""
    want = content_words(ref)
    return max(
        range(len(rag._index()["passages"])),
        key=lambda i: (len(want & content_words(rag._index()["passages"][i]["text"])), -len(rag._index()["passages"][i]["text"])),
    )


def score(turns: list[tuple[list[dict], dict]], k_ctx: int = 4) -> tuple[dict, dict]:
    """Returns (metrics per method, contexts per method)."""
    passages = rag._index()["passages"]
    metrics: dict = {}
    ctx: dict = {m: [] for m in (*METHODS, "oracle")}
    ranks: dict = {m: [] for m in METHODS}
    recall: dict = {m: [] for m in (*METHODS, "oracle")}
    for history, ref in turns:
        q = rag.query_from(history)
        gold = oracle_id(ref["content"])
        want = content_words(ref["content"])
        for m in METHODS:
            top = rag.retrieve(q, 100, m)  # 100 deep to read the oracle rank
            ids = [p["id"] for p in top]
            ranks[m].append(ids.index(gold) + 1 if gold in ids else None)
            got = top[:k_ctx]
            ctx[m].append(rag.build_prompt(history[-1]["content"], got))
            recall[m].append(len(want & content_words(" ".join(p["text"] for p in got))) / max(len(want), 1))
        ctx["oracle"].append(rag.build_prompt(history[-1]["content"], [passages[gold]]))
        recall["oracle"].append(len(want & content_words(passages[gold]["text"])) / max(len(want), 1))
    n = len(turns)
    for m in METHODS:
        r = ranks[m]
        metrics[m] = {
            "hit@1": sum(x == 1 for x in r) / n,
            "hit@3": sum(x is not None and x <= 3 for x in r) / n,
            "hit@5": sum(x is not None and x <= 5 for x in r) / n,
            "mrr": sum(1 / x for x in r if x) / n,
            "context_recall": sum(recall[m]) / n,
        }
    metrics["oracle"] = {"context_recall": sum(recall["oracle"]) / n}
    return metrics, ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    turns = assistant_turns(ROOT / "data/private/mlx_sft/valid.jsonl")
    out = ROOT / "results/local"
    out.mkdir(parents=True, exist_ok=True)
    metrics, ctx = score(turns)
    (out / "rag_ctx.json").write_text(json.dumps({"retrieval": metrics, "contexts": ctx}, ensure_ascii=False, indent=1))
    print(json.dumps(metrics, indent=1))
    if a.sweep:
        sweep = {}
        for w, s in ((2, 1), (4, 2), (8, 4)):
            rag.WINDOW, rag.STRIDE = w, s
            rag.reset()
            sweep[f"window={w},stride={s}"] = {"passages": len(rag._index()["passages"]), **score(turns)[0]}
        (out / "rag_chunk_sweep.json").write_text(json.dumps(sweep, indent=1))
        print(json.dumps(sweep, indent=1))
