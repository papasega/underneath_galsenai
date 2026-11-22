"""Metrics shared by the Gemma and galsenai evaluations (pure Python, no dependencies)."""

import json
import math
import re
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 on lowercase word tokens (longest common subsequence)."""
    p, r = words(pred), words(ref)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    return 0.0 if lcs == 0 else 2 * lcs / (len(p) + len(r))


def token_f1(pred: str, ref: str) -> float:
    p, r = words(pred), words(ref)
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def distinct2(text: str) -> float:
    """Share of unique word bigrams: low values flag repetition loops."""
    w = words(text)
    bi = list(zip(w, w[1:], strict=False))
    return len(set(bi)) / len(bi) if bi else 0.0


def summarize(rows: list[dict]) -> dict:
    """rows: one per assistant turn with nll (nats, summed), n_tokens, ref_bytes, pred, ref."""
    n = len(rows)
    def mean(f):
        return sum(f(r) for r in rows) / n
    nll, toks, nbytes = (sum(r[k] for r in rows) for k in ("nll", "n_tokens", "ref_bytes"))
    return {
        "n_turns": n,
        "bits_per_byte": nll / (math.log(2) * nbytes),
        "nll_per_token": nll / toks,
        "perplexity": math.exp(nll / toks),
        "rouge_l": mean(lambda r: rouge_l(r["pred"], r["ref"])),
        "token_f1": mean(lambda r: token_f1(r["pred"], r["ref"])),
        "distinct_2": mean(lambda r: distinct2(r["pred"])),
        "length_ratio": mean(lambda r: len(words(r["pred"])) / max(len(words(r["ref"])), 1)),
        "empty_rate": mean(lambda r: not r["pred"].strip()),
    }


def assistant_turns(path: Path) -> list[tuple[list[dict], dict]]:
    """(history before the turn, reference assistant message) for every assistant turn of a JSONL file."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        out += [(msgs[:i], m) for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return out
