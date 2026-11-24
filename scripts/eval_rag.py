"""Local RAG evaluation (output goes to results/local/, git-ignored).

1. Retrieval: hit@k of bm25 / dense (embeddinggemma-2) / hybrid on hand-written paraphrased questions; a hit means a
   top-k passage matches the regex of what the answer must contain. Off-topic questions are listed for manual reading.
2. Answers (--answers, needs ./scripts/serve.sh running): per RAG mode, groundedness = share of the answer's content
   tokens present in the retrieved passages, citation rate ([n]) and refusal rate on off-topic questions.

    .venv/bin/python scripts/eval_rag.py [--answers] [--k 5]
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp"))
import rag  # noqa: E402

# (question, regex the right passage must contain). Paraphrased on purpose: little exact word overlap with the chat.
IN_TOPIC = [
    ("Quelle différence entre le fichier README et le fichier SKILL pour un agent ?", r"skill\.md"),
    ("Qui est le chercheur dont on admire les idées originales sur les LLM ?", r"karpathy"),
    ("Combien de cartes graphiques a le supercalculateur du pays ?", r"v100"),
    ("De quel matériel a-t-on besoin pour déployer un modèle de reconnaissance vocale en wolof ?", r"\bh100|asr"),
    ("Qui a suggéré de résoudre les problèmes mathématiques ouverts avec l'IA ?", r"terence tao"),
    ("Quelle vidéo de presque 20 heures recommande-t-on pour apprendre ?", r"20 heures"),
    ("Quel langage compile vite et est présenté comme alternative à Python pour l'IA ?", r"mojo"),
    ("Que pense le groupe des modèles qui ont piraté un vrai système pendant un test ?", r"mythos|hack"),
    ("Quel opérateur télécom est cité pour une panne ou un sabotage d'appels ?", r"sonatel"),
    ("Pourquoi ne pas simplement utiliser des modèles open source en local plutôt que payer un abonnement ?", r"gpu.*(open|abonn|plan)|plan de 100"),
    ("Quel appel à voter est lancé pour un projet de site de vente ?", r"sunusouba"),
    ("Quel langage fonctionnel un membre dit-il avoir appris sans voir son intérêt en IA ?", r"haskell"),
]
OFF_TOPIC = [
    "Qui a gagné la coupe du monde 2018 ?",
    "Donne-moi la recette du thiéboudienne.",
    "Quelle est la capitale de la Mongolie ?",
    "Quelle est la météo à Dakar demain ?",
]


def match(p: dict, rx: str) -> bool:
    return bool(re.search(rx, p["text"], re.I | re.S))


def retrieval(k: int) -> dict:
    out = {}
    for method in ("bm25", "dense", "hybrid"):
        hits, ranks = 0, []
        for q, rx in IN_TOPIC:
            got = rag.retrieve(q, k, method)
            r = next((i + 1 for i, p in enumerate(got) if match(p, rx)), None)
            hits += r is not None
            ranks.append(r)
        out[method] = {"hit@%d" % k: hits / len(IN_TOPIC), "first_hit_rank": ranks}
    return out


def ask(q: str, method: str, base: str) -> str:
    body = json.dumps({"model": "gemma", "rag": method, "messages": [{"role": "user", "content": q}], "max_tokens": 1500}).encode()
    text = ""
    with urllib.request.urlopen(urllib.request.Request(base + "/api/chat", body, {"Content-Type": "application/json"}), timeout=600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                ev = json.loads(line[6:])
                text += (ev.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
    return text


def answers(base: str) -> dict:
    out = {}
    for method in ("off", "bm25", "dense", "hybrid"):
        ground, cited, refused = [], 0, 0
        for q, _ in IN_TOPIC:
            a = ask(q, method, base)
            ctx = set(rag.tokens(" ".join(p["text"] for p in rag.retrieve(q, 4, method or "bm25"))) if method != "off" else [])
            toks = [t for t in rag.tokens(a) if len(t) > 3]
            ground.append(sum(t in ctx for t in toks) / max(len(toks), 1))
            cited += bool(re.search(r"\[\d\]", a))
        for q in OFF_TOPIC:
            refused += bool(re.search(r"n'en a pas parl|pas (de )?(mention|inform)|ne (parle|dit|contient)|pas dans", ask(q, method, base), re.I))
        out[method] = {"groundedness": round(sum(ground) / len(ground), 3), "cited": cited / len(IN_TOPIC), "refusal_off_topic": refused / len(OFF_TOPIC)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--base", default="http://localhost:8080")
    a = ap.parse_args()
    print((ROOT / "config/dataset_notice.txt").read_text().strip(), file=sys.stderr)
    res = {"retrieval": retrieval(a.k)}
    if a.answers:
        res["answers"] = answers(a.base)
    dest = ROOT / "results/local/eval_rag.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
