"""Retrieval over the anonymized chat: BM25 (no dependency), embeddinggemma-2 (text only), or both (RRF).

Reads only data/private/chat_anonymized.txt (format `YYYY-MM | Uxx: text`, following lines continue the message).
Dense vectors are cached in data/private/rag/ (git-ignored until the leak check is done).
Env: RAG_CHAT, RAG_DIR, EMBED_MODEL (default google/embeddinggemma-2).
"""

import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT = Path(os.environ.get("RAG_CHAT", ROOT / "data/private/chat_anonymized.txt"))
CACHE = Path(os.environ.get("RAG_DIR", ROOT / "data/private/rag"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/embeddinggemma-2")
WINDOW, STRIDE, MAX_CHARS = 4, 2, 1200
_MSG = re.compile(r"^(\d{4}-\d{2}) \| (U\d+): ")
_state: dict = {}


def load_messages(path: Path = CHAT) -> list[tuple[str, str]]:
    """(month, text) per message; the voice label is dropped from the text indexed and shown."""
    msgs: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := _MSG.match(line):
            msgs.append([m[1], line[m.end():]])
        elif msgs and line.strip():
            msgs[-1][1] += "\n" + line
    return [(month, text.strip()) for month, text in msgs]


_NOISE = re.compile(r"^(\[lien\]|\[mention\]|\S+ \(fichier joint\)|[\W_]*)$")


def make_passages(msgs: list[tuple[str, str]]) -> list[dict]:
    msgs = [(m, "\n".join(ln for ln in t.splitlines() if not _NOISE.match(ln.strip()))) for m, t in msgs]
    msgs = [(m, t) for m, t in msgs if t.strip()]  # attachments, bare links and emoji-only lines carry no content
    out = []
    for i in range(0, max(len(msgs) - WINDOW, 0) + 1, STRIDE):
        w = msgs[i:i + WINDOW]
        out.append({"id": len(out), "month": w[0][0], "text": "\n".join(t for _, t in w)[:MAX_CHARS]})
    return out


STOP = set(
    "le la les un une des du de et en au aux ce ces cet cette qui que quoi quel quelle quels quelles est sont etre a ont "
    "pour par sur dans avec sans ne pas plus ou je tu il elle on nous vous ils elles me te se mon ma mes ton ta tes son "
    "sa ses y the a an of to in is are and or for on it that this".split()
)


def tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    toks = re.findall(r"\w+", "".join(c for c in s if not unicodedata.combining(c)))
    # shortcut: crude plural stripping instead of a stemmer; upgrade if recall on French inflections is poor
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if len(t) > 1 and t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in set(q):
                if f := tf.get(t):
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def _index() -> dict:
    if "passages" not in _state:
        passages = make_passages(load_messages())
        _state.update(passages=passages, bm25=BM25([tokens(p["text"]) for p in passages]))
    return _state


def _embedder():
    if "model" not in _state:
        from sentence_transformers import SentenceTransformer

        # float32: embeddinggemma-2 returns NaN in float16; text-only config skips the 470M vision/audio encoders
        _state["model"] = SentenceTransformer(
            EMBED_MODEL, config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"torch_dtype": "float32"}
        )
    return _state["model"]


def _doc_vectors():
    import numpy as np

    st = _index()
    if "vecs" not in st:
        f = CACHE / f"emb_{EMBED_MODEL.replace('/', '_')}_w{WINDOW}s{STRIDE}_{len(st['passages'])}.npy"
        if f.exists():
            st["vecs"] = np.load(f)
        else:
            st["vecs"] = _embedder().encode(
                [p["text"] for p in st["passages"]], prompt_name="Document", batch_size=16, normalize_embeddings=True
            )
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(f, st["vecs"])
    return st["vecs"]


def reset() -> None:
    """Drop the index (keeps the loaded embedder); call after changing WINDOW / STRIDE."""
    for key in ("passages", "bm25", "vecs"):
        _state.pop(key, None)


def query_from(messages: list[dict]) -> str:
    """Last user turn; a very short one ("hello", "claude") gets the previous user turn as extra context."""
    users = [m["content"] for m in messages if m.get("role") == "user"]
    return users[-1] if len(tokens(users[-1])) > 2 or len(users) < 2 else users[-2] + " " + users[-1]


def _rank(scores) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def retrieve(query: str, k: int = 4, method: str = "bm25") -> list[dict]:
    """Top-k passages with their score; method: bm25 | dense | hybrid (reciprocal rank fusion, k=60)."""
    st = _index()
    rankings: list[list[int]] = []
    if method in ("bm25", "hybrid"):
        rankings.append(_rank(st["bm25"].scores(tokens(query))))
    if method in ("dense", "hybrid"):
        qv = _embedder().encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
        rankings.append(_rank(_doc_vectors() @ qv))
    if not rankings:
        raise ValueError(f"unknown method: {method}")
    fused: Counter[int] = Counter()
    for r in rankings:
        for pos, i in enumerate(r[:100]):
            fused[i] += 1 / (60 + pos)
    best = [i for i, _ in fused.most_common(k)]
    if method == "bm25":  # no term in common with the query: return nothing rather than noise
        sc = st["bm25"].scores(tokens(query))
        best = [i for i in best if sc[i] > 0]
    return [st["passages"][i] for i in best]


def build_prompt(question: str, passages: list[dict]) -> str:
    body = "\n\n".join(f"[{n}] ({p['month']})\n{p['text']}" for n, p in enumerate(passages, 1))
    return (
        f"Extraits du groupe de discussion (anonymisés) :\n\n{body}\n\nQuestion : {question}\n\n"
        "Consigne : réponds à la question avec tes propres mots, en 3 à 6 phrases complètes, en t'appuyant uniquement "
        "sur les extraits ci-dessus. Synthétise les points de vue, ne recopie pas les extraits. Termine chaque idée par "
        "les numéros des extraits utilisés, par exemple [1][3]. Si les extraits ne répondent pas à la question, "
        "dis que le groupe n'en a pas parlé."
    )
