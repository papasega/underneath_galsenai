import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import rag  # noqa: E402

CHAT = """2026-04 | U01: Quel GPU pour entraîner un modèle ?
2026-04 | U02: Une H100 si tu as le budget
mais une RTX suffit pour du LoRA
[lien]
2026-05 | U03: Qui a vu le match hier ?
2026-05 | U04: Haskell est élégant mais Python domine en IA
2026-06 | U05: Le wolof manque de données pour l'ASR
"""


def corpus(tmp_path):
    f = tmp_path / "chat.txt"
    f.write_text(CHAT, encoding="utf-8")
    return rag.load_messages(f)


def test_load_messages_joins_continuation_lines_and_drops_labels(tmp_path):
    msgs = corpus(tmp_path)
    assert len(msgs) == 5
    assert msgs[1] == ("2026-04", "Une H100 si tu as le budget\nmais une RTX suffit pour du LoRA\n[lien]")
    assert not any("U0" in t for _, t in msgs)


def test_make_passages_drops_noise_lines_and_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "WINDOW", 2)
    monkeypatch.setattr(rag, "STRIDE", 1)
    passages = rag.make_passages(corpus(tmp_path))
    assert len(passages) == 4
    assert all("[lien]" not in p["text"] for p in passages)
    assert [p["id"] for p in passages] == [0, 1, 2, 3]


def test_tokens_strip_accents_stopwords_and_plurals():
    assert rag.tokens("Les modèles ÉLÉGANTS, quel GPUs ?") == ["modele", "elegant", "gpu"]


def test_bm25_ranks_the_matching_passage_first_and_ignores_unknown_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "WINDOW", 1)
    monkeypatch.setattr(rag, "STRIDE", 1)
    docs = [rag.tokens(p["text"]) for p in rag.make_passages(corpus(tmp_path))]
    bm = rag.BM25(docs)
    s = bm.scores(rag.tokens("quel GPU H100 ?"))
    assert s.index(max(s)) == 0 or s.index(max(s)) == 1
    assert max(bm.scores(rag.tokens("zzzxxx"))) == 0


def test_retrieve_bm25_returns_nothing_without_a_shared_term(tmp_path, monkeypatch):
    msgs = corpus(tmp_path)
    monkeypatch.setattr(rag, "load_messages", lambda: msgs)
    monkeypatch.setattr(rag, "WINDOW", 1)
    monkeypatch.setattr(rag, "STRIDE", 1)
    rag._state.clear()
    assert rag.retrieve("quelle recette de thiéboudienne", 3, "bm25") == []
    assert "H100" in rag.retrieve("budget H100", 1, "bm25")[0]["text"]
    rag._state.clear()


def test_query_from_adds_previous_turn_for_one_word_questions():
    h = [
        {"role": "user", "content": "parlons de Haskell"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "hello"},
    ]
    assert rag.query_from(h) == "parlons de Haskell hello"
    assert rag.query_from(h[:1]) == "parlons de Haskell"


def test_dense_cache_distinguishes_same_size_corpora(tmp_path, monkeypatch):
    import numpy as np

    class Embedder:
        calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.array([[self.calls]], dtype=float)

    embedder = Embedder()
    monkeypatch.setattr(rag, "CACHE", tmp_path)
    monkeypatch.setattr(rag, "_embedder", lambda: embedder)
    monkeypatch.setattr(rag, "_state", {"passages": [{"text": "old corpus"}]})
    assert rag._doc_vectors()[0, 0] == 1
    monkeypatch.setattr(rag, "_state", {"passages": [{"text": "new corpus"}]})
    assert rag._doc_vectors()[0, 0] == 2
    monkeypatch.setattr(rag, "_state", {"passages": [{"text": "old corpus"}]})
    assert rag._doc_vectors()[0, 0] == 1
    assert embedder.calls == 2


def test_build_prompt_numbers_passages_and_keeps_the_question():
    p = rag.build_prompt("Quoi ?", [{"month": "2026-04", "text": "a"}, {"month": "2026-05", "text": "b"}])
    assert "[1] (2026-04)\na" in p and "[2] (2026-05)\nb" in p and "Question : Quoi ?" in p
    assert "ces extraits ne suffisent pas" in p
    assert "dis que le groupe n'en a pas parlé" not in p
