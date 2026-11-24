import io
import json
import subprocess
from pathlib import Path

import pytest

from data.prepare_assistant_data import build
from data.prepare_training_data import digest, encode


def test_assistant_snapshot_filters_whole_rows_and_preserves_splits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "v2"
    source.mkdir()
    hashes = {}
    for split, questions in (("train", ["Comment tester ?", "Que dit un membre ?"]), ("valid", ["Comment valider ?"])):
        rows = [
            {
                "uid": f"{split}:{i}",
                "component_id": split,
                "theme": "testing",
                "type": "complementary",
                "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "Écris un test ciblé."}],
            }
            for i, q in enumerate(questions)
        ]
        (source / f"{split}.metadata.json").write_bytes(encode(rows))
        (source / f"{split}.jsonl").write_text("".join(json.dumps({"messages": r["messages"]}) + "\n" for r in rows))
        for suffix in ("jsonl", "metadata.json"):
            name = f"{split}.{suffix}"
            hashes[name] = digest((source / name).read_bytes())
    (source / "manifest.json").write_bytes(encode({"files_sha256": hashes}))
    output = tmp_path / "assistant"
    counts = build(source, output, root)
    seeds = json.loads((root / "data/training_data/assistant_seed_examples.json").read_text())
    assert counts == {"train": 1 + len(seeds), "valid": 1}
    train = [json.loads(s) for s in (output / "train.jsonl").read_text().splitlines()]
    assert train[0]["messages"][1]["content"] == "Comment tester ?"
    assert all(r["messages"][0]["role"] == "system" for r in train)
    before = (output / "train.jsonl").read_bytes()
    assert build(source, output, root) == counts
    (output / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="frozen"):
        build(source, output, root)
    assert (output / "train.jsonl").read_text() == "changed"
    assert before


def test_webchat_uses_training_system_and_disables_thinking(monkeypatch):
    from webapp import server

    captured = {}

    def upstream(path, body):
        captured.update(body)
        return io.BytesIO(b"")

    monkeypatch.setattr(server, "upstream", upstream)
    monkeypatch.setattr(server, "upstream_model", lambda: "default_model")
    monkeypatch.setattr(server, "SYSTEM", server.DEFAULT_SYSTEM)
    handler = server.Handler.__new__(server.Handler)
    handler.wfile = io.BytesIO()
    monkeypatch.setattr(handler, "_sse_start", lambda: None)
    handler._gemma([{"role": "user", "content": "Bonjour"}], 512, 0)
    assert captured["messages"][0] == {"role": "system", "content": server.DEFAULT_SYSTEM}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["model"] == "default_model"


def test_serve_rejects_missing_selected_adapter_before_starting_server(tmp_path):
    import os

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts/serve.sh")],
        env={**os.environ, "ADAPTER": str(tmp_path / "missing")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Missing adapter file" in result.stderr
    script = (root / "scripts/serve.sh").read_text()
    assert "MODEL=${MODEL:-google/gemma-4-E2B-it}" in script
    assert "pilot_v4_assistant/google_gemma4_rank8_run1" in script
