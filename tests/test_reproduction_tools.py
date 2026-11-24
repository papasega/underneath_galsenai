"""Exercise portable data preparation and private archival without model downloads."""

import hashlib
import json
import tarfile

from click.testing import CliRunner

from galsenai.model_sources import pinned_revision
from scripts import archive_reproduction, mlx_entry, prepare_sft_mlx


def test_converter_accepts_objects_and_legacy_messages_without_overwrite(tmp_path):
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    source = tmp_path / "examples.json"
    source.write_text(json.dumps([{"id": "example", "messages": messages}, messages]))
    output = tmp_path / "snapshot"
    args = ["--source", str(source), "--output", str(output), "--valid-size", "1"]
    result = CliRunner().invoke(prepare_sft_mlx.main, args, color=True)
    assert result.exit_code == 0, result.output
    assert "\x1b[32m" in result.output
    for split in ("train", "valid"):
        assert json.loads((output / f"{split}.jsonl").read_text()) == {"messages": messages}
    assert CliRunner().invoke(prepare_sft_mlx.main, args).exit_code == 1
    source.write_text(json.dumps([{"messages": {"messages": []}}, {"messages": messages}]))
    result = CliRunner().invoke(prepare_sft_mlx.main, [*args[:3], str(tmp_path / "bad"), "--valid-size", "1"])
    assert result.exit_code == 1
    assert not (tmp_path / "bad").exists()


def test_pins_are_immutable_and_local_models_stay_local(tmp_path):
    import pytest

    assert len(pinned_revision("google/gemma-4-E2B-it")) == 40
    assert pinned_revision(str(tmp_path)) is None
    with pytest.raises(ValueError, match="immutable"):
        pinned_revision("unknown/model", "main")


def test_mlx_entry_forwards_local_model_and_flags(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(mlx_entry.os, "execv", lambda executable, args: seen.extend(args))
    result = CliRunner().invoke(mlx_entry.main, ["server", "--model", str(tmp_path), "--port", "8001"])
    assert result.exit_code == 0, result.output
    assert seen[1:3] == ["-m", "mlx_lm.server"]
    assert seen[-2:] == ["--model", str(tmp_path)]
    assert "8001" in seen


def test_private_archive_checksums_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_reproduction, "ROOT", tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source/train.jsonl").write_text("private examples\n")
    output = tmp_path / "backups/run1"
    args = ["--include", "source", "--output", str(output)]
    result = CliRunner().invoke(archive_reproduction.main, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["archive_sha256"] == hashlib.sha256((output / "inputs.tar.gz").read_bytes()).hexdigest()
    with tarfile.open(output / "inputs.tar.gz") as archive:
        for name, digest in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    assert CliRunner().invoke(archive_reproduction.main, args).exit_code == 1
    (tmp_path / "source/weights.safetensors").write_text("excluded")
    result = CliRunner().invoke(archive_reproduction.main, [*args[:-1], str(tmp_path / "other")])
    assert result.exit_code == 0, result.output
    filtered = json.loads((tmp_path / "other/manifest.json").read_text())
    assert "source/weights.safetensors" not in filtered["files_sha256"]


def test_mlx_entry_resolves_pin_and_writes_training_provenance(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    source = tmp_path / "data"
    source.mkdir()
    for split in ("train", "valid"):
        (source / f"{split}.jsonl").write_text("{}\n")
    output = tmp_path / "adapter"
    output.mkdir()
    seen = {}

    def download(model, **kwargs):
        seen.update(model=model, **kwargs)
        return str(tmp_path / "model")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(DEFAULT_ALLOW_PATTERNS=["*.json"]))
    monkeypatch.setattr(mlx_entry.subprocess, "run", lambda args, **kwargs: seen.update(args=args))
    monkeypatch.setattr(mlx_entry.importlib.metadata, "version", lambda package: "test-version")
    result = CliRunner().invoke(
        mlx_entry.main,
        [
            "lora",
            "--model",
            "google/gemma-4-E2B-it",
            "--revision",
            "a" * 40,
            "--data",
            str(source),
            "--adapter-path",
            str(output),
            "--iters",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["revision"] == "a" * 40
    assert "--revision" not in seen["args"]
    assert seen["args"][-2:] == ["--model", str(tmp_path / "model")]
    provenance = json.loads((output / "training_provenance.json").read_text())
    assert provenance["revision"] == "a" * 40
    assert provenance["inputs"]["train.jsonl"] == hashlib.sha256(b"{}\n").hexdigest()
