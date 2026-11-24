"""Check dataset isolation and wrapper scheduling without launching MLX training."""

import os
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_v2_configuration():
    cfg = yaml.safe_load((ROOT / "config/config_gemma.yml").read_text())
    assert cfg["data"] == "data/training_data/pilot_v4_assistant"
    assert "pilot_v4_assistant" in cfg["adapter_path"]
    assert cfg["mask_prompt"] is True
    assert cfg["epochs"] == 1
    assert cfg["learning_rate"] == 5e-5


def test_wrapper_respects_config_cadence_and_derives_iterations(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text("{}\n" * 7)
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "dataset_notice.txt").write_text((ROOT / "config/dataset_notice.txt").read_text())
    cfg = config / "config_gemma.yml"
    text = f"epochs: 1\nbatch_size: 2\ndata: {dataset}\n"
    cfg.write_text(text + "steps_per_eval: 2\nsave_every: 2\n")
    wrapper = scripts / "train_lora.sh"
    wrapper.write_text(
        (ROOT / "scripts/train_lora.sh")
        .read_text()
        .replace(
            '"$MLXPY" scripts/mlx_entry.py lora --config "$CONFIG" "$@"', 'printf "ARG:%s\\n" --config "$CONFIG" "$@"'
        )
    )
    # Reproduce the runner without ripgrep: the wrapper only needs standard tools.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("bash", "dirname", "cat", "grep", "python3"):
        (bin_dir / command).symlink_to(shutil.which(command))
    env = {**os.environ, "PATH": str(bin_dir)}
    result = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True, check=True)
    assert "iters=4 (4 per epoch)" in result.stdout
    assert "example data only" in result.stderr
    assert "\x1b[32m" in result.stderr
    assert "ARG:--steps-per-eval" not in result.stdout
    assert "ARG:--save-every" not in result.stdout
    cfg.write_text(text)
    result = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True, check=True)
    assert "ARG:--steps-per-eval\nARG:4" in result.stdout
    assert "ARG:--save-every\nARG:4" in result.stdout
    result = subprocess.run(["bash", str(wrapper), "--batch-size", "1"], capture_output=True, text=True, check=True)
    assert "iters=7 (7 per epoch)" in result.stdout
    result = subprocess.run(["bash", str(wrapper), "--iters=3"], capture_output=True, text=True, check=True)
    assert "epochs from config" not in result.stdout


def test_v2_full_and_nanochat_isolate_data_and_outputs():
    full = yaml.safe_load((ROOT / "config/config_gemma_full.yml").read_text())
    nano = yaml.safe_load((ROOT / "config/config_nanochat.yml").read_text())
    assert full["data"] == "data/training_data/pilot_v2"
    assert full["fine_tune_type"] == "full"
    assert "pilot_v2" in full["adapter_path"]
    assert nano["tokenizer"]["corpus"] is None
    assert nano["tokenizer"]["sft_train"] == nano["sft"]["train"]
    assert nano["sft"]["train"] == "data/training_data/pilot_v2/train.jsonl"
    assert nano["sft"]["valid"] == "data/training_data/pilot_v2/valid.jsonl"
    assert nano["sft"]["init_from"] is None
    assert "pilot_v2" in nano["sft"]["out"]
    assert "pilot_v2" in nano["tokenizer"]["out_prefix"]


def test_tokenizer_can_use_only_training_messages(tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    from scripts import train_tokenizer

    seen = {}

    class Tokenizer:
        def train(self, text, vocab_size, verbose):
            seen["text"] = text

        def register_special_tokens(self, tokens):
            pass

        def save(self, path):
            pass

    def reject_raw_corpus(_):
        raise AssertionError("Raw corpus read")

    monkeypatch.setattr(train_tokenizer, "ROOT", tmp_path)
    monkeypatch.setattr(train_tokenizer, "RegexTokenizer", Tokenizer)
    monkeypatch.setattr(train_tokenizer, "chat_messages", reject_raw_corpus)
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]})
    )
    config = tmp_path / "tokenizer.yml"
    config.write_text(
        yaml.safe_dump(
            {"tokenizer": {"corpus": None, "sft_train": "train.jsonl", "vocab_size": 20, "out_prefix": "tok/model"}}
        )
    )
    result = CliRunner().invoke(train_tokenizer.main, ["--config", str(config)])
    assert result.exit_code == 0, result.output
    assert seen["text"].strip() == "Question\nAnswer"
