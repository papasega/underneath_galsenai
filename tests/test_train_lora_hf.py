"""CPU checks for the portable LoRA path; no model downloads."""

import importlib.util
import json

import click
import pytest
from click.testing import CliRunner

from scripts import train_lora_hf


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}!" for m in messages)
        return list(text + ("<assistant>" if add_generation_prompt else ""))


def write_split(path, messages):
    path.write_text(json.dumps({"messages": messages}) + "\n")


MESSAGES = [
    {"role": "system", "content": "Be helpful"},
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"},
]


def test_only_assistant_tokens_contribute_to_loss(tmp_path):
    path = tmp_path / "train.jsonl"
    write_split(path, MESSAGES + [{"role": "user", "content": "Next"}, {"role": "assistant", "content": "Again"}])
    rows = train_lora_hf.encode_split(path, Tokenizer(), 200)
    assert len(rows) == 2
    for row, answer in zip(rows, ("Answer!", "Again!"), strict=True):
        assert "".join(t for t in row["labels"] if t != -100) == answer
        assert row["labels"][:10] == [-100] * 10
        assert len(row["labels"]) == len(row["input_ids"]) == len(row["attention_mask"])


def test_rejects_truncation_and_malformed_or_empty_data(tmp_path):
    path = tmp_path / "train.jsonl"
    write_split(path, MESSAGES)
    with pytest.raises(click.ClickException, match="no silent truncation"):
        train_lora_hf.encode_split(path, Tokenizer(), 2)
    for content in ('{"messages": []}', '{"messages": [{"role": "assistant", "content": "A"}]}', "", "bad"):
        path.write_text(content)
        with pytest.raises(click.ClickException):
            train_lora_hf.encode_split(path, Tokenizer(), 200)


def test_preflight_prints_notice_and_never_overwrites(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("untouched")
    result = CliRunner().invoke(train_lora_hf.main, ["--output", str(output)])
    assert result.exit_code != 0
    assert "example data only" in result.output
    assert "GalsenAI community" in result.output
    assert "Output already exists" in result.output
    assert sentinel.read_text() == "untouched"
    result = CliRunner().invoke(
        train_lora_hf.main, ["--output", str(tmp_path / "new"), "--data", str(tmp_path / "missing")]
    )
    assert result.exit_code != 0
    assert "Request dataset access first" in result.output
    assert not (tmp_path / "new").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("peft") is None or importlib.util.find_spec("accelerate") is None,
    reason="Install .[hf-train] for the real CPU training smoke check",
)
def test_real_lora_step_on_tiny_local_cpu_model(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    base = tmp_path / "base"
    backend = FastTokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' [EOS] ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    tokenizer.save_pretrained(base)
    network = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=3,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    network.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train", "valid"):
        write_split(data / f"{name}.jsonl", MESSAGES)
    output = tmp_path / "adapter"
    result = CliRunner().invoke(
        train_lora_hf.main,
        [
            "--model",
            str(base),
            "--data",
            str(data),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--max-length",
            "64",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "checkpoint-1" / "trainer_state.json").is_file()
    state = json.loads((output / "checkpoint-1" / "trainer_state.json").read_text())
    assert any("eval_loss" in entry for entry in state["log_history"])
    provenance = json.loads((output / "training_provenance.json").read_text())
    assert provenance["device"] == "cpu"
    assert provenance["train_targets"] == 1
    assert set(provenance["inputs"]) == {"train.jsonl", "valid.jsonl"}
