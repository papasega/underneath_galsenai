"""Train the minbpe tokenizer on the anonymized chat plus the SFT train split, then add the chat specials."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from galsenai.chat import SPECIAL_TOKENS  # noqa: E402
from galsenai.train import chat_messages  # noqa: E402
from minbpe import RegexTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "config_nanochat.yml")
cfg = yaml.safe_load(ap.parse_args().config.read_text())["tokenizer"]

text = "\n".join(chat_messages(ROOT / cfg["corpus"]))
for line in (ROOT / cfg["sft_train"]).read_text(encoding="utf-8").splitlines():
    text += "\n" + "\n".join(m["content"] for m in json.loads(line)["messages"])
tok = RegexTokenizer()
tok.train(text, cfg["vocab_size"], verbose=False)
tok.register_special_tokens({t: cfg["vocab_size"] + i for i, t in enumerate(SPECIAL_TOKENS)})
out = ROOT / cfg["out_prefix"]
out.parent.mkdir(parents=True, exist_ok=True)
tok.save(str(out))
print(f"{len(text)} chars -> {out}.model, vocab {cfg['vocab_size'] + len(SPECIAL_TOKENS)}")
