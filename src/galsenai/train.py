"""Base pre-training and SFT in the nanochat recipe.

    python -m galsenai.train --config config/pilotv1_config/config_nanochat.yml --stage sft

``base`` learns next-token prediction on chat text; ``sft`` trains on conversations with the loss on assistant
tokens only. AdamW with warmup + cosine decay, grad clipping, float32 (MPS/CUDA/CPU).
nanochat's Muon optimizer is not ported.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import torch
import yaml

from galsenai import GPT, ModelConfig
from galsenai.chat import BOS, load_tokenizer, render_conversation
from galsenai.checkpoint import load_checkpoint, save_checkpoint
from minbpe import RegexTokenizer

Cfg = dict[str, Any]
Batch = tuple[torch.Tensor, torch.Tensor]

ROOT = Path.cwd()
PREFIX = re.compile(r"^\d{4}-\d\d \| U\d+: ")
SKIP = re.compile(r"^(<Médias omis>|\[lien\]|\[mention\]|\[numero\])?\s*$")


def chat_messages(path: Path) -> list[str]:
    """Message texts of an anonymized chat export: prefix stripped, empty/media-only lines dropped."""
    out = (PREFIX.sub("", line).strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [m for m in out if not SKIP.match(m)]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # decays to 10% of peak


def fit(
    model: GPT,
    batch_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    steps_per_epoch: int,
    hp: Cfg,
    eval_fn: Callable[[], float],
    out: Path,
) -> None:
    """Train for `steps`, evaluating each epoch; the best-validation checkpoint is written to `out`."""
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
    best, t0 = float("inf"), time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, hp["learning_rate"], hp["warmup_steps"])
        model.train()
        x, y = batch_fn(step)
        loss = model(x, y).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % steps_per_epoch == 0 or step + 1 == steps:
            val = eval_fn()
            epoch = (step + 1) / steps_per_epoch
            mark = ""
            if val < best:
                best, mark = val, " *"
                save_checkpoint(model, out)
            print(
                f"epoch {epoch:5.1f} step {step + 1:5d} train {loss.item():.3f} "
                f"val {val:.3f}{mark} ({time.time() - t0:.0f}s)"
            )
    print(f"best val {best:.3f} -> {out}")


def pad_batch(rows: list[tuple[list[int], list[int]]], block: int, ignore: int, device: str) -> Batch:
    """Right-pad (ids, mask) rows into inputs and targets; masked and padded targets become ignore_index."""
    n = min(max(len(i) for i, _ in rows), block + 1)
    x = torch.zeros(len(rows), n - 1, dtype=torch.long)
    y = torch.full((len(rows), n - 1), ignore, dtype=torch.long)
    for r, (ids, mask) in enumerate(rows):
        ids, mask = ids[:n], mask[:n]
        x[r, : len(ids) - 1] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        y[r, : len(ids) - 1] = torch.where(torch.tensor(mask[1:]) == 1, tgt, torch.full_like(tgt, ignore))
    return x.to(device), y.to(device)


def build_model(cfg: Cfg, vocab_size: int) -> GPT:
    return GPT(ModelConfig.nanochat(vocab_size, **cfg["model"]))


def train_base(cfg: Cfg, tok: RegexTokenizer, device: str) -> None:
    hp, rng = cfg["base"], random.Random(cfg["seed"])
    bos = tok.special_tokens[BOS]
    ids = [t for m in chat_messages(ROOT / hp["corpus"]) for t in (bos, *tok.encode(m))]
    cut = int(len(ids) * (1 - hp["val_fraction"]))
    train_ids, val_ids = torch.tensor(ids[:cut]), torch.tensor(ids[cut:])
    T, B = hp["seq_len"], hp["batch_size"]
    steps_per_epoch = max(len(train_ids) // (T * B), 1)
    steps = hp["epochs"] * steps_per_epoch
    print(f"base: {len(ids)} tokens, {steps_per_epoch} steps/epoch x {hp['epochs']} epochs = {steps} steps")
    model = build_model(cfg, tok_vocab(tok)).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M params on {device}")

    def window(src: torch.Tensor, starts: list[int]) -> Batch:
        x = torch.stack([src[s : s + T] for s in starts]).to(device)
        y = torch.stack([src[s + 1 : s + T + 1] for s in starts]).to(device)
        return x, y

    def batch(_: int) -> Batch:
        return window(train_ids, [rng.randrange(len(train_ids) - T - 1) for _ in range(B)])

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        starts = list(range(0, len(val_ids) - T - 1, T))[:64] or [0]
        x, y = window(val_ids, starts)
        return model(x, y).loss.item()

    fit(model, batch, steps, steps_per_epoch, hp, evaluate, ROOT / hp["out"])


def read_jsonl(path: Path) -> list[list[dict[str, str]]]:
    return [json.loads(line)["messages"] for line in path.read_text(encoding="utf-8").splitlines()]


def tok_vocab(tok: RegexTokenizer) -> int:
    return max(max(tok.vocab), *tok.special_tokens.values()) + 1


def train_sft(cfg: Cfg, tok: RegexTokenizer, device: str) -> None:
    hp, rng = cfg["sft"], random.Random(cfg["seed"])
    train = [render_conversation(tok, c) for c in read_jsonl(ROOT / hp["train"])]
    valid = [render_conversation(tok, c) for c in read_jsonl(ROOT / hp["valid"])]
    B = hp["batch_size"]
    steps_per_epoch = math.ceil(len(train) / B)
    steps = hp["epochs"] * steps_per_epoch
    print(
        f"sft: {len(train)} train / {len(valid)} valid conversations, "
        f"{steps_per_epoch} steps/epoch x {hp['epochs']} epochs = {steps} steps"
    )
    init = hp.get("init_from")
    model = load_checkpoint(ROOT / init, device) if init else build_model(cfg, tok_vocab(tok)).to(device)
    block, ign = model.cfg.block_size, model.cfg.ignore_index
    order: list[int] = []

    def batch(_: int) -> Batch:
        nonlocal order
        if len(order) < B:
            order = rng.sample(range(len(train)), len(train))  # a fresh shuffle each epoch
        take, order = order[:B], order[B:]
        return pad_batch([train[i] for i in take], block, ign, device)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        x, y = pad_batch(valid, block, ign, device)
        return model(x, y).loss.item()

    fit(model, batch, steps, steps_per_epoch, hp, evaluate, ROOT / hp["out"])


@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=ROOT / "config/pilotv1_config/config_nanochat.yml")
@click.option("--stage", type=click.Choice(["base", "sft"]), required=True)
@click.option("--set", "overrides", multiple=True, help="Repeat for each override, e.g. --set sft.epochs=3")
def main(config: Path, stage: str, overrides: tuple[str, ...]) -> int:
    cfg = yaml.safe_load(config.read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(value)
    torch.manual_seed(cfg["seed"])
    tok = load_tokenizer(ROOT / (cfg["tokenizer"]["out_prefix"] + ".model"))
    (train_base if stage == "base" else train_sft)(cfg, tok, pick_device())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
