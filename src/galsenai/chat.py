"""Chat format (nanochat style) and interactive chat: ``python -m galsenai.chat --help``.

A conversation renders as ``<|bos|>`` then, per turn, ``<|user_start|>..<|user_end|>`` or
``<|assistant_start|>..<|assistant_end|>``. A system message is merged into the first user turn, as in
nanochat. During SFT only assistant content and ``<|assistant_end|>`` are trained on (mask = 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from galsenai.checkpoint import load_checkpoint
from galsenai.generate import generate
from minbpe import RegexTokenizer

BOS, USER_START, USER_END, ASSISTANT_START, ASSISTANT_END = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
)
SPECIAL_TOKENS = [BOS, USER_START, USER_END, ASSISTANT_START, ASSISTANT_END]


def load_tokenizer(model_file: Path) -> RegexTokenizer:
    tokenizer = RegexTokenizer()
    tokenizer.load(model_file=str(model_file))
    return tokenizer


def merge_system(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fold a leading system message into the first user message."""
    if turns and turns[0]["role"] == "system":
        if len(turns) < 2 or turns[1]["role"] != "user":
            raise ValueError("a system message must be followed by a user message")
        return [{"role": "user", "content": f"{turns[0]['content']}\n\n{turns[1]['content']}"}, *turns[2:]]
    return turns


def render_conversation(tokenizer: RegexTokenizer, turns: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """Token ids and a 0/1 training mask (1 = assistant content and its end token)."""
    sp = tokenizer.special_tokens
    ids, mask = [sp[BOS]], [0]
    for i, turn in enumerate(merge_system(turns)):
        if turn["role"] != ("user", "assistant")[i % 2]:
            raise ValueError("turns must alternate user, assistant, starting with user")
        body = tokenizer.encode(turn["content"])
        if turn["role"] == "user":
            ids += [sp[USER_START], *body, sp[USER_END]]
            mask += [0] * (len(body) + 2)
        else:
            ids += [sp[ASSISTANT_START], *body, sp[ASSISTANT_END]]
            mask += [0] + [1] * (len(body) + 1)
    return ids, mask


def encode_prompt(tokenizer: RegexTokenizer, turns: list[dict[str, str]]) -> list[int]:
    """Ids of turns ending with an open assistant turn for the model to complete."""
    ids, _ = render_conversation(tokenizer, turns)
    return [*ids, tokenizer.special_tokens[ASSISTANT_START]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with a galsenai checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="minbpe .model file")
    parser.add_argument("--system", default="", help="Optional system message.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)
    stop = {tokenizer.special_tokens[ASSISTANT_END]}

    turns: list[dict[str, str]] = [{"role": "system", "content": args.system}] if args.system else []
    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            return 0
        turns.append({"role": "user", "content": user})
        prompt = torch.tensor([encode_prompt(tokenizer, turns)], device=device)
        out = generate(model, prompt, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, eos_ids=stop)
        new_tokens = [t for t in out[0, prompt.size(1) :].tolist() if t not in stop]
        answer = tokenizer.decode(new_tokens).strip()
        turns.append({"role": "assistant", "content": answer})
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())
