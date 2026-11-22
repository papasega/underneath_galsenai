"""Interactive chat with a trained checkpoint: ``python -m galsenai.chat --help``."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from galsenai.checkpoint import load_checkpoint
from galsenai.generate import generate
from minbpe import RegexTokenizer

TURN_START, TURN_END, SEPARATOR, EOS = "<|start_turn|>", "<|end_turn|>", "<|separator|>", "<|endoftext|>"


def format_prompt(turns: list[dict[str, str]]) -> str:
    """Render chat turns, ending with an open assistant turn for the model to complete."""
    body = "".join(f"{TURN_START}{t['role']}{SEPARATOR}{t['content']}{TURN_END}" for t in turns)
    return f"{body}{TURN_START}assistant{SEPARATOR}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with a galsenai checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="minbpe .model file")
    parser.add_argument("--system", default="", help="Optional system message.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    tokenizer = RegexTokenizer()
    tokenizer.load(model_file=str(args.tokenizer))
    stop = {tokenizer.special_tokens[EOS], tokenizer.special_tokens[TURN_END]}

    turns: list[dict[str, str]] = [{"role": "system", "content": args.system}] if args.system else []
    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            return 0
        turns.append({"role": "user", "content": user})
        prompt = torch.tensor([tokenizer.encode(format_prompt(turns), allowed_special="all")], device=device)
        out = generate(model, prompt, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, eos_ids=stop)
        new_tokens = [t for t in out[0, prompt.size(1) :].tolist() if t not in stop]
        answer = tokenizer.decode(new_tokens).strip()
        turns.append({"role": "assistant", "content": answer})
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())
