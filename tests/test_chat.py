"""The chat CLI imports the vendored tokenizer, so this also guards the install dependencies."""

from galsenai.chat import format_prompt
from minbpe import RegexTokenizer


def test_prompt_ends_with_an_open_assistant_turn() -> None:
    prompt = format_prompt([{"role": "system", "content": "be kind"}, {"role": "user", "content": "salam"}])
    assert prompt == (
        "<|start_turn|>system<|separator|>be kind<|end_turn|>"
        "<|start_turn|>user<|separator|>salam<|end_turn|>"
        "<|start_turn|>assistant<|separator|>"
    )


def test_tokenizer_roundtrips_text() -> None:
    tokenizer = RegexTokenizer()
    tokenizer.train("salam aleykum, nanga def? maa ngi fi rekk, jerejef" * 20, 262, verbose=False)
    text = "salam aleykum"
    assert tokenizer.decode(tokenizer.encode(text)) == text
