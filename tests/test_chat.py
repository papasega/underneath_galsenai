"""The chat CLI imports the vendored tokenizer, so this also guards the install dependencies."""

import pytest

from galsenai.chat import SPECIAL_TOKENS, encode_prompt, render_conversation
from minbpe import RegexTokenizer


def make_tokenizer() -> RegexTokenizer:
    tokenizer = RegexTokenizer()
    tokenizer.train("salam aleykum, nanga def? maa ngi fi rekk, jerejef" * 20, 262, verbose=False)
    tokenizer.register_special_tokens({t: 262 + i for i, t in enumerate(SPECIAL_TOKENS)})
    return tokenizer


def test_tokenizer_roundtrips_text() -> None:
    tokenizer = make_tokenizer()
    assert tokenizer.decode(tokenizer.encode("salam aleykum")) == "salam aleykum"


def test_mask_trains_only_on_assistant_content_and_end_token() -> None:
    tok = make_tokenizer()
    sp = tok.special_tokens
    turns = [{"role": "user", "content": "salam"}, {"role": "assistant", "content": "maa ngi fi"}]
    ids, mask = render_conversation(tok, turns)
    n_user, n_asst = len(tok.encode("salam")), len(tok.encode("maa ngi fi"))
    assert ids[0] == sp["<|bos|>"] and ids[-1] == sp["<|assistant_end|>"]
    assert mask == [0] + [0] * (n_user + 2) + [0] + [1] * (n_asst + 1)
    assert len(ids) == len(mask)


def test_prompt_ends_with_an_open_assistant_turn_and_system_is_merged() -> None:
    tok = make_tokenizer()
    ids = encode_prompt(tok, [{"role": "system", "content": "be kind"}, {"role": "user", "content": "salam"}])
    assert ids[-1] == tok.special_tokens["<|assistant_start|>"]
    assert ids.count(tok.special_tokens["<|user_start|>"]) == 1
    assert tok.decode(ids[2:-2]) == "be kind\n\nsalam"


def test_turns_must_alternate() -> None:
    with pytest.raises(ValueError):
        render_conversation(make_tokenizer(), [{"role": "assistant", "content": "x"}])
