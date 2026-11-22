"""Behavioural tests: each checks a property the implementation must satisfy, not its code."""

from __future__ import annotations

import pytest
import torch

from galsenai import GPT, ModelConfig, generate
from galsenai.checkpoint import load_checkpoint, save_checkpoint
from galsenai.generate import filter_logits
from galsenai.lora import get_lora_model, trainable_fraction
from galsenai.model import apply_rotary, rotary_cos_sin

TINY = dict(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def make(**overrides: object) -> GPT:
    return GPT(ModelConfig(**{**TINY, **overrides})).eval()  # type: ignore[arg-type]


def test_causality_future_tokens_do_not_change_past_logits() -> None:
    model = make()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50
    with torch.no_grad():
        la, lb = model(a).logits, model(b).logits
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"sliding_window": 4, "global_every": 2},
        {"n_kv_head": 4},
        {"n_kv_head": 1},
        {"mlp": "relu2", "qk_norm": True, "norm_affine": False, "layer_lambdas": True, "logit_softcap": 15.0},
    ],
)
def test_kv_cache_matches_full_forward(overrides: dict[str, int]) -> None:
    model = make(**overrides)
    ids = torch.randint(0, 50, (2, 16))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :6])
        steps = [out.logits]
        for t in range(6, 16):
            out = model(ids[:, t : t + 1], past=out.past)
            steps.append(out.logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_sliding_window_bounds_the_receptive_field() -> None:
    # One local layer with window 3: position t only sees t-2..t, so token 0 cannot reach t >= 3.
    model = make(n_layer=1, sliding_window=3, global_every=99)
    a = torch.randint(0, 50, (1, 10))
    b = a.clone()
    b[0, 0] = (b[0, 0] + 1) % 50
    with torch.no_grad():
        la, lb = model(a).logits, model(b).logits
    assert torch.allclose(la[:, 3:], lb[:, 3:], atol=1e-5)
    assert not torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)


def test_rope_scores_depend_only_on_relative_position() -> None:
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> torch.Tensor:
        cm, sm = rotary_cos_sin(torch.tensor([m]), 16, 10_000.0)
        cn, sn = rotary_cos_sin(torch.tensor([n]), 16, 10_000.0)
        return (apply_rotary(q, cm, sm) * apply_rotary(k, cn, sn)).sum()

    assert torch.allclose(score(3, 1), score(10, 8), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-5)


def test_parameter_count_matches_closed_form() -> None:
    cfg = ModelConfig(**TINY)
    d, hd = cfg.n_embd, cfg.head_dim
    per_layer = d * (cfg.n_head + 2 * cfg.n_kv_head) * hd + cfg.n_head * hd * d + 3 * d * cfg.hidden_dim + 2 * d
    expected = cfg.vocab_size * d + cfg.n_layer * per_layer + d  # tied head counted once
    model = GPT(cfg)
    assert model.num_params() == expected
    assert model.lm_head.weight is model.tok_emb.weight


def test_loss_ignores_padding_targets() -> None:
    model = make(ignore_index=0)
    ids = torch.randint(1, 50, (2, 10))
    targets = ids.clone()
    targets[:, 5:] = 0
    with torch.no_grad():
        out = model(ids, targets)
        ref = torch.nn.functional.cross_entropy(out.logits[:, :5].flatten(0, 1), targets[:, :5].flatten())
    assert out.loss is not None and torch.allclose(out.loss, ref, atol=1e-6)


def test_model_memorizes_a_short_sequence() -> None:
    model = make().train()
    seq = torch.randint(0, 50, (1, 17))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(150):
        opt.zero_grad()
        loss = model(seq[:, :-1], seq[:, 1:]).loss
        assert loss is not None
        loss.backward()
        opt.step()
    assert loss is not None and loss.item() < 0.1


def test_sequence_longer_than_block_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="block_size"):
        make()(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize(
    "bad",
    [
        {"n_embd": 30},
        {"n_kv_head": 3},
        {"n_embd": 36, "n_head": 4},
        {"dropout": 1.0},
        {"sliding_window": 0},
        {"vocab_size": 0},
    ],
)
def test_invalid_config_is_rejected(bad: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**{**TINY, **bad})  # type: ignore[arg-type]


def test_greedy_generation_is_deterministic_and_respects_limits() -> None:
    model = make()
    prompt = torch.randint(0, 50, (1, 5))
    a = generate(model, prompt, 10, temperature=0)
    assert torch.equal(a, generate(model, prompt, 10, temperature=0))
    assert a.shape == (1, 15) and torch.equal(a[:, :5], prompt)
    assert generate(model, prompt, 1000, temperature=0).size(1) == 32  # capped at block_size
    with pytest.raises(ValueError):
        generate(model, torch.zeros(2, 3, dtype=torch.long), 1)


def test_generation_stops_after_eos_and_restores_training_mode() -> None:
    model = make().train()
    prompt = torch.randint(0, 50, (1, 5))
    first = generate(model, prompt, 1, temperature=0)[0, -1].item()
    out = generate(model, prompt, 10, temperature=0, eos_ids={first})
    assert out.size(1) == 6 and model.training


def test_seeded_sampling_is_reproducible() -> None:
    model = make()
    prompt = torch.randint(0, 50, (1, 5))
    runs = [
        generate(model, prompt, 8, top_k=10, top_p=0.9, generator=torch.Generator().manual_seed(7)) for _ in range(2)
    ]
    assert torch.equal(runs[0], runs[1])


def test_filter_logits_keeps_top_k_and_minimal_nucleus() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isinf(filter_logits(logits, top_k=2, top_p=None)[0, 2:]).all()
    nucleus = filter_logits(logits, top_k=None, top_p=0.7)  # 0.5 < 0.7 so 0.3 is needed, 0.15 is not
    assert torch.isfinite(nucleus[0, :2]).all() and torch.isinf(nucleus[0, 2:]).all()
    assert torch.isfinite(filter_logits(logits, top_k=None, top_p=0.0)[0, 0])  # top token always survives


def test_checkpoint_roundtrip_reproduces_logits(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = make()
    path = tmp_path / "ckpt.pth"
    save_checkpoint(model, path)
    restored = load_checkpoint(path).eval()
    ids = torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        assert torch.equal(model(ids).logits, restored(ids).logits)
    assert restored.cfg == model.cfg


def test_lora_starts_as_identity_and_trains_only_adapters() -> None:
    base = make()
    lora = get_lora_model(base, rank=2, alpha=4.0).eval()
    ids = torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        assert torch.allclose(base(ids).logits, lora(ids).logits, atol=1e-6)
    trainable, total = trainable_fraction(lora)
    assert 0 < trainable < 0.2 * total
    assert all(("lora_" in n) == p.requires_grad for n, p in lora.named_parameters())
    assert all(p.requires_grad for p in base.parameters())  # the original model is untouched


def nano() -> GPT:
    return GPT(ModelConfig.nanochat(**{k: v for k, v in TINY.items() if k not in ("n_kv_head",)} | {"sliding_window": 4})).eval()


def test_nanochat_preset_has_its_distinguishing_parts() -> None:
    model = nano()
    names = {n for n, _ in model.named_parameters()}
    assert model.cfg.n_kv_head == model.cfg.n_head and model.lm_head.weight is not model.tok_emb.weight
    assert {"resid_lambdas", "x0_lambdas"} <= names and not any("norm" in n for n in names)  # parameter-free norms
    assert model.blocks[0].mlp.up.out_features == 4 * model.cfg.n_embd
    assert model(torch.randint(0, 50, (1, 8))).logits.abs().max() <= model.cfg.logit_softcap  # soft-cap bound


def test_nanochat_variant_memorizes_a_short_sequence() -> None:
    model = nano().train()
    ids = torch.randint(0, 50, (1, 17))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = model(ids[:, :-1], ids[:, 1:]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3
