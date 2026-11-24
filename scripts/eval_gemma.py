"""Evaluate Gemma with explicit decoding and provenance; legacy data default retained."""

import hashlib
import json
import re
import sys
from pathlib import Path

import click

from galsenai.model_sources import pinned_revision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import assistant_turns, summarize  # noqa: E402


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_history(history, system):
    if system is None:
        return history
    return [
        {"role": "system", "content": system},
        *(history[1:] if history and history[0]["role"] == "system" else history),
    ]


@click.command()
@click.option("--name", required=True)
@click.option("--model", default="mlx-community/gemma-4-E2B-it-qat-4bit", show_default=True)
@click.option("--revision", help="Immutable Hub commit; defaults to config/model_revisions.json.")
@click.option("--adapter", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / "data/private/mlx_sft/valid.jsonl",
)
@click.option("--max-new-tokens", type=click.IntRange(min=1), default=1500)
@click.option("--thinking/--no-thinking", default=True)
@click.option("--repetition-penalty", type=click.FloatRange(min=1), default=1.1)
@click.option("--seed", type=int, default=0)
@click.option("--system-prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--limit", type=click.IntRange(min=1), help="Smoke check: first N assistant turns.")
@click.option("--rag-method", type=click.Choice(["bm25", "dense", "hybrid", "oracle"]))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path))
def main(
    name,
    model,
    revision,
    adapter,
    data,
    max_new_tokens,
    thinking,
    repetition_penalty,
    seed,
    system_prompt_file,
    limit,
    rag_method,
    out,
):
    click.echo((ROOT / "config/dataset_notice.txt").read_text().strip(), err=True)
    out = out or ROOT / f"results/{name}.json"
    if out.exists():
        raise click.ClickException(f"Output already exists, choose a new --out: {out}")
    turns = assistant_turns(data)
    if not turns:
        raise click.ClickException("No assistant turns in dataset")
    ctxs = None
    if rag_method:
        ctxs = json.loads((ROOT / "results/local/rag_ctx.json").read_text())["contexts"][rag_method]
        if len(ctxs) != len(turns):
            raise click.ClickException("RAG context count differs from dataset")
    turns = turns[:limit] if limit else turns
    system = system_prompt_file.read_text().strip() if system_prompt_file else None
    try:
        revision = pinned_revision(model, revision)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    settings = dict(
        revision=revision,
        data_sha256=file_hash(data),
        model=model,
        max_new_tokens=max_new_tokens,
        thinking=thinking,
        temperature=0,
        repetition_penalty=repetition_penalty,
        seed=seed,
        limit=limit,
        rag_method=rag_method,
        system_sha256=hashlib.sha256(system.encode()).hexdigest() if system is not None else None,
    )
    adapter_hashes = (
        {p.name: file_hash(p) for p in (adapter / "adapter_config.json", adapter / "adapters.safetensors")}
        if adapter
        else {}
    )
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    network, tok = load(model, revision=revision, adapter_path=str(adapter) if adapter else None)
    rows = []
    for i, (history, ref) in enumerate(turns):
        history = prepare_history(history, system)
        if ctxs is not None:
            history = [*history[:-1], {**history[-1], "content": ctxs[i]}]
        prompt = tok.apply_chat_template(
            history, add_generation_prompt=True, tokenize=True, enable_thinking=thinking, return_dict=False
        )
        full = tok.apply_chat_template([*history, ref], tokenize=True, enable_thinking=thinking, return_dict=False)
        if full[: len(prompt)] != prompt or len(full) <= len(prompt):
            raise click.ClickException("Chat template cannot isolate assistant target")
        ids = mx.array([full])
        logits = network(ids[:, :-1])[0].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok_lp = mx.take_along_axis(logp, ids[0, 1:, None], axis=-1)[:, 0]
        sel = tok_lp[len(prompt) - 1 :]
        nll, n_tokens = -float(sel.sum()), int(sel.size)
        del logits, logp, tok_lp, sel
        mx.random.seed(seed)
        raw, last = "", None
        for response in stream_generate(
            network,
            tok,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=0),
            logits_processors=make_logits_processors(repetition_penalty=repetition_penalty),
        ):
            raw += response.text
            last = response
        pred = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", raw, flags=re.S).replace("<turn|>", "").strip()
        rows.append(
            dict(
                nll=nll,
                n_tokens=n_tokens,
                ref_bytes=len(ref["content"].encode()),
                pred=pred,
                ref=ref["content"],
                raw=raw,
                history=history,
                prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                finish_reason=last.finish_reason if last else "empty",
                generation_tokens=last.generation_tokens if last else 0,
                **({"ctx": ctxs[i]} if ctxs is not None else {}),
            )
        )
        click.echo(f"{name}: {i + 1}/{len(turns)}", err=True)
    metrics = summarize(rows)
    metrics.update(
        truncated_rate=sum(r["finish_reason"] == "length" for r in rows) / len(rows),
        narrative_marker_rate=sum(
            bool(re.search(r"\b(?:un membre|un participant|le passage|cet extrait)\b", r["pred"], re.I)) for r in rows
        )
        / len(rows),
    )
    res = dict(
        name=name,
        params_m=4628.6,
        settings=settings,
        data=str(data),
        adapter=str(adapter) if adapter else None,
        adapter_sha256=adapter_hashes,
        metrics=metrics,
        samples=rows,
    )
    serialized = json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        f.write(serialized)
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
