"""Text-only LoRA SFT with Transformers/PEFT on a single CUDA GPU or CPU."""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import click

from galsenai.model_sources import pinned_revision

ROOT = Path(__file__).resolve().parents[1]


def encode_split(path, tokenizer, max_length):
    """Mask history tokens and score each assistant turn, including its end marker."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            messages = json.loads(line)["messages"]
            if not messages or any(
                m["role"] not in {"system", "user", "assistant"} or not isinstance(m["content"], str) for m in messages
            ):
                raise ValueError("Expected nonempty role/content messages")
            targets = 0
            for i, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                if not i or messages[i - 1]["role"] != "user" or not message["content"].strip():
                    raise ValueError("Each assistant answer must follow a user message and contain text")
                options = dict(tokenize=True, enable_thinking=False, return_dict=False)
                prompt = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **options)
                full = tokenizer.apply_chat_template(messages[: i + 1], **options)
                if full[: len(prompt)] != prompt or len(full) <= len(prompt):
                    raise ValueError("Chat template cannot isolate assistant tokens")
                if len(full) > max_length:
                    raise ValueError(f"{len(full)} tokens exceed --max-length {max_length}; no silent truncation")
                rows.append(
                    dict(
                        input_ids=full,
                        attention_mask=[1] * len(full),
                        labels=[-100] * len(prompt) + full[len(prompt) :],
                    )
                )
                targets += 1
            if not targets:
                raise ValueError("Conversation has no assistant answer")
        except (ValueError, KeyError, TypeError) as exc:
            raise click.ClickException(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise click.ClickException(f"Empty dataset: {path}")
    return rows


@click.command()
@click.option("--model", default="google/gemma-4-E2B-it", show_default=True)
@click.option("--revision", help="Immutable Hub commit; defaults to config/model_revisions.json.")
@click.option("--data", type=click.Path(path_type=Path), default=ROOT / "data/training_data/pilot_v4_assistant")
@click.option(
    "--output", type=click.Path(path_type=Path), required=True, help="New adapter directory, never overwritten."
)
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto")
@click.option("--epochs", type=click.FloatRange(min=0, min_open=True), default=1.0)
@click.option("--batch-size", type=click.IntRange(min=1), default=1)
@click.option("--gradient-accumulation", type=click.IntRange(min=1), default=2)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), default=5e-5)
@click.option("--max-length", type=click.IntRange(min=2), default=2048)
@click.option("--rank", type=click.IntRange(min=1), default=8)
@click.option("--target-modules", default="q_proj,v_proj", help="Comma-separated PEFT module names; model dependent.")
@click.option("--max-steps", type=click.IntRange(min=1), help="Optional short smoke run; overrides epochs.")
def main(
    model,
    revision,
    data,
    output,
    device,
    epochs,
    batch_size,
    gradient_accumulation,
    learning_rate,
    max_length,
    rank,
    target_modules,
    max_steps,
):
    click.secho((ROOT / "config/dataset_notice.txt").read_text().strip(), fg="green", err=True)
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}. Choose a new directory.")
    paths = [data / f"{split}.jsonl" for split in ("train", "valid")]
    for path in paths:
        if not path.is_file():
            raise click.ClickException(f"Missing private snapshot: {path}. Request dataset access first.")
    modules = [name.strip() for name in target_modules.split(",") if name.strip()]
    if not modules:
        raise click.ClickException("--target-modules must contain at least one module name")
    if sys.version_info < (3, 11):
        raise click.ClickException("HF training requires Python 3.11 or newer.")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise click.ClickException('Install training dependencies: pip install -e ".[hf-train]"') from exc
    cuda = torch.cuda.is_available() and device != "cpu"
    if device == "cuda" and not cuda:
        raise click.ClickException(
            "CUDA requested but unavailable. Install a matching CUDA PyTorch build or use --device cpu."
        )
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32
    click.echo(f"Device: {'cuda' if cuda else 'cpu'}; dtype: {dtype}", err=True)
    set_seed(0)
    try:
        revision = pinned_revision(model, revision)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise click.ClickException("Tokenizer requires a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train, valid = [encode_split(path, tokenizer, max_length) for path in paths]
    network = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype, attn_implementation="sdpa")
    network.config.use_cache = False
    network = get_peft_model(
        network,
        LoraConfig(
            task_type="CAUSAL_LM", r=rank, lora_alpha=rank * 2, lora_dropout=0.0, target_modules=modules, bias="none"
        ),
    )
    network.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=cuda and not bf16,
        use_cpu=not cuda,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=0,
        data_seed=0,
    )
    trainer = Trainer(
        model=network,
        args=args,
        train_dataset=train,
        eval_dataset=valid,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    provenance = dict(
        model=model,
        revision=revision,
        device="cuda" if cuda else "cpu",
        dtype=str(dtype),
        training_args=args.to_dict(),
        train_targets=len(train),
        valid_targets=len(valid),
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
    )
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
