"""Resolve a pinned snapshot before invoking an MLX CLI without --revision support."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import click
import yaml

from galsenai.model_sources import pinned_revision


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("command", type=click.Choice(["lora", "server", "fuse"]))
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def main(command, arguments):
    args = list(arguments)

    def option(name, default=None):
        for i, arg in enumerate(args):
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
            if arg == name:
                if i + 1 == len(args):
                    raise click.ClickException(f"Missing value for {name}")
                return args[i + 1]
        return default

    config_path = option("--config")
    config = yaml.safe_load(Path(config_path).read_text()) if config_path else {}
    model = option("--model", config.get("model"))
    if not model:
        raise click.ClickException("Supply --model or a model in --config")
    try:
        revision = pinned_revision(model, option("--revision"))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # MLX CLI does not accept --revision. Remove it before forwarding.
    for i in range(len(args) - 1, -1, -1):
        if args[i].startswith("--revision="):
            del args[i]
        elif args[i] == "--revision":
            del args[i : i + 2]
    resolved = model
    if revision:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

        resolved = snapshot_download(model, revision=revision, allow_patterns=DEFAULT_ALLOW_PATTERNS)
    # The last --model wins in argparse, including over the YAML value.
    args.extend(["--model", resolved])
    invocation = [sys.executable, "-m", f"mlx_lm.{command}", *args]
    if command != "lora":
        os.execv(sys.executable, invocation)
    else:
        subprocess.run(invocation, check=True)
    if command == "lora":
        output = Path(option("--adapter-path", config.get("adapter_path", "adapters")))
        data = Path(option("--data", config.get("data", "data")))
        provenance = dict(
            model=model,
            revision=revision,
            arguments=args,
            config=config,
            versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "transformers")},
            inputs={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (data / "train.jsonl", data / "valid.jsonl")
                if p.is_file()
            },
        )
        (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
