"""Convert example conversations to MLX chat JSONL without overwriting a snapshot."""

import json
import random
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]


@click.command()
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / "data/private/supervised_fine_tuning_galsenai.json",
)
@click.option("--output", type=click.Path(path_type=Path), default=ROOT / "data/private/mlx_sft")
@click.option("--valid-size", type=click.IntRange(min=1), default=8)
@click.option("--seed", type=int, default=0)
def main(source, output, valid_size, seed):
    click.secho((ROOT / "config/dataset_notice.txt").read_text().strip(), fg="green", err=True)
    if output.exists():
        raise click.ClickException(f"Output already exists: {output}. Choose another --output.")
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(records, list) or len(records) <= valid_size:
            raise ValueError("Expected a list with more records than --valid-size")
        conversations = []
        for index, record in enumerate(records, 1):
            messages = record.get("messages") if isinstance(record, dict) else record
            if (
                not isinstance(messages, list)
                or not messages
                or any(
                    not isinstance(m, dict)
                    or m.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(m.get("content"), str)
                    or not m["content"].strip()
                    for m in messages
                )
                or not {"user", "assistant"}.issubset({m["role"] for m in messages})
            ):
                raise ValueError(f"Invalid role/content messages at record {index}")
            conversations.append({"messages": messages})
    except (ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    random.Random(seed).shuffle(conversations)
    output.mkdir(parents=True)
    for name, rows in (("valid", conversations[:valid_size]), ("train", conversations[valid_size:])):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        click.echo(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
