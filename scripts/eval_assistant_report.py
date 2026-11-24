"""Report an isolated base/v2/v3 comparison; reject mismatched conditions."""

import csv
import json
from pathlib import Path

import click


def report(directory):
    results = [json.loads((directory / f"{name}.json").read_text()) for name in ("base", "v2", "v3")]
    first = results[0]
    if not first["samples"]:
        raise ValueError("Empty comparison")

    def signatures(result):
        return [(s["prompt_sha256"], s["ref"]) for s in result["samples"]]

    for r in results:
        if r["settings"] != first["settings"] or signatures(r) != signatures(first):
            raise ValueError("Results differ in data, prompts, references or decoding settings")
    metrics = ("bits_per_byte", "rouge_l", "token_f1", "distinct_2", "empty_rate",
               "truncated_rate", "narrative_marker_rate")
    lines = ["# Assistant validation comparison", "",
             f"{len(first['samples'])} validation turns; temperature 0, matched prompts and decoding.", "",
             "| Model | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for r in results:
        lines.append("| " + r["name"] + " | " + " | ".join(f"{r['metrics'][m]:.4f}" for m in metrics) + " |")
    lines += ["", "Narrative markers are a review flag, not an automatic correctness verdict.",
              "ROUGE/F1 measure word overlap, not factual correctness. Review answers side by side.",
              "This filtered validation set is for development; it is not a final test result."]
    for name in ("comparison.md", "review.csv"):
        if (directory / name).exists():
            raise ValueError(f"Output already exists: {name}")
    with (directory / "comparison.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    with (directory / "review.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "question", "reference", "base", "v2", "v3",
                         "base_correct", "v2_correct", "v3_correct", "notes"])
        for i, samples in enumerate(zip(*(r["samples"] for r in results), strict=True)):
            history = samples[0]["history"]
            question = next(m["content"] for m in reversed(history) if m["role"] == "user")
            writer.writerow([i + 1, question, samples[0]["ref"], *(s["pred"] for s in samples), "", "", "", ""])


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(directory):
    try:
        report(directory)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote {directory / 'comparison.md'} and review.csv")


if __name__ == "__main__":
    main()
