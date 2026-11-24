"""Derive an immutable assistant-only train/validation snapshot from frozen v2."""

import json
import re
from collections import Counter
from pathlib import Path

import click

from data.prepare_training_data import ROOT, digest, encode, normalize

# Exclude whole context-dependent examples; never trim attribution out of a target.
CONTEXT = re.compile(
    r"\b(?:un|une|le|la|ce|cet|cette)\s+(?:membre|participant|participante|passage|extrait|message)\b"
    r"|\b(?:dans|selon)\s+(?:la|cette|le|cet)\s+(?:discussion|chat|groupe|échange)\b"
    r"|\b(?:le groupe|du groupe|le chat|du chat|l’article|l'article|le site historique|sources consultées)\b",
    re.IGNORECASE,
)


def build(source, output, root=ROOT):
    source_manifest = json.loads((source / "manifest.json").read_text())
    system = (root / "config/assistant_system.txt").read_text().strip()
    seeds_raw = (root / "data/training_data/assistant_seed_examples.json").read_bytes()
    files, audit, counts, components = {}, {}, {}, {}
    splits, heldout_questions = {}, set()
    for split in ("train", "valid"):
        for name in (f"{split}.jsonl", f"{split}.metadata.json"):
            if digest((source / name).read_bytes()) != source_manifest["files_sha256"][name]:
                raise ValueError(f"Frozen input changed: {name}")
        rows = json.loads((source / f"{split}.metadata.json").read_text())
        if split == "valid":
            heldout_questions = {normalize(m["content"]) for r in rows
                                 for m in r["messages"] if m["role"] == "user"}
        exported = [json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()]
        if [r["messages"] for r in rows] != [r["messages"] for r in exported]:
            raise ValueError(f"Metadata mismatch: {split}")
        kept, rejected = [], []
        for row in rows:
            reason = ("chat_grounded_without_context" if row["type"] == "chat_grounded" else
                      "context_dependent_language" if any(CONTEXT.search(m["content"])
                                                         for m in row["messages"]) else None)
            if reason:
                rejected.append({"uid": row["uid"], "reason": reason})
            else:
                kept.append({**row, "messages": [{"role": "system", "content": system}, *row["messages"]]})
        splits[split] = kept
        components[split] = {r["component_id"] for r in rows}
        audit[split] = {"input": len(rows), "removed": rejected,
                        "reasons": dict(Counter(r["reason"] for r in rejected))}
    if components["train"] & components["valid"]:
        raise ValueError("Source components overlap")
    for seed in json.loads(seeds_raw):
        if normalize(seed["question"]) in heldout_questions:
            raise ValueError("Seed duplicates a validation question")
        splits["train"].append({"uid": f"assistant_seed:{seed['id']}", "theme": seed["theme"],
                                "component_id": "assistant_seed", "type": "assistant_seed",
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": seed["question"]},
                                             {"role": "assistant", "content": seed["answer"]}]})
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        counts[split] = len(rows)
        files[f"{split}.jsonl"] = b"".join(
            (json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n").encode() for r in rows)
        files[f"{split}.metadata.json"] = encode(rows)
    files["audit.json"] = encode(audit)
    files["manifest.json"] = encode({
        "status": "frozen_assistant_pilot_not_human_validated", "counts": counts,
        "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()), "system_sha256": digest(system.encode()),
        "seed_examples_sha256": digest(seeds_raw), "seed_examples": len(json.loads(seeds_raw)),
        "by_theme": {s: dict(Counter(r["theme"] for r in rows)) for s, rows in splits.items()},
        "test": "Not exported or inspected. Original v2 test remains unchanged; new seeds need exposure audit.",
        "files_sha256": {n: digest(b) for n, b in files.items()},
    })
    if output.exists():
        if not all((output / n).is_file() and (output / n).read_bytes() == b for n, b in files.items()):
            raise ValueError("Snapshot is frozen or incomplete; use a new output directory")
    else:
        output.mkdir(parents=True)
        for name, content in files.items():
            (output / name).write_bytes(content)
    return counts


@click.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=ROOT / "data/training_data/pilot_v2")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              default=ROOT / "data/training_data/pilot_v3_assistant")
def main(source, output):
    try:
        click.echo(json.dumps(build(source, output)))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
