"""Apply reviewed answer replacements to frozen v3 training rows, preserving validation."""

import csv
import io
import json
from pathlib import Path

import click

from data.prepare_training_data import ROOT, digest, encode, normalize


def build(source, review, output):
    provenance = {}

    def checked(path, expected):
        raw = path.read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        provenance[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = expected
        return raw

    manifest = json.loads((source / "manifest.json").read_text())
    rows, files = {}, {}
    for split in ("train", "valid"):
        for suffix in ("jsonl", "metadata.json"):
            name = f"{split}.{suffix}"
            files[name] = checked(source / name, manifest["files_sha256"][name])
        rows[split] = json.loads(files[f"{split}.metadata.json"])
        if [r["messages"] for r in rows[split]] != [
                json.loads(line)["messages"] for line in files[f"{split}.jsonl"].splitlines()]:
            raise ValueError(f"Metadata mismatch: {split}")
    delivery = json.loads((review / "manifest.json").read_text())
    for name, expected in delivery["files"].items():
        checked(review / name, expected)
    review_csv = checked(review.parent / "review_corrected.csv", delivery["review_corrected_sha256"])
    heldout = {normalize(r["question"]) for r in csv.DictReader(io.StringIO(review_csv.decode("utf-8-sig")))}
    heldout.update(normalize(m["content"]) for r in rows["valid"] for m in r["messages"] if m["role"] == "user")
    train = rows["train"]
    if any(normalize(m["content"]) in heldout for r in train for m in r["messages"] if m["role"] == "user"):
        raise ValueError("Training question overlaps reviewed validation")
    if {r["component_id"] for r in train} & {r["component_id"] for r in rows["valid"]}:
        raise ValueError("Train/validation components overlap")
    candidates = json.loads((review / "dataset_candidate/sft_llm_general.json").read_text())
    candidates = {r["id"]: r for r in candidates}
    applied, absent = [], []
    with (review / "dataset_corrections_log.csv").open(encoding="utf-8-sig", newline="") as f:
        for correction in csv.DictReader(f):
            if correction["file"] != "sft_llm_general.json":
                raise ValueError("Unsupported correction source")
            uid = f"sft_llm_general_100.json:{correction['id']}"
            matches = [r for r in train if r["uid"] == uid]
            if not matches:
                absent.append(uid)
                continue
            if len(matches) != 1:
                raise ValueError(f"Ambiguous correction: {uid}")
            row = matches[0]
            candidate = candidates[correction["id"]]
            def questions(messages):
                return [m["content"] for m in messages if m["role"] == "user"]

            if questions(row["messages"]) != questions(candidate["messages"]):
                raise ValueError(f"Correction question changed: {uid}")
            answers = [m for m in row["messages"] if m["role"] == "assistant"]
            targets = [m for m in candidate["messages"] if m["role"] == "assistant"]
            if (len(answers) != 1 or len(targets) != 1 or answers[0]["content"] != correction["old_answer"]
                    or targets[0]["content"] != correction["corrected_answer"]):
                raise ValueError(f"Correction target mismatch: {uid}")
            answers[0]["content"] = correction["corrected_answer"]
            applied.append(uid)
    if not applied:
        raise ValueError("No training corrections applied")
    files["train.metadata.json"] = encode(train)
    files["train.jsonl"] = b"".join(
        (json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n").encode() for r in train)
    files["audit.json"] = encode({"applied": applied, "absent_from_filtered_train": absent,
                                  "validation": "Preserved byte for byte; no review answers added to train."})
    counts = {s: len(r) for s, r in rows.items()}
    files["manifest.json"] = encode({
        "status": "frozen_review_corrected_development_snapshot", "counts": counts,
        "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
        "review_manifest_sha256": digest((review / "manifest.json").read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()), "input_sha256": provenance,
        "corrections_applied": len(applied), "test": "Not read or exported; no independent test claim.",
        "files_sha256": {n: digest(b) for n, b in files.items()},
    })
    if output.exists():
        if not all((output / n).is_file() and (output / n).read_bytes() == b for n, b in files.items()):
            raise ValueError("Snapshot is frozen or incomplete; use a new output directory")
    else:
        output.mkdir(parents=True)
        for name, raw in files.items():
            (output / name).write_bytes(raw)
    return {"counts": counts, "corrections_applied": len(applied)}


@click.command()
@click.option("--source", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=ROOT / "data/training_data/pilot_v3_assistant")
@click.option("--review", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=ROOT / "results/local/assistant-eval-GI297S/dataset_review_corrections")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              default=ROOT / "data/training_data/pilot_v4_assistant")
def main(source, review, output):
    try:
        click.echo(json.dumps(build(source, review, output)))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
