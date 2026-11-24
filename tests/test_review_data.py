import csv
import json

import pytest

from data.prepare_review_data import build
from data.prepare_training_data import digest, encode


def inputs(tmp_path):
    source = tmp_path / "v3"
    source.mkdir()
    hashes = {}
    for split in ("train", "valid"):
        row = {"uid": f"sft_llm_general_100.json:{split}", "component_id": split,
               "messages": [{"role": "system", "content": "system"},
                            {"role": "user", "content": split},
                            {"role": "assistant", "content": "old"}]}
        for suffix, raw in (("metadata.json", encode([row])),
                            ("jsonl", (json.dumps({"messages": row["messages"]}) + "\n").encode())):
            name = f"{split}.{suffix}"
            (source / name).write_bytes(raw)
            hashes[name] = digest(raw)
    (source / "manifest.json").write_bytes(encode({"files_sha256": hashes}))
    review = tmp_path / "delivery"
    (review / "dataset_candidate").mkdir(parents=True)
    (review / "dataset_candidate/sft_llm_general.json").write_bytes(encode([{
        "id": "train", "messages": [{"role": "user", "content": "train"},
                                     {"role": "assistant", "content": "new"}]}]))
    with (review / "dataset_corrections_log.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "id", "old_answer", "corrected_answer"])
        writer.writerow(["sft_llm_general.json", "train", "old", "new"])
        writer.writerow(["sft_llm_general.json", "absent", "old", "new"])
    csv_path = tmp_path / "review_corrected.csv"
    csv_path.write_text("question\nvalid\n")
    manifest = {"files": {str(p.relative_to(review)): digest(p.read_bytes())
                           for p in review.rglob("*") if p.is_file()},
                "review_corrected_sha256": digest(csv_path.read_bytes())}
    (review / "manifest.json").write_bytes(encode(manifest))
    return source, review, tmp_path / "v4"


def test_review_snapshot_changes_only_target_and_preserves_validation(tmp_path):
    source, review, output = inputs(tmp_path)
    result = build(source, review, output)
    assert result == {"counts": {"train": 1, "valid": 1}, "corrections_applied": 1}
    old = json.loads((source / "train.jsonl").read_text())["messages"]
    new = json.loads((output / "train.jsonl").read_text())["messages"]
    assert old[:-1] == new[:-1]
    assert old[-1]["content"] == "old" and new[-1]["content"] == "new"
    for name in ("valid.jsonl", "valid.metadata.json"):
        assert (source / name).read_bytes() == (output / name).read_bytes()
    assert build(source, review, output) == result
    (output / "train.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="frozen"):
        build(source, review, output)


def test_review_snapshot_rejects_altered_delivery(tmp_path):
    source, review, output = inputs(tmp_path)
    (review / "dataset_corrections_log.csv").write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(source, review, output)
    assert not output.exists()


def test_review_snapshot_rejects_review_questions_in_train(tmp_path):
    source, review, output = inputs(tmp_path)
    csv_path = tmp_path / "review_corrected.csv"
    csv_path.write_text("question\ntrain\n")
    manifest = json.loads((review / "manifest.json").read_text())
    manifest["review_corrected_sha256"] = digest(csv_path.read_bytes())
    (review / "manifest.json").write_bytes(encode(manifest))
    with pytest.raises(ValueError, match="overlaps"):
        build(source, review, output)
    assert not output.exists()
