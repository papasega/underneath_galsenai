import csv
import json

import pytest
from click.testing import CliRunner

from scripts.eval_assistant_report import report
from scripts.eval_gemma import main, prepare_history


def test_system_prompt_replaced_without_mutating_history():
    history = [{"role": "system", "content": "old"}, {"role": "user", "content": "Q"}]
    assert prepare_history(history, "new") == [{"role": "system", "content": "new"}, history[1]]
    assert history[0]["content"] == "old"
    assert prepare_history(history[1:], "new")[0]["role"] == "system"
    assert prepare_history(history, None) == history


def test_eval_refuses_overwrite_before_loading_model(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("")
    out = tmp_path / "out.json"
    out.write_text("preserve")
    result = CliRunner().invoke(main, ["--name", "test", "--data", str(data), "--out", str(out)])
    assert result.exit_code != 0
    assert "Output already exists" in result.output
    assert out.read_text() == "preserve"


def comparison(tmp_path):
    for name in ("base", "v2", "v3"):
        result = dict(
            name=name,
            settings={"thinking": False},
            metrics=dict.fromkeys(
                (
                    "bits_per_byte",
                    "rouge_l",
                    "token_f1",
                    "distinct_2",
                    "empty_rate",
                    "truncated_rate",
                    "narrative_marker_rate",
                ),
                0.5,
            ),
            samples=[
                {
                    "prompt_sha256": "abc",
                    "ref": "R",
                    "pred": name,
                    "history": [{"role": "user", "content": "Q, avec virgule"}],
                }
            ],
        )
        (tmp_path / f"{name}.json").write_text(json.dumps(result))


def test_report_aligns_answers_and_preserves_existing_outputs(tmp_path):
    comparison(tmp_path)
    report(tmp_path)
    with (tmp_path / "review.csv").open() as f:
        row = list(csv.DictReader(f))[0]
    assert row["question"] == "Q, avec virgule"
    assert [row[k] for k in ("base", "v2", "v3")] == ["base", "v2", "v3"]
    original = (tmp_path / "comparison.md").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        report(tmp_path)
    assert (tmp_path / "comparison.md").read_bytes() == original


@pytest.mark.parametrize("mismatch", ["settings", "prompt_sha256", "ref"])
def test_report_rejects_incomparable_results(tmp_path, mismatch):
    comparison(tmp_path)
    path = tmp_path / "v3.json"
    result = json.loads(path.read_text())
    if mismatch == "settings":
        result["settings"]["thinking"] = True
    else:
        result["samples"][0][mismatch] = "different"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Results differ"):
        report(tmp_path)
    assert not (tmp_path / "comparison.md").exists()
