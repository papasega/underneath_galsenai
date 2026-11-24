"""Regression tests for grouped splits and immutable snapshots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from click.testing import CliRunner

from data.prepare_training_data import allocate_groups, build, main, prepare, split_ratios


def example(identifier, group, lines, question):
    return {
        "id": identifier,
        "group_id": group,
        "type": "chat_grounded",
        "source_lines": lines,
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "Answer"}],
    }


class PreparationTests(TestCase):
    def test_balanced_allocation_preserves_groups_and_covers_themes(self):
        groups = [(f"{theme}_{i}", [{"theme": theme}] * 2) for theme in ("voice", "agents") for i in range(50)]
        config = {
            "seed": 42,
            "split_method": "balanced_grouped",
            "split_ratios": {"train": 0.8, "valid": 0.1, "test": 0.1},
        }
        assignments = allocate_groups(groups, config)
        self.assertEqual(assignments, allocate_groups(list(reversed(groups)), config))
        for theme in ("voice", "agents"):
            for split, expected in (("train", 80), ("valid", 10), ("test", 10)):
                count = sum(
                    len(rows) for key, rows in groups if rows[0]["theme"] == theme and assignments[key] == split
                )
                self.assertEqual(count, expected)

    def test_rejects_invalid_ratios(self):
        for ratios in (
            {"train": 0.8, "valid": 0.1, "test": 0.2},
            {"train": 1, "valid": 0, "test": 0},
            {"train": 0.8},
            {"train": float("nan"), "valid": 0.1, "test": 0.1},
        ):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                split_ratios({"split_ratios": ratios})

    def test_cli_rejects_missing_config(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot"
            result = CliRunner().invoke(
                main, ["--config", str(Path(directory) / "missing.json"), "--output", str(output)]
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse(output.exists())

    def test_overlap_duplicates_and_missing_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                example("a", "g1", [1], "First question"),
                example("b", "g2", [{"file": "chat_anonymized.txt", "start": 1, "end": 3}], "Second"),
                example("c", "g3", [3], "Third"),
                example("d", "g4", [9], "First question"),
                example("e", None, [], "Missing"),
            ]
            (root / "input.json").write_text(json.dumps({"examples": rows}))
            config = {"seed": 4, "sources": [{"path": "input.json", "theme": "test"}]}
            splits, audit = prepare(config, root)
            kept = [row for part in splits.values() for row in part]
            self.assertEqual(len(kept), 3)
            self.assertEqual(len({row["component_id"] for row in kept}), 1)
            self.assertEqual(sum(bool(part) for part in splits.values()), 1)
            self.assertEqual(len(audit["quarantine"]), 1)
            self.assertEqual(len(audit["duplicates"]), 1)
            self.assertEqual(splits, prepare(config, root)[0])

    def test_snapshot_is_immutable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(json.dumps([example("a", "g", [1], "Q")]))
            config = root / "config.json"
            config.write_text(json.dumps({"seed": 4, "sources": [{"path": "input.json", "theme": "test"}]}))
            output = root / "snapshot"
            build(config, output, root)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            build(config, output, root)
            source.write_text(json.dumps([example("a", "g", [1], "Changed")]))
            with self.assertRaises(ValueError):
                build(config, output, root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_rejects_malformed_messages_and_ranges(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"seed": 4, "sources": [{"path": "input.json", "theme": "test"}]}
            for row in [example("a", "g", [-1], "Q"), example("a", "g", [1], "")]:
                (root / "input.json").write_text(json.dumps([row]))
                with self.assertRaises(ValueError):
                    prepare(config, root)

    def test_portable_source_names_use_explicit_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "external_sources"
            sources.mkdir()
            (sources / "input.json").write_text(json.dumps([example("a", "g", [1], "Q")]))
            config = {"seed": 4, "sources": [{"path": "../input.json", "theme": "test"}]}
            with self.assertRaisesRegex(ValueError, "source-dir"):
                prepare(config, root)
            splits, audit = prepare(config, root, sources)
            self.assertEqual(sum(map(len, splits.values())), 1)
            self.assertEqual(list(audit["input_sha256"]), ["../input.json"])
            self.assertEqual(next(r for rows in splits.values() for r in rows)["source_file"], "../input.json")
            config["sources"][0]["path"] = "../../input.json"
            with self.assertRaisesRegex(ValueError, "source-dir"):
                prepare(config, root, sources)
