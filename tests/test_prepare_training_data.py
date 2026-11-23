"""Regression tests for grouped splits and immutable snapshots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from click.testing import CliRunner

from data.prepare_training_data import build, main, prepare


def example(identifier, group, lines, question):
    return {"id": identifier, "group_id": group, "type": "chat_grounded", "source_lines": lines,
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "Answer"}]}


class PreparationTests(TestCase):
    def test_cli_rejects_missing_config(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "snapshot"
            result = CliRunner().invoke(main, ["--config", str(Path(directory) / "missing.json"),
                                                "--output", str(output)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertFalse(output.exists())

    def test_overlap_duplicates_and_missing_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [example("a", "g1", [1], "First question"),
                    example("b", "g2", [{"file": "chat_anonymized.txt", "start": 1, "end": 3}], "Second"),
                    example("c", "g3", [3], "Third"),
                    example("d", "g4", [9], "First question"),
                    example("e", None, [], "Missing")]
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
