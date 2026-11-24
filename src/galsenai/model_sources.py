"""Immutable Hub revisions shared by training, evaluation and serving."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pinned_revision(model: str, revision: str | None = None) -> str | None:
    if Path(model).is_dir():
        return None
    pins = json.loads((ROOT / "config/model_revisions.json").read_text())
    revision = revision or pins.get(model)
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Supply an immutable 40-character --revision for Hub model {model}")
    return str(revision)
