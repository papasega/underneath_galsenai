"""Tests of data/prepare_data_galesnai.py: filtering and phone pseudonymization."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "data" / "prepare_data_galesnai.py"
KEY = bytes(range(32))


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_data_galesnai", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_script()


def token(number: str) -> str:
    return prep.PhoneAnonymizer(KEY).token_for(number)


@pytest.mark.parametrize(
    "raw",
    ["+221 78 100 00 01", "+221781000001", "+221-78-100-00-01", "‎+221 78 100 00 01", "781000002 Wave"],
)
def test_senegal_formats_share_one_token(raw: str) -> None:
    anon = prep.PhoneAnonymizer(KEY)
    expected = anon.token_for("+221781000001") if "781000002" not in raw else anon.token_for("+221781000002")
    out, count = anon.anonymize(raw)
    assert count == 1 and expected in out and "+221" not in out and "781000002" not in out


@pytest.mark.parametrize("raw", ["+1 (555) 010-0199", "+254 700 000001", "+44 7700 900123", "+229 01 00 00 00 01"])
def test_other_international_formats_are_replaced(raw: str) -> None:
    out, count = prep.PhoneAnonymizer(KEY).anonymize(f"12/05/2026, 10:00 - {raw}: salut")
    assert count == 1 and out.startswith("12/05/2026, 10:00 - TEL_") and out.endswith(": salut")


@pytest.mark.parametrize(
    "text",
    [
        "1+1=2 en francais",
        "à +10ans d'exp",
        "https://x.com/i/status/1905947722206564801",
        "https://www.shopmeaway.com/p/1098166302?k=AI",
        "IMG-20260421-WA0004.jpg",
        "21/04/2026, 19:50 - Bob: rdv à 19:50",
        "prix 1 500 000 xof",
    ],
)
def test_non_phone_text_is_left_alone(text: str) -> None:
    assert prep.PhoneAnonymizer(KEY).anonymize(text) == (text, 0)


def test_token_is_stable_keyed_and_short() -> None:
    a, b = token("+221781000001"), token("+221781000001")
    assert a == b and a.startswith("TEL_") and len(a) == len("TEL_") + 8
    other_key = prep.PhoneAnonymizer(bytes(32)).token_for("+221781000001")
    assert other_key != a  # unkeyed enumeration cannot reproduce tokens
    assert token("+221781000003") != a


def test_hash_collision_extends_the_token() -> None:
    first = token("+221781000001")
    anon = prep.PhoneAnonymizer(KEY, {first: "+999999999999"})  # another number already owns that token
    longer = anon.token_for("+221781000001")
    assert longer != first and longer.startswith(first) and anon.mapping[longer] == "+221781000001"


def test_short_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        prep.PhoneAnonymizer(b"short")


def test_rules_remove_system_lines_only() -> None:
    kept, removed = prep.filter_lines(
        ["a: salut\n", "x - Votre code de sécurité avec y\n", "b: <Médias omis>\n", "c: ok\n"]
    )
    assert kept == ["a: salut\n", "c: ok\n"] and sum(removed.values()) == 2


def make_dataset(root: Path) -> Path:
    for name in prep.TARGET_DIRS:
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "chat.txt").write_text(
            "01/01/2026, 10:00 - +221 78 100 00 01: salut\n"
            "01/01/2026, 10:01 - +221 78 100 00 01: <Médias omis>\n"
            "01/01/2026, 10:02 - Awa: appelle le 781000002\n",
            encoding="utf-8",
        )
    return root


def test_end_to_end_pseudonymizes_backs_up_and_is_idempotent(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    assert prep.main(["--data-dir", str(root)]) == 0

    for name in prep.TARGET_DIRS:
        text = (root / name / "chat.txt").read_text(encoding="utf-8")
        assert "+221" not in text and "781000002" not in text and "Médias" not in text
        assert text.count("TEL_") == 2  # one sender and one in-text number; the media line is gone

    backups = list((root / "backups").glob("backup_*/*/chat.txt"))
    assert len(backups) == 2 and "+221 78 100 00 01" in backups[0].read_text(encoding="utf-8")

    table = root / "private" / "phone_map.csv"
    assert stat.S_IMODE((root / "private" / "phone_key.bin").stat().st_mode) == 0o600
    assert stat.S_IMODE(table.stat().st_mode) == 0o600
    rows = table.read_text(encoding="utf-8").splitlines()[1:]
    assert {r.split(",")[1] for r in rows} == {"+221781000001", "+221781000002"}

    snapshot = (root / prep.TARGET_DIRS[0] / "chat.txt").read_bytes()
    assert prep.main(["--data-dir", str(root)]) == 0  # second run changes nothing
    assert (root / prep.TARGET_DIRS[0] / "chat.txt").read_bytes() == snapshot


def test_tokens_are_consistent_across_runs(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    prep.main(["--data-dir", str(root)])
    first = (root / prep.TARGET_DIRS[0] / "chat.txt").read_text(encoding="utf-8")
    second_root = make_dataset(tmp_path / "again")
    (second_root / "private").mkdir()
    for fname in ("phone_key.bin", "phone_map.csv"):
        (second_root / "private" / fname).write_bytes((root / "private" / fname).read_bytes())
    prep.main(["--data-dir", str(second_root)])
    assert (second_root / prep.TARGET_DIRS[0] / "chat.txt").read_text(encoding="utf-8") == first


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    root = make_dataset(tmp_path)
    before = (root / prep.TARGET_DIRS[0] / "chat.txt").read_bytes()
    assert prep.main(["--data-dir", str(root), "--dry-run"]) == 0
    assert (root / prep.TARGET_DIRS[0] / "chat.txt").read_bytes() == before
    assert not (root / "private").exists() and not (root / "backups").exists()


def test_missing_input_fails_with_nonzero_exit(tmp_path: Path) -> None:
    assert prep.main(["--data-dir", str(tmp_path)]) == 1
