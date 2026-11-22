"""Prepare the WhatsApp exports: drop system lines, then pseudonymize phone numbers.

1. Lines containing any literal rule in RULES are removed.
2. Every remaining phone number is replaced by a short keyed hash (``TEL_ab12cd34``).
   The key and the token-to-number table stay in ``data/private/`` (git-ignored).

Phone numbers have a tiny search space, so a plain short hash can be reversed by
enumeration. The hash is therefore an HMAC-SHA256 under a secret local key: without
``data/private/phone_key.bin`` the tokens cannot be linked back to numbers.

Before any write, every target file is copied to ``data/backups/`` (git-ignored). Those
backups still hold raw numbers: delete them once the output is verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import re
import secrets
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

RULES: tuple[str, ...] = (
    " - Votre code de sécurité avec",
    "a rejoint le groupe à partir de la communauté",
    "a été remplacé par",
    "a changé de numéro de téléphone",
    "a utilisé un lien pour rejoindre le groupe",
    "a créé le nom de profil",
    "a supprimé son nom de profil",
    "a supprimé ",
    "est parti·e",
    "a été ajouté·e",
    "a ajouté ",
    "a créé le groupe",
    "a changé la description du groupe",
    "a épinglé un message",
    "Groupe rejoint depuis la communauté",
    "chiffrés de bout en bout",
    "Ce message a été supprimé.",
    "Vous avez supprimé ce message",
    "<Médias omis>",
)

TARGET_DIRS: tuple[str, ...] = ("whatsapp_initiation_ia", "whatsapp_questions_reponses")

# International numbers: "+" then 8 to 15 digits (E.164 maximum), separated by at most
# two of space ( ) . - between digits, e.g. "+1 (438) 465-7299".
INTL_PHONE = re.compile(r"\+\d(?:[\s().-]{0,2}\d){7,14}")
# Senegalese local mobiles written without "+221" (70, 75, 76, 77, 78 prefixes). The
# look-behind keeps URL ids, dates and file names out. Other local formats are not
# matched on purpose: long digit runs in this data are mostly URL ids.
SN_LOCAL_PHONE = re.compile(r"(?<![\d/:.+=-])7[05-8][ .-]?\d{3}[ .-]?\d{2}[ .-]?\d{2}(?!\d)")

TOKEN_PREFIX = "TEL_"
MIN_TOKEN_HEX = 8


class PhoneAnonymizer:
    """Replace phone numbers by stable keyed-hash tokens, remembering the mapping."""

    def __init__(self, key: bytes, token_to_number: dict[str, str] | None = None) -> None:
        if len(key) < 16:
            raise ValueError("HMAC key must be at least 16 bytes")
        self._key = key
        self._token_to_number: dict[str, str] = dict(token_to_number or {})
        self._number_to_token = {number: token for token, number in self._token_to_number.items()}

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._token_to_number)

    def token_for(self, number: str) -> str:
        """Return the token of a normalized number, extending the hash on collisions."""
        known = self._number_to_token.get(number)
        if known is not None:
            return known
        digest = hmac.new(self._key, number.encode("ascii"), hashlib.sha256).hexdigest()
        for length in range(MIN_TOKEN_HEX, len(digest) + 1, 2):
            token = TOKEN_PREFIX + digest[:length]
            if token not in self._token_to_number:
                self._token_to_number[token] = number
                self._number_to_token[number] = token
                return token
        raise RuntimeError("unreachable: a full SHA-256 digest collided")

    def anonymize(self, text: str) -> tuple[str, int]:
        """Return the text with every phone number tokenized, and the replacement count."""
        count = 0

        def replace_intl(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return self.token_for("+" + re.sub(r"\D", "", match.group()))

        def replace_local(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return self.token_for("+221" + re.sub(r"\D", "", match.group()))

        text = INTL_PHONE.sub(replace_intl, text)
        return SN_LOCAL_PHONE.sub(replace_local, text), count

    @classmethod
    def load(cls, private_dir: Path, persist: bool) -> PhoneAnonymizer:
        """Load the key and table from private_dir, creating the key when absent.

        With persist=False (dry run) nothing is created on disk; a missing key is
        replaced by an ephemeral one.
        """
        key_path, map_path = private_dir / "phone_key.bin", private_dir / "phone_map.csv"
        if key_path.exists():
            key = key_path.read_bytes()
        else:
            key = secrets.token_bytes(32)
            if persist:
                private_dir.mkdir(parents=True, exist_ok=True)
                _write_private(key_path, key)
        table: dict[str, str] = {}
        if map_path.exists():
            with map_path.open(encoding="utf-8", newline="") as handle:
                table = {row["token"]: row["number"] for row in csv.DictReader(handle)}
        return cls(key, table)

    def save(self, private_dir: Path) -> None:
        """Write the token-to-number table (owner read/write only)."""
        private_dir.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._token_to_number.items())
        lines = ["token,number"] + [f"{token},{number}" for token, number in rows]
        _write_private(private_dir / "phone_map.csv", ("\n".join(lines) + "\n").encode("utf-8"))


def _write_private(path: Path, data: bytes) -> None:
    """Atomically write a file readable by its owner only."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    tmp.replace(path)


def matching_rule(line: str) -> str | None:
    """Return the first rule contained in the line, or None."""
    return next((rule for rule in RULES if rule in line), None)


def filter_lines(lines: list[str]) -> tuple[list[str], Counter[str]]:
    """Split lines into kept lines and per-rule removal counts."""
    kept: list[str] = []
    removed: Counter[str] = Counter()
    for line in lines:
        rule = matching_rule(line)
        if rule is None:
            kept.append(line)
        else:
            removed[rule] += 1
    return kept, removed


def process_file(path: Path, anonymizer: PhoneAnonymizer, dry_run: bool) -> tuple[Counter[str], int]:
    """Filter and pseudonymize one file in place; return removal counts and phone replacements."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, removed = filter_lines(lines)
    text, replaced = anonymizer.anonymize("".join(kept))
    if (removed or replaced) and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return removed, replaced


def backup_targets(files: list[Path], data_dir: Path, backup_root: Path) -> Path:
    """Copy target files, keeping their folder layout, into a new timestamped directory."""
    backup_dir = backup_root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for path in files:
        destination = backup_dir / path.relative_to(data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return backup_dir


def main(argv: list[str] | None = None) -> int:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Filter WhatsApp system lines and pseudonymize phone numbers.")
    parser.add_argument("--data-dir", type=Path, default=base_dir, help="Directory containing the WhatsApp folders.")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing, backing up or saving the table.")
    args = parser.parse_args(argv)

    files = sorted(path for name in TARGET_DIRS for path in (args.data_dir / name).glob("*.txt"))
    if not files:
        print(f"No .txt files found under {args.data_dir}", file=sys.stderr)
        return 1

    private_dir = args.data_dir / "private"
    anonymizer = PhoneAnonymizer.load(private_dir, persist=not args.dry_run)
    known_before = len(anonymizer.mapping)

    if not args.dry_run:
        print(f"Backup: {backup_targets(files, args.data_dir, args.data_dir / 'backups')}")

    total: Counter[str] = Counter()
    phones = 0
    for path in files:
        removed, replaced = process_file(path, anonymizer, args.dry_run)
        total.update(removed)
        phones += replaced
        print(f"{path.relative_to(args.data_dir)}: {sum(removed.values())} lines removed, {replaced} numbers replaced")

    if not args.dry_run:
        anonymizer.save(private_dir)

    print("\nRemovals per rule:")
    for rule in RULES:
        print(f"  {total.get(rule, 0):6d}  {rule}")
    print(f"  {sum(total.values()):6d}  TOTAL lines removed")
    print(f"\nPhone numbers replaced: {phones} ({len(anonymizer.mapping) - known_before} new distinct)")
    if not args.dry_run:
        print(f"Correspondence table (keep local): {private_dir / 'phone_map.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
