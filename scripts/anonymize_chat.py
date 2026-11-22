"""Anonymize a WhatsApp export for sharing: random sender labels (no mapping kept), scrubbed text, coarse dates.

    python3 scripts/anonymize_chat.py INPUT.txt OUTPUT.txt [--redact-words words.txt]

What it does: senders become U01, U02... in random order and the table is never written; sender names and
TEL_ tokens quoted inside messages become the same label; URLs, e-mails, @mentions and long digit runs are
removed; each line keeps only year-month. What it cannot do: spot other people's names or identifying details
in free text. Pass them with --redact-words (one per line) and proofread a sample before sharing.
"""

import argparse
import random
import re
import secrets
from pathlib import Path

LINE = re.compile(r"^(\d\d)/(\d\d)/(\d{4}), \d\d:\d\d - ([^:]+?): ?(.*)")
SCRUB = [
    (re.compile(r"https?://\S+|www\.\S+"), "[lien]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"@\S+"), "[mention]"),
    (re.compile(r"\+?\d[\d ().-]{6,}\d"), "[numero]"),
]
TEL = re.compile(r"TEL_[0-9a-f]{8}")


def anonymize(text: str, extra_words: list[str] = (), seed: int | None = None) -> str:
    lines = text.split("\n")
    senders = sorted({m[4] for m in map(LINE.match, lines) if m})
    order = random.Random(secrets.randbits(64) if seed is None else seed).sample(
        range(1, len(senders) + 1), len(senders)
    )
    label = {s: f"U{n:02d}" for s, n in zip(senders, order, strict=True)}
    # Full names and name parts both map to the sender's label; longest first so "Jean Dupont" beats "Jean".
    parts = {
        w.lower(): label[s]
        for s in senders
        if not s.startswith("TEL_")
        for w in (s, *s.replace(",", " ").split())
        if len(w) > 2
    }
    name_re = re.compile("|".join(map(re.escape, sorted(parts, key=len, reverse=True))), re.I) if parts else None
    redact = (
        re.compile("|".join(map(re.escape, sorted(extra_words, key=len, reverse=True))), re.I) if extra_words else None
    )

    def clean(s: str) -> str:
        s = TEL.sub(lambda m: label.get(m.group(), "[membre]"), s)
        if name_re:
            s = name_re.sub(lambda m: parts[m.group().lower()], s)
        if redact:
            s = redact.sub("[nom]", s)
        for rx, repl in SCRUB:
            s = rx.sub(repl, s)
        return s

    out = []
    for line in lines:
        m = LINE.match(line)
        out.append(f"{m[3]}-{m[2]} | {label[m[4]]}: {clean(m[5])}" if m else clean(line))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--redact-words", type=Path, help="file with one extra word or name to remove per line")
    a = ap.parse_args()
    words = (
        [w.strip() for w in a.redact_words.read_text(encoding="utf-8").splitlines() if w.strip()]
        if a.redact_words
        else []
    )
    a.dst.write_text(anonymize(a.src.read_text(encoding="utf-8"), words), encoding="utf-8")
    print(f"wrote {a.dst}")
