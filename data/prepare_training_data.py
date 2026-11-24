"""Build an immutable, source-grouped SFT pilot snapshot."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def split_ratios(config):
    ratios = config.get("split_ratios", {"train": 0.7, "valid": 0.15, "test": 0.15})
    if (set(ratios) != {"train", "valid", "test"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in ratios.values())
            or not math.isclose(sum(ratios.values()), 1)):
        raise ValueError("Split ratios must be positive train/valid/test proportions summing to one")
    return ratios


def allocate_groups(groups, config):
    """Greedily balance conversation and theme counts while keeping source components intact."""
    ratios = split_ratios(config)
    method = config.get("split_method", "hash_grouped")
    if method == "hash_grouped":
        assignments = {}
        for component, _ in groups:
            bucket = int(digest(f'{config["seed"]}:{component}'.encode())[:16], 16) / 2**64
            assignments[component] = ("train" if bucket < ratios["train"] else
                                      "valid" if bucket < ratios["train"] + ratios["valid"] else "test")
        return assignments
    if method != "balanced_grouped":
        raise ValueError(f"Unknown split method: {method}")
    totals = Counter(row["theme"] for _, members in groups for row in members)
    total = sum(totals.values())
    counts = Counter()
    themes = {split: Counter() for split in ratios}
    assignments = {}
    ordered = sorted(groups, key=lambda group: (
        -len(group[1]), digest(f'{config["seed"]}:{group[0]}'.encode())))
    for component, members in ordered:
        sizes = Counter(row["theme"] for row in members)

        def cost(split, members=members, sizes=sizes, component=component):
            target = total * ratios[split]
            delta = ((counts[split] + len(members) - target) ** 2 - (counts[split] - target) ** 2) / target
            for theme, size in sizes.items():
                target = totals[theme] * ratios[split]
                before = themes[split][theme] - target
                delta += ((before + size) ** 2 - before ** 2) / target
            return delta, digest(f'{config["seed"]}:{component}:{split}'.encode())

        split = min(ratios, key=cost)
        assignments[component] = split
        counts[split] += len(members)
        themes[split].update(sizes)
    return assignments


def source_keys(row, default_source):
    keys = set()
    lines = row.get("source_lines", [])
    if not isinstance(lines, list):
        raise ValueError("source_lines must be a list")
    for line in lines:
        if type(line) is int and line > 0:
            keys.add((default_source, line))
        elif isinstance(line, dict):
            start, end = line.get("start"), line.get("end")
            filename = line.get("file")
            if (not isinstance(filename, str) or not filename or type(start) is not int
                    or type(end) is not int or not 0 < start <= end or end - start > 100000):
                raise ValueError("Invalid source line range")
            keys.update((filename, n) for n in range(start, end + 1))
        else:
            raise ValueError("Invalid source line reference")
    return keys


def prepare(config, root, source_dir=None):
    split_ratios(config)
    rows, quarantine, inputs = [], [], {}
    for source in config["sources"]:
        source_path = Path(source["path"])
        if source_path.parts[:1] == ("..",):
            if len(source_path.parts) != 2 or source_dir is None:
                raise ValueError("Sources named ../file.json require --source-dir (or GALSENAI_SOURCE_DIR)")
            path = source_dir / source_path.name
        else:
            path = root / source_path
        raw = path.read_bytes()
        inputs[source["path"]] = digest(raw)
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data["examples"]
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list or an examples object")
        ids = set()
        for original in data:
            if not isinstance(original, dict):
                raise ValueError(f"{path}: expected an example object")
            row = dict(original)
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"{path}: missing or duplicate ID")
            ids.add(identifier)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or len(messages) % 2:
                raise ValueError(f"{path}:{identifier}: incomplete conversation")
            for i, message in enumerate(messages):
                if (not isinstance(message, dict) or message.get("role") != ("assistant" if i % 2 else "user")
                        or not isinstance(message.get("content"), str) or not message["content"].strip()):
                    raise ValueError(f"{path}:{identifier}: invalid message")
            if row.get("type") not in {"chat_grounded", "complementary", "general_llm"}:
                raise ValueError(f"{path}:{identifier}: unknown data type")
            row.update(uid=f"{path.name}:{identifier}", source_file=source["path"], theme=source["theme"])
            group = row.get("group_id")
            refs = source_keys(row, source.get("source_document", "chat_anonymized.txt"))
            if not isinstance(group, str) or not group.strip() or (row["type"] == "chat_grounded" and not refs):
                row["reason"] = "missing_source_group_or_grounding"
                quarantine.append(row)
            else:
                row["_refs"] = refs
                rows.append(row)
    rows.sort(key=lambda r: r["uid"])
    # Connected components prevent indirect source overlap across splits.
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    owners, shingles, duplicates, seen = {}, [], [], {}
    for i, row in enumerate(rows):
        questions = [normalize(m["content"]) for m in row["messages"] if m["role"] == "user"]
        keys = [("group", row["group_id"]), *[("line", *ref) for ref in row["_refs"]],
                *[("question", q) for q in questions]]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            else:
                owners[key] = i
        words = " ".join(questions).split()
        shingles.append({tuple(words[k:k + 3]) for k in range(len(words) - 2)})
        signature = tuple((m["role"], normalize(m["content"])) for m in row["messages"])
        if signature in seen:
            union(i, seen[signature])
            duplicates.append({"uid": row["uid"], "duplicate_of": rows[seen[signature]]["uid"]})
        else:
            seen[signature] = i
    near_pairs = []
    for i, left in enumerate(shingles):
        for j in range(i):
            right = shingles[j]
            if left and right and len(left & right) / len(left | right) >= 0.8:
                union(i, j)
                near_pairs.append([rows[j]["uid"], rows[i]["uid"]])
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    duplicate_ids = {r["uid"] for r in duplicates}
    splits = {name: [] for name in ("train", "valid", "test")}
    groups = []
    for members in components.values():
        component = digest("\n".join(r["uid"] for r in members).encode())
        for row in members:
            row.pop("_refs")
            row["component_id"] = component
        groups.append((component, [row for row in members if row["uid"] not in duplicate_ids]))
    assignments = allocate_groups(groups, config)
    for component, members in groups:
        splits[assignments[component]].extend(members)
    for part in splits.values():
        part.sort(key=lambda r: r["uid"])
    return splits, {"input_sha256": inputs, "quarantine": quarantine, "duplicates": duplicates,
                    "near_duplicate_pairs": near_pairs, "component_count": len(components)}


def build(config_path, output, root=ROOT, source_dir=None):
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    splits, audit = prepare(config, root, source_dir)
    files = {}
    for name, rows in splits.items():
        files[f"{name}.jsonl"] = b"".join(
            (json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n").encode() for row in rows
        )
        files[f"{name}.metadata.json"] = encode(rows)
    files["audit.json"] = encode(audit)
    files["collection.json"] = config_raw
    manifest = {
        "status": "frozen_pilot_not_human_validated", "seed": config["seed"],
        "target_ratios": split_ratios(config),
        "split_method": config.get("split_method", "hash_grouped"),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "config_sha256": digest(config_raw), "input_sha256": audit["input_sha256"],
        "counts": {name: len(rows) for name, rows in splits.items()},
        "by_theme": {name: dict(Counter(r["theme"] for r in rows)) for name, rows in splits.items()},
        "by_type": {name: dict(Counter(r["type"] for r in rows)) for name, rows in splits.items()},
        "assistant_turns": {name: sum(len(r["messages"]) // 2 for r in rows) for name, rows in splits.items()},
        "quarantine_count": len(audit["quarantine"]), "duplicate_count": len(audit["duplicates"]),
        "files_sha256": {name: digest(content) for name, content in files.items()},
    }
    files["manifest.json"] = encode(manifest)
    if output.exists():
        if all((output / name).is_file() and (output / name).read_bytes() == content
               for name, content in files.items()):
            print(f"Unchanged snapshot: {output}")
            return manifest
        raise ValueError("Snapshot is frozen or incomplete; no files changed. Review new data separately.")
    output.mkdir(parents=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    print(json.dumps({"output": str(output), "counts": manifest["counts"],
                      "quarantine": manifest["quarantine_count"]}, ensure_ascii=False))
    return manifest


@click.command(help=__doc__)
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=ROOT / "data/training_data/collection.json", show_default=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path),
              default=ROOT / "data/training_data/pilot_v1", show_default=True)
@click.option("--source-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              envvar="GALSENAI_SOURCE_DIR", help="Directory containing sources named ../file.json.")
def main(config, output, source_dir):
    try:
        build(config, output, source_dir=source_dir)
    except (ValueError, KeyError, OSError) as error:
        raise click.ClickException(f"Preparation stopped: {error}") from error


if __name__ == "__main__":
    main()
