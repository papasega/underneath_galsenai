"""Convert the SFT JSON (list of conversations) to mlx-lm chat JSONL with a fixed train/valid split."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/private/supervised_fine_tuning_galsenai.json"
OUT = ROOT / "data/private/mlx_sft"
N_VALID = 8

convs = json.loads(SRC.read_text(encoding="utf-8"))
random.Random(0).shuffle(convs)
OUT.mkdir(parents=True, exist_ok=True)
for name, part in (("valid", convs[:N_VALID]), ("train", convs[N_VALID:])):
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for c in part:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    print(name, len(part))
