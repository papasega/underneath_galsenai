# Dataset documentation

## Data access and schema

The public [SFT example dataset](../data/private/supervised_fine_tuning_galsenai.json)
contains 100 conversations: 89 `complementary` and 11 `chat_grounded`.
The `private` directory name is historical; this specific file is published.
It is an example subset, not the complete experiment dataset.

**Complete datasets and trained adapters are available on request by contacting
the GalsenAI community through this repository, for example by opening a GitHub
issue. Publication on Hugging Face is planned; no release URL is available yet.**

Each JSON record contains:

| Field | Meaning |
|---|---|
| `id` | Example identifier within its source file |
| `type` | `complementary` or `chat_grounded` |
| `group_id` | Related examples kept together during splitting |
| `source_lines` | Source line references, including `{file, start, end}` ranges |
| `external_sources` | External references, when available |
| `messages` | Ordered turns with `role` and `content` |

Source references do not contain the private source text. Grounded examples may
require excerpts that are not included in the public subset.

Published preparation material:

- [Assistant seeds](../data/training_data/assistant_seed_examples.json): questions,
  answers and optional factual-check metadata.
- [v1 collection](../data/training_data/collection.json) and
  [v2 collection](../data/training_data/collection_v2.json): source labels, themes,
  seeds and split settings.
- [v2 summary](../data/training_data/pilot_v2_summary.json): counts and hashes.
- [Dataset versions and protocol](../data/training_data/PILOT_V2.md).

Collection source labels such as `../filename.json` are portable identifiers.
The preparation CLI resolves them in the directory supplied through `--source-dir`
or `GALSENAI_SOURCE_DIR`. Full generated batches and frozen splits are not included
in Git. See [training](../scripts/TRAINING.md) for the entry points.

## Data preparation and privacy

[prepare_data_galesnai.py](../data/prepare_data_galesnai.py) filters WhatsApp system
messages and pseudonymizes phone numbers with a local HMAC key. Display names are
not anonymized by this step.

[anonymize_chat.py](../scripts/anonymize_chat.py) replaces senders with random
labels, masks links, emails, mentions and phone-like numbers, and reduces dates
to year-month. It does not retain the sender mapping. Names and identifying details
in free text can remain; review the output before sharing it.

Run from the repository root with your own input files:

```bash
python scripts/anonymize_chat.py INPUT.txt OUTPUT.txt --redact-words NAMES.txt
python data/prepare_data_galesnai.py --dry-run
```

`NAMES.txt` contains one word or name to redact per line. The second command checks
the preparator's configured input; supply the required export before using it.
Raw conversations, contact mappings, generated batches and model outputs remain
private. Do not publish them without reviewing identifying information and source
permissions. Anonymization reduces identification risk but does not eliminate it.
