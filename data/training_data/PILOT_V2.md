# Pilot v2

Branch: `pilot-v2`. This is a new exploratory benchmark version. The frozen v1 files
are retained; comparisons between v1 and v2 scores are not paired comparisons.

## Data

All 16 corrected JSON source batches listed as `../filename.json` in `collection_v2.json` are included,
100 conversations each, including `supervised_fine_tuning_galsenai_complete_corrige.json`
as `galsenai_general`, following the user's explicit inclusion decision.
There are 1,600 candidates, zero exact conversation duplicates removed and zero
quarantined records under the current schema/provenance rules. This is structural
validation, not independent factual or linguistic validation.

The 16 source themes are recorded in `collection_v2.json`. Community participation
is its own theme; evaluation/reliability is not a separate 100-example batch.
The original 15-theme target from v1 is not an achieved coverage claim for v2.
Source ranges all name `chat_anonymized.txt`; their factual support and numbering
still require human checking against the exact version used during generation.

Using the unchanged seed and a balanced source-component allocation targeting
80/10/10 yields:

| Split | Conversations |
|---|---:|
| Train | 1,266 |
| Validation | 167 |
| Test | 167 |

There are 284 independent components under the implemented grouping rules.
The actual proportions are 79.125/10.4375/10.4375 because source groups stay intact.
There are 222 train, 29 validation and 33 test components. Each validation/test theme
has 8 to 15 conversations. These are correlated examples, not 167 independent sources.
Per-theme/type counts and input hashes are in `pilot_v2_summary.json`; the full local
snapshot manifest additionally records file hashes and supervised assistant-turn counts.

## Reproduction

Use the project environment, which includes Click:

```sh
uv sync --extra dev
.venv/bin/python data/prepare_training_data.py \
  --config data/training_data/collection_v2.json \
  --source-dir /path/to/source-files \
  --output data/training_data/pilot_v2
```

The original collection config and CLI defaults still name v1; always pass both v2
arguments. Old input filenames were replaced during ongoing collection, so rebuilding
v1 from the current incoming directory is not supported. Its frozen bytes and hashes
remain available. The initial 1,500-example provisional export is retained locally
as `pilot_v2_provisional_1500/` and must not be used for experiments.

## Evaluation contract

Follow the review, prompt locking, training-seed, paired component-bootstrap and RAG
requirements in `README.md`. Independently review the v2 test and freeze its rubric
before evaluating models. Do not tune on this test.

Fifteen v1 test conversations share source lines with v2 training data; 26 have an
identical normalized question in v2 training. These counts overlap and must not be
summed. The v1 test is therefore unsuitable for reporting clean performance of a
v2-trained model. All baselines must be evaluated on the v2 test under the same
conditions; historical adapters need a separate training-exposure audit.

Within v2, source-line sets, exact normalized questions and detected source components
are disjoint between all split pairs. Semantic paraphrases and undisclosed pretraining
exposure remain outside that guarantee. Human factual review, tokenizer lengths,
model training and final test evaluation have not been performed by this preparation.

## Pre-training split revision

The original v2 hash allocation (1,026/256/318) was superseded before any v2 training
or model-output evaluation. It remains locally in `pilot_v2_superseded_hash_split/`;
a preparation draft is retained in `pilot_v2_balanced_draft/`. Neither is active.
All comparisons must use the active `pilot_v2/` files and their new fingerprints.
The v1 snapshot is unchanged.

We compared data-only coverage for 80/10/10 and 85/5/10, with the same fixed seed and
no model scores. The latter yielded 1,341/91/168 conversations and only 17 validation
components, with at least one validation theme absent. We chose 80/10/10 to preserve
validation coverage while increasing training data. This choice does not establish
statistical power; per-theme test results remain descriptive at this sample size.

The deterministic allocator processes components in descending retained conversation
count, ties broken by SHA-256(seed:component). For each candidate split it minimizes
the increase in squared deviation from global and per-theme target counts, each term
divided by its positive target. Global count and the sum of theme terms have equal
coefficient 1. Ties use SHA-256(seed:component:split). No source component is divided;
exact target sizes and coverage are not guaranteed for arbitrary future corpora.
The configured method and ratios are recorded in the frozen manifest.

The active training configuration is `config/config_gemma.yml`; the three historical
configurations are under `config/pilotv1_config/`. Run `./scripts/train_lora.sh` for
v2 (one epoch = 633 nominal steps at batch size 2). All v2 configuration files belong
directly under `config/`, without a v2 subdirectory.

## Additional v2 training configurations

`config/config_gemma_full.yml` uses the same train/valid split as LoRA, the bf16 base,
Adafactor, 16 trainable blocks, lr 2e-5 and one epoch. Outputs are separate from LoRA.
Its memory requirements at sequence length 2,048 have not been validated.

```sh
CONFIG=config/config_gemma_full.yml ./scripts/train_lora.sh
```

`config/config_nanochat.yml` is the from-scratch SFT control (10 epochs, best checkpoint
by validation loss). Its tokenizer sees train messages only, using `corpus: null`.
No base stage is configured because the complete raw chat includes held-out sources.
The historical raw-chat pretraining experiment remains in the archived v1 config.

```sh
.venv/bin/python scripts/train_tokenizer.py --config config/config_nanochat.yml
.venv/bin/python -m galsenai.train --config config/config_nanochat.yml --stage sft
```

These explicit commands select v2. The legacy `run_full.sh` chain and nanochat CLI
defaults still select the archived v1 configurations. Do not use their defaults for
v2. Token lengths, truncation and memory need measurement before the actual runs.

## Dataset versions and publication

| Dataset | Contents and role | Availability |
|---|---|---|
| Public SFT JSON | 100 conversations, 89 complementary and 11 chat-grounded; six metadata/conversation fields | `data/private/supervised_fine_tuning_galsenai.json` is intentionally public |
| Historical legacy subset | 76 examples used during early pilot preparation; distinct from the public 100-example file | Local only |
| Pilot v1 | 356 input candidates, 204 train / 48 valid / 28 test after filtering and quarantine | Frozen splits local; public `collection.json` describes historical inputs |
| Pilot v2 | 1,600 candidates in 16 batches; 1,266 train / 167 valid / 167 test | Source batches and frozen splits local; collection and aggregate summary public |
| Assistant v3 | Filtered v2: 799 train including six explicit seeds, 114 valid; context-dependent targets excluded | Seeds public; full snapshot local |
| Assistant v4 | Same rows and split membership as v3, with eight training answers replaced from review corrections | Local; QAT reference is step 400, Google server adapter is step 799 |

The six public JSON fields are `id`, `type`, `group_id`, `source_lines`,
`external_sources` and `messages`. Each message has `role` and `content`. Source
references are metadata, not the private source text. Group IDs and source references
support grouping but do not establish factual correctness or complete provenance.
The public 100-example JSON, the 76-example legacy subset and the 100-example extended
batch are not interchangeable. Do not concatenate them without checking overlap.

Collection entries contain portable `../filename.json` source labels. The preparation
CLI resolves these filenames inside `--source-dir PATH` (or `GALSENAI_SOURCE_DIR`),
regardless of the current working directory when absolute CLI paths are used. No
machine-specific absolute root is stored in the public configuration. Other historical
repository-relative source paths remain supported by the preparator.
The public summary retains original v2 input hashes under these portable labels;
local later corrections can legitimately differ. The original frozen snapshots retain
their original paths and hashes. Do not regenerate them to update documentation.

The full generated datasets are not distributed in Git. Contact the maintainer through
this repository for access. A Hugging Face dataset publication is planned soon; no
release URL is available yet. Current review-informed validation is a development set,
not a new independent test. See the root README for the selected model and aggregate results.
