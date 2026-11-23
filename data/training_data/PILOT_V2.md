# Pilot v2

Branch: `pilot-v2`. This is a new exploratory benchmark version. The frozen v1 files
are retained; comparisons between v1 and v2 scores are not paired comparisons.

## Data

All 16 corrected JSON batches from `data/private/gemine_generate` are included,
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

Using the unchanged seed and connected-component split method yields:

| Split | Conversations |
|---|---:|
| Train | 1,026 |
| Validation | 256 |
| Test | 318 |

There are 284 independent components under the implemented grouping rules.
The actual proportions differ from nominal 70/15/15 because source groups stay intact.
Per-theme/type counts and input hashes are in `pilot_v2_summary.json`; the full local
snapshot manifest additionally records file hashes and supervised assistant-turn counts.

## Reproduction

Use the project environment, which includes Click:

```sh
uv sync --extra dev
.venv/bin/python data/prepare_training_data.py \
  --config data/training_data/collection_v2.json \
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

Seven v1 test conversations share source lines with v2 training data; ten have an
identical normalized question in v2 training. These counts overlap and must not be
summed. The v1 test is therefore unsuitable for reporting clean performance of a
v2-trained model. All baselines must be evaluated on the v2 test under the same
conditions; historical adapters need a separate training-exposure audit.

Within v2, source-line sets, exact normalized questions and detected source components
are disjoint between all split pairs. Semantic paraphrases and undisclosed pretraining
exposure remain outside that guarantee. Human factual review, tokenizer lengths,
model training and final test evaluation have not been performed by this preparation.
