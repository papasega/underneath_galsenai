# GalsenAI SFT data and evaluation protocol

## Scope and current status

This is a reproducible **pilot protocol**, not evidence of model quality or a completed paper.
The first snapshot contains 356 input conversations: LLM fundamentals 40, Senegal 40,
agents 100, voice 100, legacy 76. Generation is ongoing. The 15 thematic collection
objectives in `collection.json` target 100 candidates each (1,500 candidates), before
quality filtering; these are not claims of achieved coverage or quotas for each split.
The legacy set is additional. Theme labels currently describe source batches, not
independently reviewed per-example labels. In particular, voice examples also cover
low-resource languages; do not double-count these as 100 independent Wolof examples.

The current snapshot contains 204 train, 48 validation and 28 test conversations.
76 legacy examples lack provenance and are quarantined, including for training:
putting them in train could leak answers related to test source threads.
The pilot test is exploratory and too small for strong per-theme conclusions.
The previous 8-conversation validation set and previously trained adapters are
historical exploratory artifacts, not clean baselines for this new test.

## Reproduce and inspect

From the repository root:

```sh
python3 data/prepare_training_data.py
python3 -m unittest discover -s tests -p test_prepare_training_data.py
```

The existing scripts under `scripts/` are unchanged. Inputs accept either a JSON
list or an object with an `examples` list. Original data is never rewritten.
`pilot_v1/` contains:

- `train.jsonl`, `valid.jsonl`, `test.jsonl`: MLX-compatible `messages` conversations.
- `*.metadata.json`: aligned IDs, source references, themes and connected components.
- `audit.json`: input hashes, quarantined records, removed exact duplicates and
  detected near-duplicate question pairs.
- `collection.json`: frozen collection configuration.
- `manifest.json`: SHA-256 fingerprints, seed, split counts, themes, types and
  assistant-turn counts. Counts are conversations unless explicitly marked otherwise.

An identical rerun is a no-op. Changed inputs, configuration, preparation code or
snapshot bytes cause refusal without overwriting the snapshot. Do not edit frozen
outputs manually. A failed partial write is also refused on rerun; inspect it before
choosing a new output directory. The manifest detects changes; it is not an external
signed archive or an operating-system write lock. Keep an independent backup.

## Split policy and limitations

Use fixed seed 20261123. Construct connected components globally across input files:
shared `group_id`, overlapping source lines, an identical normalized user question,
an identical normalized conversation, or question word-trigram Jaccard >= 0.8.
Keep every component in one split, including transitive links across themes.
Remove exact conversation duplicates only after joining their provenance components.
Unknown groups or chat-grounded examples without line references are quarantined.
Malformed schema, invalid ranges or duplicate IDs within a file stop the build.
Local IDs are namespaced by filename.

Hash the seed and sorted component member IDs into nominal 70/15/15 train/valid/test
buckets. This is deterministic grouped allocation, **not stratified sampling**.
Real ratios and theme coverage can deviate, especially with large source components.
Never retry seeds to improve a model's results. Report missing/undersized slices;
collect independent source groups rather than breaking components to meet quotas.

Integer source lines refer to `chat_anonymized.txt`; range objects retain their named
source document. Verify that all batches use the same exact source version and line
numbering. Freeze its SHA-256 before the paper experiment. The script checks reference
syntax and overlap, not whether referenced lines actually support the answers.
`source_messages` is preserved but is not used for grouping; examples grounded only
by message IDs require conversion to verified line ranges first.

Lexical matching cannot guarantee semantic decontamination. Review paraphrases,
thread boundaries and source ranges manually, including across the legacy corpus.
Generic identical short questions can conservatively join unrelated conversations.
Do not treat `chat_grounded` as a certificate of factual correctness.

## Collection and human review before a paper benchmark

Track all 15 themes in `collection.json`; add explicit source entries for new lots.
Keep one canonical primary theme per example, plus optional secondary labels in the
annotation ledger. Distinguish knowledge of the discussion from current technical
truth. Record language (French/Wolof/mixed), task, difficulty and grounded/complementary
origin separately. Do not infer Wolof competence from French explanations about Wolof.

For every generated batch retain the exact Gemini model/version, generation date,
prompt and its hash, source corpus hash, generation settings when exposed, and original
output. Unknown generation settings must be marked unknown, not reconstructed.
Record permission/license and provenance for the chat and external materials before
publication. Derived snapshots remain git-ignored by default.

Before training, review candidate examples for correctness, source support, ambiguity,
privacy, duplicates and length. Record accept/revise/reject with reasons and original
IDs. Verify complementary factual claims against dated primary sources. Synthetic
answers are draft labels, not independent ground truth. Use a competent Wolof reviewer
for Wolof and mixed-language items. Use two independent reviewers for the final test,
resolve disagreements, and report inter-rater agreement before adjudication.

The supplied corpus was generated before source splitting. State that limitation in
the paper. For new confirmatory data, reserve independent source threads first and
create test questions independently, preferably by humans. Never send held-out QA
pairs to the training-data generator. Prior model exposure cannot be excluded by this
local pipeline.

## Continue generation without moving the benchmark

Keep `pilot_v1` unchanged. New generation remains in the private incoming directory.
Do not create `v2` by rerunning a random split and then compare its scores to `v1`.
New members change connected-component hashes and can change assignments.

For the same benchmark, future training additions must first be checked against ALL
frozen validation/test components, source lines and question paraphrases. Related
examples must remain held out and must not enter training. A new bridge between train
and a held-out component is contamination requiring investigation, not reassignment.
This script deliberately refuses automatic incremental ingestion into an existing
snapshot. Incremental extension is not implemented; prepare and review that operation
before using additional batches with this benchmark.

A new benchmark version is permitted for a deliberately new study, before tuning,
using `--output data/training_data/<new_version>`. Document its differences and rerun
all baselines; results from different test versions are not paired comparisons.

## Training contract

Use train only for gradient updates; valid for learning rate, epochs and checkpoint
selection; test only after choices are locked. Never concatenate valid/test into SFT.
Measure actual tokenizer lengths and assistant-supervised tokens before training.
Reject or explicitly handle overlong examples instead of silently truncating answers.
Keep the exact tokenizer, chat template, system prompt, thinking mode, loss mask and
model revision in the experiment manifest. The preparation script does not tokenize.

Pass the new snapshot directory through a separate MLX config or `--data`; the old
training scripts still default to the old dataset. Derive iterations from the actual
training count and batch size, not the previous 34-step recipe. On this pilot, batch 2
means 102 nominal steps per epoch. Do not assume the old wrapper computes epochs from
an overridden `--data` directory: use explicit `--iters`, or a separate config whose
`data` field names the snapshot.

Retrain from the same initial model for every clean comparison. Historical adapters
may already have seen new test sources through old training data.

## Evaluation protocol to lock before training

Primary endpoint: proportion of answers rated fully correct and supported, aggregated
first within source components, then across components. Define a task-specific rubric
and expected facts for each test question before inspecting model answers. Report also
micro accuracy and theme-macro scores, denominators, missing slices and confidence
intervals. Rate factuality, completeness, unsupported claims and appropriate abstention
separately. Blind reviewers to model identity and randomize response order.

For multi-turn conversations use gold previous turns for the main controlled test;
report free-running conversation evaluation separately. Multiple assistant turns in
one thread are correlated, not independent test samples. Use paired cluster bootstrap
by source component (10,000 replicates, seed 20261123) for 95% intervals and paired
method differences. Train three seeds (0, 1, 2); report each and their mean/dispersion.
Do not interpret a non-significant difference as equivalence. A sample-size/power
analysis requires an estimated effect and number of independent components, not just
1,500 generated examples. With 28 pilot conversations, no small-gain claim is justified.

Secondary metrics: BPB/reference likelihood, ROUGE-L, token F1, repetition, empty answers,
latency and peak memory. Lexical overlap and the existing `Grounded` word-overlap score
are not factuality metrics. Report refusal errors separately from correct abstention.
Text QA about ASR, TTS, vision or tools does not demonstrate audio, image or tool-use
competence; those claims need modality-specific tasks and executable checks.

Compare: initial model, SFT grounded-only, SFT mixed grounded/complementary, each with
and without the same RAG pipeline where feasible. Keep model, prompt, decoding,
retrieval and validation-selection budget fixed for a method comparison. Data-scale
comparisons should report both rows and supervised tokens plus compute; changing data
and token budget does not isolate data quality. Pre-register the chosen controls,
hyperparameters and primary comparisons before reading test outputs.

Use greedy decoding for the primary evaluation, fixed maximum 1500 generated tokens,
and report cap hits/unfinished thinking. Lock the chat template and thinking mode.
The existing non-RAG evaluator accepts `--data <snapshot>/test.jsonl` and
`--max-new-tokens 1500`. Its default input is the OLD validation set.
Do NOT use its existing RAG contexts with this snapshot: they are indexed by position
for the old validation questions. Rebuild contexts keyed by stable example/turn IDs
before evaluating RAG; that evaluator adaptation is outside this preparation script.

Distinguish closed-book source-held-out evaluation from retrieval-available evaluation.
In RAG, document whether the index contains test source documents (valid for retrieval
QA) and ensure it never contains generated test questions or answer keys. Freeze the
index corpus, chunking, retriever, embedding revision, top-k, prompts and context hashes.
An oracle that selects passages using reference answers is diagnostic only.

## Evidence and reporting

Publish a data-flow diagram, collection/filter counts, theme/type/language tables,
source-group counts, generator details, reviewer protocol, frozen fingerprints, training
budgets and paired results. Separate pilot findings from confirmatory results. Do not
claim full decontamination or statistically established gains from this preparation.

References informing the caution around contamination and synthetic quality:
- https://aclanthology.org/2023.findings-emnlp.722/
- https://aclanthology.org/2024.emnlp-main.285/

The numerical choices above are this project's proposed protocol, not guarantees or
settings prescribed by those papers. Evaluation scoring, bootstrap and human review
are specified here but have not been implemented or performed by the data preparer.
