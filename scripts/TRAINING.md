# Assistant training and evaluation

Run commands from the repository root after installing the dependencies described
in the [main README](../README.md#install).

**Complete training snapshots and trained adapters are not included in Git.
They are available on request by contacting the GalsenAI community through this
repository, for example by opening a GitHub issue.** See
[dataset access and schema](../dataset/README.md#data-access-and-schema).

## Assistant training (current)

[config_gemma.yml](../config/config_gemma.yml) selects the QAT Gemma model, one
epoch, rank 8, 16 layers, batch size 2 and learning rate 5e-5. It expects the v4
snapshot (799 train, 114 validation), evaluates and saves every 50 steps.
The default server instead selects the Google model with its v4 step-799 adapter.
These model/adapter pairs must stay consistent.

| Entry point | Purpose |
|---|---|
| [prepare_training_data.py](../data/prepare_training_data.py) | Build grouped pilot splits from supplied source batches |
| [prepare_assistant_data.py](../data/prepare_assistant_data.py) | Filter context-dependent targets and add assistant seeds |
| [prepare_review_data.py](../data/prepare_review_data.py) | Apply supplied review corrections to the assistant snapshot |
| [train_lora.sh](train_lora.sh) | Train MLX LoRA using the selected configuration |
| [train_lora_hf.py](train_lora_hf.py) | Train Transformers/PEFT LoRA on CUDA or CPU |

With the requested v4 snapshot installed at the configuration's data path:

```bash
./scripts/train_lora.sh
```

Choose a new `--adapter-path` for another run. The wrapper derives iteration count
from epochs, dataset size and effective batch size; explicit CLI values override
configuration. [assistant_system.txt](../config/assistant_system.txt) supplies the
shared training/webchat system instruction.

V4 preserves v3 split membership and validation data, replacing eight training
answers from human review. The filtered assistant snapshot removes examples that
need missing excerpts or describe a discussion instead of answering directly.
Six published seeds address assistant behavior. These corrections do not establish
factual correctness or a clean independent test. See the
[dataset protocol](../data/training_data/PILOT_V2.md) and
[development comparison](../results/assistant_v4_compare.md).

## Evaluation

[run_eval.sh](run_eval.sh) compares the QAT base, v2 LoRA and v3 LoRA on 114
assistant validation turns. It requires the matching snapshots and both adapters;
it does not evaluate v4. Thinking is disabled, temperature is zero, repetition
penalty is 1.1 and the output budget is 512 tokens.

```bash
LIMIT=2 ./scripts/run_eval.sh
./scripts/run_eval.sh
```

Each invocation creates a new private output directory with metrics, predictions,
comparison and review files. [eval_gemma.py](eval_gemma.py) supports other model and
adapter pairs; use `--help` for its options. Close the inference server before
large evaluations to free memory. ROUGE/F1 measure lexical overlap, not correctness.

## Public example conversion

[prepare_sft_mlx.py](prepare_sft_mlx.py) converts the published 100-example JSON
into MLX JSONL (92 train / 8 validation by default). It accepts record objects or
legacy message lists and refuses an existing output directory:

```bash
uv run --no-sync python scripts/prepare_sft_mlx.py --output data/private/example_sft
```

This creates new demonstration files. It does not reconstruct the v4 snapshot.

## Other workflows

- [Linux / Unix and CUDA training](LINUX_TRAINING.md).
- [Reproducing the experiments](REPRODUCIBILITY.md).
- [train_tokenizer.py](train_tokenizer.py) and `python -m galsenai.train` train the
  nanochat-style model. [config_nanochat.yml](../config/config_nanochat.yml) selects
  v2 SFT from scratch and requires the corresponding supplied data.
- [run_full.sh](run_full.sh) supports historical full fine-tuning, fusion, merging
  and evaluation. [run_eval_rag.sh](run_eval_rag.sh) is the historical RAG sweep.
  Both require their own configured data and model artifacts.
