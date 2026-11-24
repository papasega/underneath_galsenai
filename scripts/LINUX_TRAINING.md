# Linux / Unix training (without MLX)

The training and evaluation entry points print an English dataset notice. The public
`supervised_fine_tuning_galsenai.json` is an example subset, not the complete experiment
data. Complete datasets and trained adapters are available on request by contacting
the GalsenAI community through this repository (for example, a GitHub issue). A Hugging Face dataset release is planned. The scripts
do not download private data or silently substitute the public examples.

For Gemma, use `scripts/train_lora_hf.py`. It uses
[Transformers](https://huggingface.co/docs/transformers/model_doc/gemma4) and
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora), with no MLX dependency.
Run these commands from the repository root with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install the PyTorch build matching your CUDA driver using the selector linked below.
python -m pip install -e '.[hf-train]'
nvidia-smi
python -c 'import torch; print("CUDA:", torch.cuda.is_available())'
hf auth login  # if model access requires authentication; accept its license on Hugging Face first

# Short end-to-end check. Requires the complete local train.jsonl and valid.jsonl snapshot.
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_hf.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --data data/training_data/pilot_v4_assistant \
  --max-steps 1 \
  --output data/private/adapters/hf/gemma4_v4_smoke1

# One epoch. Always choose a new output directory for each run.
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_hf.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --data data/training_data/pilot_v4_assistant \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --learning-rate 5e-5 \
  --rank 8 \
  --max-length 2048 \
  --output data/private/adapters/hf/gemma4_v4_run1
```

These CUDA commands require an NVIDIA GPU; they do not run on a Mac GPU.
Choose the CUDA or CPU wheel with the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Without an NVIDIA GPU, pass `--device cpu`; this uses float32 and is very slow for Gemma.
`--device auto` selects CUDA when available, otherwise CPU. CUDA uses bfloat16 when
supported, otherwise float16. This is single-process training; expose one GPU as above.
The default model is `google/gemma-4-E2B-it`, using ordinary floating-point weights,
not the MLX 4-bit checkpoint. It needs more memory; no 4-bit QLoRA or CPU offloading is
implemented. Reduce `--batch-size` (default 1), or choose a smaller compatible
`--model` and its `--target-modules` if memory is insufficient.

The script trains on assistant tokens only, preserves each JSONL conversation's
system/history, disables thinking, and rejects overlength examples instead of silently
truncating their answers. Defaults: one epoch, rank 8, alpha 16, q/v projections on
all layers, learning rate 5e-5, batch size 1 with accumulation 2, maximum 2,048 tokens.
Every epoch evaluates validation loss and saves a checkpoint. Final adapters,
tokenizer, Trainer logs/checkpoints and `training_provenance.json` (input hashes,
arguments and library versions) go to the new `--output` directory. Keep it under
`data/private/adapters/` to exclude it from Git.

This is a separate training recipe, not an exact reproduction of the MLX experiments
(different weight precision, LoRA scaling and layer selection). PEFT adapters cannot
be passed directly to the MLX webchat. Existing `train_lora.sh`, `run_eval.sh`,
`run_eval_rag.sh`, `run_full.sh`, and `eval_gemma.py` remain MLX-specific. The HF trainer
reports validation loss, but does not run their generated-answer benchmark.

The **nanochat-style `galsenai` model already supports CUDA, MPS and CPU** through
PyTorch, in that priority order, for both training and `eval_galsenai.py`. No separate
Linux implementation is needed. With the historical private corpus and JSONL splits
available at the paths in `config/pilotv1_config/config_nanochat.yml`:

```bash
python scripts/train_tokenizer.py --config config/pilotv1_config/config_nanochat.yml
python -m galsenai.train --config config/pilotv1_config/config_nanochat.yml --stage base
python -m galsenai.train --config config/pilotv1_config/config_nanochat.yml --stage sft
python scripts/eval_galsenai.py --name galsenai-sft-linux \
  --checkpoint data/private/checkpoints/galsenai_sft.pt \
  --out results/local/galsenai-sft-linux.json
```

These are the historical nanochat paths, not the current Gemma v4 snapshot. Raw-chat pretraining exposes source material
behind validation answers; do not describe this as a clean held-out benchmark. The
alternative `config/config_nanochat.yml` runs v2 SFT from scratch without a raw corpus:
use it for tokenizer training and `--stage sft`, and use its tokenizer/checkpoint/data
paths explicitly when evaluating. CPU tokenization is platform independent. RAG
retrieval and report scripts also run without MLX, but answer evaluation still needs
a compatible inference server; the provided Gemma server uses MLX.

See [training and evaluation](TRAINING.md) and [dataset access](../dataset/README.md#data-access-and-schema).
