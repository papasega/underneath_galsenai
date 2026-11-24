# Reproducing the experiments

The repository publishes code, configurations, tests, a 100-conversation example
JSON and [experiment summaries](../results/google_vs_qat.md). Complete training
snapshots, review corrections and trained adapters are not included in Git.
**They are available on request by contacting the GalsenAI community through this
repository, for example by opening a GitHub issue.** Hugging Face releases are
planned; no download URL is available yet.

## From a fresh clone

1. Install the locked environment using the [main README](../README.md#install),
   or follow the [CUDA/CPU guide](LINUX_TRAINING.md).
2. Use the [public example converter](TRAINING.md#public-example-conversion) to
   inspect the supported format. Full experiment reproduction requires the
   requested snapshots and model/adapter pairs.
3. Once those inputs are supplied, follow [training and evaluation](TRAINING.md).
   Retain their manifests and hashes; do not regenerate frozen splits from newer
   corrected source files.

## Versions and limits

[model_revisions.json](../config/model_revisions.json) pins the Google and QAT Hub
commits. [mlx_entry.py](mlx_entry.py) resolves those snapshots for training, serving
and fusion. MLX evaluation and HF training use the same pins. For another Hub model,
supply an immutable 40-character `--revision`; the server accepts `MODEL_REVISION`.
Google model access may require accepting its license and authenticating with
`hf auth login`.

New training runs write model revision, input hashes and dependency versions into
`training_provenance.json`. Evaluations record their revision and decoding settings.
Keep these generated records with the Git commit and hardware description.
The Linux uv lock selects CPU PyTorch; CUDA users must record their chosen PyTorch
wheel and driver versions separately.

The pinned revisions were captured at project closure. Historical runs did not
record every dependency or model revision, so exact historical reconstruction
cannot be guaranteed. Results use development data and one seed. Identical inputs
and seeds do not guarantee bit-for-bit agreement across MLX, CUDA and CPU hardware.
