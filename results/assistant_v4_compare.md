# Assistant v4 experiment results

Selected for local inference: **Gemma 4 E2B QAT 4-bit + v4 LoRA, step 400**.
This selection closes the current local experiment cycle; it is not an independent quality certification.

## Matched evaluation

114 development turns, identical input token hashes and references across all three candidates.
Shared assistant system prompt, thinking disabled, temperature 0, repetition penalty 1.1,
512 output tokens, seed 0, no RAG. V4 validation is byte-identical to v3 validation.

| Candidate | BPB ↓ | Perplexity ↓ | ROUGE-L ↑ | Token F1 ↑ | Distinct-2 | Narrative flags | Truncated |
|---|---|---|---|---|---|---|---|
| gemma-v3 | 0.6950 | 9.5523 | 0.1662 | 0.2731 | 0.9891 | 0.0000 | 0.0000 |
| gemma-v4-step200 | 0.7190 | 10.3283 | 0.1641 | 0.2696 | 0.9912 | 0.0088 | 0.0000 |
| gemma-v4-step400 | 0.6939 | 9.5195 | 0.1765 | 0.2777 | 0.9844 | 0.0088 | 0.0000 |

V4 step 400 has slightly better aggregate reference likelihood and lexical overlap than v3.
The gains are modest. Both v4 checkpoints have one narrative flag out of 114; v3 has none.
No model has empty or truncated answers in this comparison.
The lowest sampled training-validation loss (step 200) did not select the best full-generation result.

## Training and provenance

- 799 training conversations, 114 validation conversations; 400 steps, batch 2, one scheduled epoch.
- LoRA rank 8, 16 layers, scale 20, learning rate 5e-5, seed 0, masked prompts, sequence limit 2048.
- Eight existing training answers corrected. No new review questions or answers added to train.
- Snapshot builder: `python -m data.prepare_review_data`; configuration: `config/config_gemma.yml`.
- Aggregate machine-readable metrics and hashes: [assistant_v4_summary.json](assistant_v4_summary.json).
- Full generations, correction delivery, snapshots and adapters remain local and ignored by Git.
- Local source results: `results/local/v3-v4-Ngmbrd/`.
- Historical v1 results remain in [eval_compare.md](eval_compare.md); do not pool their scores with this run.

## Limits

This validation set guided the corrections, so results are development evidence. One seed, no significance
test, no independent final test, no complete factual review of v4 outputs, and no RAG quality evaluation.
The RAG corpus is `data/private/chat_anonymized_v2_rag.txt`; selecting it does not establish RAG quality.
