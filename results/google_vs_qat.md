# Google weights vs QAT 4-bit: assistant v4

114 development questions, identical prompt token hashes and references. Evaluation:
thinking disabled, temperature 0, repetition penalty 1.1, 512 output tokens, seed 0,
no RAG. Both adapted models use MLX, LoRA rank 8, 16 layers, scale 20 and learning rate 5e-5.

| Model | BPB ↓ | Perplexity ↓ | ROUGE-L ↑ | Token F1 ↑ | Narrative flags | Truncated |
|---|---|---|---|---|---|---|
| QAT 4-bit + v4, step 400 | 0.6939 | 9.5195 | 0.1765 | 0.2777 | 1/114 | 0/114 |
| Google original, no LoRA | 2.0165 | 698.0617 | 0.0741 | 0.1187 | 0/114 | 105/114 |
| Google + v4, step 550 | 0.6980 | 9.6456 | 0.1675 | 0.2714 | 0/114 | 0/114 |
| Google + v4, step 799 | 0.6811 | 9.1308 | 0.1709 | 0.2697 | 0/114 | 0/114 |

Google step 799 improves reference perplexity by 4.1% over QAT, without a clear
generated-answer advantage. QAT retains higher lexical overlap and is the comparison
reference. Google step 799 is now the user-selected webchat default for local testing.
The original Google model produces long answers under this
protocol; its high truncation rate limits interpretation of its generation scores.

Limits: one seed, development data already used for model selection, no complete human
correctness review. QAT used batch 2 / 400 updates; Google used batch 1 / 799 updates
for the same 799-example snapshot. This does not isolate quantization's effect.

Private generations and review: `results/local/google-vs-qat-qDz5zc/`.
Earlier comparison: [assistant v4](assistant_v4_compare.md).
