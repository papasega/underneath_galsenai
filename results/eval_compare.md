# Evaluation: Gemma 4 E2B LoRA vs galsenai (nanochat-style)

Same 10 assistant turns of the held-out validation conversations (`data/private/mlx_sft/valid.jsonl`, never trained on), gold history as context, greedy decoding. Regenerate with `scripts/run_eval.sh`.

| Model | Params | BPB ↓ | PPL/token ↓ | ROUGE-L ↑ | Token F1 ↑ | Distinct-2 ↑ | Len ratio | Empty | What it is |
|---|---|---|---|---|---|---|---|---|---|
| gemma-base | 4629M | 1.130 | 30.6 | 0.082 | 0.144 | 0.86 | 8.37 | 0% | Gemma 4 E2B (4-bit QAT), no fine-tuning |
| gemma-lora | 4629M | 1.378 | 65.0 | 0.150 | 0.244 | 0.98 | 0.80 | 0% | + LoRA r8, 16 layers, 5 epochs (config_gemma.yml) |
| gemma-lora-1ep | 4629M | 0.854 | 13.3 | 0.196 | 0.272 | 0.69 | 0.99 | 0% | + LoRA, checkpoint after epoch 1 (lowest val loss) |
| galsenai-sft | 12M | 1.825 | 65.5 | 0.149 | 0.191 | 0.76 | 0.47 | 0% | nanochat-style 12M model, base on chat text + SFT |
| galsenai-sft-only | 12M | 2.279 | 185.1 | 0.124 | 0.152 | 0.82 | 0.38 | 0% | same model, SFT only (no raw-chat pre-training) |

## How to read it

- **BPB** (bits per byte of the reference answer) and **PPL/token** score the reference answer under the model; BPB is comparable across tokenizers, PPL/token is not.
- **ROUGE-L / Token F1**: word overlap between the generated and the reference answer. Low absolute values are normal: the references are paraphrases of chat opinions.
- **Distinct-2**: share of unique word bigrams (low = repetition loops). **Len ratio**: generated words / reference words.

## Caveats

- **Tiny sample**: 10 turns from 8 conversations. Differences of a few hundredths are noise; only large gaps mean something.
- **Leakage**: `galsenai-sft` was pre-trained on the anonymized chat from which the validation Q&A were extracted, so its numbers are optimistic; `galsenai-sft-only` is the leak-free galsenai row. Gemma never saw the chat.
- Gemma answers with its thinking mode on (same chat template as training); the thought block is removed before scoring. An unfinished thought counts as an empty answer.
- Gemma is 4.6B parameters pre-trained on trillions of tokens; the galsenai model is 12M parameters trained on ~130k tokens. The comparison shows what each recipe reaches with this data, not that the architectures are equivalent.
- Only the validation loss was used to pick checkpoints (best epoch for galsenai, epoch 1 vs 5 for Gemma); with 8 conversations that choice is noisy.

## Sample outputs (first 3 turns)

**Reference 1:** Dans la discussion, un argument est que les modèles récents n'ont plus besoin de ces couches : une simple boucle agentique suffit, surtout avec la recherche web pour rester cohérent. Un membre nuance : il faut garder un MCP quand un cas d'usage le demande, car beaucoup de services proposent des API 

- *gemma-base*: C'est une question qui dépend entièrement de l'**état actuel de votre agent**, de ses **objectifs**, et de la **complexité de votre projet**.

Il n'y a pas de réponse unique (oui ou non). Voici une analyse détaillée de la fonction de chaque élément et des scénarios dans lesquels vous devriez envisag
- *gemma-lora*: Deux positions existent. D'un côté, on peut voir ces éléments comme des outils essentiels à l'agent, et d'un autre, ils peuvent gêner ses explorations. Il reste à choisir entre l'efficacité et la simplicité du contexte.
- *gemma-lora-1ep*: Un membre rappelle que les MCP, les hooks et les skills sont des skills qui ont été développés avec Claude. Il est à vérifier si l'agent de code a besoin de supprimer les fichiers CLAUDE.md, les MCP, les hooks et les skills.
- *galsenai-sft*: Un modèle peut faire tourner un modèle de l'usage, ce qui peut faire une base de l'objectif de l'objectif de l'objectif de l'obser.
- *galsenai-sft-only*: Un membre explique.

**Reference 2:** Haskell est un langage purement fonctionnel, sans effets de bord, ce qui le rend élégant. Dans la discussion, il sert surtout à construire des compilateurs et des outils comme Pandoc. Pour le machine learning, Python avec des bibliothèques comme PyTorch reste le choix dominant.

- *gemma-base*: C'est une excellente question. La réponse courte est : **Oui, Haskell a un intérêt, mais pas dans le rôle que l'on lui attribue souvent (le développement direct de modèles de Deep Learning).**

L'intérêt de Haskell en IA réside moins dans la création de modèles neuronaux bruts (ce qui est le domaine
- *gemma-lora*: Haskell est jugé excellent pour l'IA, et il est difficile de voir son usage actuel : de gros acteurs comme Google, DeepMind, et plusieurs équipes de recherche sont activement dans la programmation ML en Haskell. La discussion ne donne pas encore de retour clair sur la place finale du langage.
- *gemma-lora-1ep*: Oui, Haskell est un langage puissant et syntaxiquement proche de C++, qui est souvent utilisé pour les applications d'accueil des utilisateurs. Il est aussi utilisé pour les applications d'accueil des utilisateurs.
- *galsenai-sft*: Un modèle est de la discussion, est un modèle de la discussion, et des modèles de la discussion, et des modèles de la discussion, et des modèles de la discussion, et des modèles de la discussion.
- *galsenai-sft-only*: Un membreent de la discussion, et de la discussion : de la discussion : les modèles de l'usage.

**Reference 3:** Gemini est déjà très bon sur les langues peu dotées, mais son wolof ne sera pas aussi propre qu'avec un modèle dédié. Ça reste bien mieux que rien, et l'infrastructure est gérée par le fournisseur, donc pas besoin de déployer soi-même des GPU.

- *gemma-base*: C'est une excellente question qui touche au cœur du défi du traitement du langage naturel (NLP) pour les langues à faible ressource (low-resource languages).

La réponse n'est pas un simple oui ou non, mais dépend entièrement de **la tâche spécifique** que vous souhaitez accomplir et de la **qualité
- *gemma-lora*: Deux positions existent. D'un côté, un modèle généraliste comme Gemini est excellent en général, et on peut lui demander de l'aide pour des textes wolof. De l'autre, un modèle dédié est plus simple et plus robuste.
- *gemma-lora-1ep*: Un modèle généraliste comme Gemini peut être utilisé pour l'apprentissage des bases de la langue, comme les conjugaisons, les structures de phrases et les bases de la grammaire. Un modèle dédié, comme l'Odoo.ai ou l'Odoo.ai, est moins généraliste, mais peut offrir une meilleure compréhension des nua
- *galsenai-sft*: Un membre est un modèle de la discussion, mais il est un modèle comme une base de la discussion.
- *galsenai-sft-only*: Un membre explique.

