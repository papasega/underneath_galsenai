# Evaluation: Gemma 4 E2B LoRA vs galsenai (nanochat-style)

Same 10 assistant turns of the held-out validation conversations (`data/private/mlx_sft/valid.jsonl`, never trained on), gold history as context, greedy decoding. Regenerate with `scripts/run_eval.sh`.

| Model | Params | BPB ↓ | PPL/token ↓ | ROUGE-L ↑ | Token F1 ↑ | Distinct-2 ↑ | Len ratio | Empty | Grounded ↑ | Cites [n] ↑ | What it is |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-base | 4629M | 1.130 | 30.6 | 0.082 | 0.144 | 0.86 | 8.37 | 0% | – | – | Gemma 4 E2B (4-bit QAT), no fine-tuning |
| gemma-lora | 4629M | 1.378 | 65.0 | 0.150 | 0.244 | 0.98 | 0.80 | 0% | – | – | + LoRA r8, 16 layers, 5 epochs (config_gemma.yml) |
| gemma-lora-1ep | 4629M | 0.854 | 13.3 | 0.196 | 0.272 | 0.69 | 0.99 | 0% | – | – | + LoRA, checkpoint after epoch 1 (lowest val loss) |
| galsenai-sft | 12M | 1.825 | 65.5 | 0.149 | 0.191 | 0.76 | 0.47 | 0% | – | – | nanochat-style 12M model, base on chat text + SFT |
| galsenai-sft-only | 12M | 2.279 | 185.1 | 0.124 | 0.152 | 0.82 | 0.38 | 0% | – | – | same model, SFT only (no raw-chat pre-training) |
| gemma-base-rag-bm25 | 4629M | 0.983 | 19.6 | 0.179 | 0.328 | 0.96 | 1.93 | 0% | 0.47 | 100% | Gemma 4 E2B (4-bit QAT) + RAG: BM25 top-4 passages |
| gemma-base-rag-dense | 4629M | 0.938 | 17.1 | 0.181 | 0.324 | 0.96 | 1.89 | 0% | 0.44 | 100% | Gemma 4 E2B (4-bit QAT) + RAG: embeddinggemma-2 top-4 passages |
| gemma-base-rag-hybrid | 4629M | 0.964 | 18.5 | 0.184 | 0.333 | 0.96 | 1.97 | 0% | 0.44 | 90% | Gemma 4 E2B (4-bit QAT) + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-base-rag-oracle | 4629M | 0.946 | 17.5 | 0.209 | 0.336 | 0.98 | 1.53 | 0% | 0.43 | 90% | Gemma 4 E2B (4-bit QAT) + RAG: the single best passage (upper bound, uses the reference) |
| gemma-lora-rag-bm25 | 4629M | 1.184 | 36.1 | 0.177 | 0.287 | 0.98 | 1.27 | 0% | 0.50 | 90% | + LoRA r8 + RAG: BM25 top-4 passages |
| gemma-lora-rag-dense | 4629M | 1.123 | 30.0 | 0.195 | 0.322 | 0.98 | 1.14 | 0% | 0.51 | 100% | + LoRA r8 + RAG: embeddinggemma-2 top-4 passages |
| gemma-lora-rag-hybrid | 4629M | 1.142 | 31.8 | 0.170 | 0.302 | 0.98 | 1.18 | 0% | 0.46 | 100% | + LoRA r8 + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-lora-rag-oracle | 4629M | 1.157 | 33.2 | 0.173 | 0.277 | 0.99 | 0.97 | 0% | 0.45 | 90% | + LoRA r8 + RAG: the single best passage (upper bound, uses the reference) |
| gemma-lora-1ep-rag-bm25 | 4629M | 0.734 | 9.2 | 0.171 | 0.290 | 0.84 | 1.95 | 0% | 0.59 | 90% | + LoRA + RAG: BM25 top-4 passages |
| gemma-lora-1ep-rag-dense | 4629M | 0.705 | 8.5 | 0.201 | 0.319 | 0.87 | 1.89 | 0% | 0.58 | 80% | + LoRA + RAG: embeddinggemma-2 top-4 passages |
| gemma-lora-1ep-rag-hybrid | 4629M | 0.722 | 8.9 | 0.180 | 0.296 | 0.82 | 1.96 | 0% | 0.57 | 80% | + LoRA + RAG: BM25 + embeddinggemma-2 (RRF) top-4 |
| gemma-lora-1ep-rag-oracle | 4629M | 0.701 | 8.4 | 0.254 | 0.375 | 0.91 | 1.34 | 0% | 0.53 | 80% | + LoRA + RAG: the single best passage (upper bound, uses the reference) |

## RAG: retrieval quality (same turns)

Query = last user message. The *oracle* passage is the one whose content words best cover the reference answer; hit@k = that passage is in the top k; context recall = share of the reference's content words found in the 4 passages given to the model.

| Retriever | hit@1 | hit@3 | hit@5 | MRR | Context recall |
|---|---|---|---|---|---|
| bm25 | 10% | 60% | 90% | 0.38 | 0.37 |
| dense | 20% | 60% | 70% | 0.44 | 0.45 |
| hybrid | 20% | 60% | 90% | 0.45 | 0.39 |
| oracle (1 passage) | – | – | – | – | 0.36 |

## RAG: chunk size (messages per passage, stride)

| Chunking | Passages | Retriever | hit@5 | MRR | Context recall |
|---|---|---|---|---|---|
| window=2,stride=1 | 3134 | bm25 | 60% | 0.37 | 0.30 |
| window=2,stride=1 | 3134 | dense | 70% | 0.30 | 0.36 |
| window=2,stride=1 | 3134 | hybrid | 70% | 0.37 | 0.29 |
| window=4,stride=2 | 1566 | bm25 | 90% | 0.38 | 0.37 |
| window=4,stride=2 | 1566 | dense | 70% | 0.44 | 0.45 |
| window=4,stride=2 | 1566 | hybrid | 90% | 0.45 | 0.39 |
| window=8,stride=4 | 782 | bm25 | 80% | 0.64 | 0.44 |
| window=8,stride=4 | 782 | dense | 90% | 0.61 | 0.53 |
| window=8,stride=4 | 782 | hybrid | 90% | 0.72 | 0.49 |

## How to read it

- **RAG rows** (`*-rag-*`): same turns, but the last user message is preceded by retrieved chat passages and an instruction to answer from them. BPB / PPL are then conditioned on the passages. **Grounded**: share of the answer's content words (4+ letters) present in the passages. **Cites [n]**: answers containing a citation.
- **RAG leakage caveat**: the reference answers were written from the same chat that the retriever indexes, so RAG is tested on questions whose source is in the corpus (the intended use), not on unseen knowledge. `oracle` shows the ceiling when the right chunk is given.
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
- *gemma-base-rag-bm25*: Le groupe présente un avis partagé concernant la nécessité de ces éléments pour les agents de code. Certains suggèrent de les supprimer car les nouveaux modèles n'ont plus besoin de composants comme les skills, les hooks et les MCP, se contentant d'une simple boucle agentique et d'une recherche web 
- *gemma-base-rag-dense*: Le groupe suggère de supprimer ces éléments car les nouveaux modèles n'ont pas besoin de structures complexes, se contentant d'une simple boucle agentique et d'une recherche web pour rester cohérents [1][2]. De plus, l'utilisation de systèmes multi-agents est jugée excessive et peut engendrer des pr
- *gemma-base-rag-hybrid*: Le groupe présente un avis tranché concernant la suppression de ces éléments, suggérant que les nouveaux modèles n'ont pas besoin de fichiers comme CLAUDE.md, mcp, hooks et skills, car une simple boucle agentique avec une recherche web suffit [1][2]. Cependant, il existe une nuance technique, car ce
- *gemma-base-rag-oracle*: Le groupe suggère de supprimer ces éléments car les nouveaux modèles n'en ont plus besoin, car une simple boucle agentique et l'accès au web suffisent à assurer la cohérence [1]. Cette approche est considérée comme plus efficace que l'utilisation de systèmes multi-agents, qui sont jugés inutiles et 
- *gemma-lora-rag-bm25*: [1][2] selon l'avis du créateur de Claude Code, il faut supprimer CLAUDE.md, mcp, hooks et skills, car ces éléments sont désormais superflues et peuvent être remplacés par une simple boucle agentic, surtout avec un outil de recherche web. Il est aussi conseillé de ne jamais utiliser des multi-agents
- *gemma-lora-rag-dense*: [1][2] selon l'avis du créateur de Claude_code, il faut supprimer CLAUDE.md, mcp, hooks et skills ; ces éléments sont désormais obsolètes et une simple boucle agentic suffit, surtout avec les outils de recherche web. Il est aussi conseillé de ne jamais utiliser des multi-agents pour éviter des probl
- *gemma-lora-rag-hybrid*: [1][2] selon l'avis du créateur de Claude Code, il faut supprimer CLAUDE.md, mcp, hooks et skills ; ces éléments sont désormais obsolètes et l'agent ne nécessite plus d'outils comme les web search pour maintenir sa cohérence. Il est aussi conseillé de ne jamais utiliser des multi-agents, car cela re
- *gemma-lora-rag-oracle*: [1] l'avis est de ne pas garder ces éléments, car ils ont été jugés inutiles par le master des masters ; en effet, les nouveaux modèles ne nécessitent plus ces outils, et une simple boucle agentique suffit largement aux tâches ; il conseille aussi de ne jamais utiliser des multi-agents, car cela res
- *gemma-lora-1ep-rag-bm25*: Un membre conseille de supprimer CLAUDE.md, MCP, hooks et skills, car les nouveaux modèles n'en ont pas besoin de ces éléments, et qu'une simple boucle agentique suffit, surtout avec un accès à la recherche web [1][2]. Un autre point est de ne jamais utiliser de multi-agents, car c'est overkill et c
- *gemma-lora-1ep-rag-dense*: Un membre suggère de supprimer CLAUDE.md, mcp, hooks et skills, car les nouveaux modèles n'en ont pas besoin et qu'une simple boucle agentique suffit, surtout s'ils ont accès à la recherche web [1][2]. Un autre avis est qu'il faut éviter les multi-agents, car ils sont utiles à l'usage, mais créent d
- *gemma-lora-1ep-rag-hybrid*: Un membre suggère de supprimer CLAUDE.md, mcp, hooks et skills, car les nouveaux modèles n'en ont pas besoin et qu'une simple boucle agentique suffit, surtout s'ils ont accès à la recherche web [1][2]. Un autre avis est qu'il faut supprimer les skills et les skills, car ils sont utiles pour le conte
- *gemma-lora-1ep-rag-oracle*: Un membre suggère de supprimer CLAUDE.md, mcp, hooks et skills, car les nouveaux modèles n'en ont pas besoin et qu'une simple boucle agentique suffit, surtout si l'agent a accès à la recherche web [1]. Un autre avis est qu'il faut éviter les multi-agents, car ils sont utiles à l'usage, créant des pr

**Reference 2:** Haskell est un langage purement fonctionnel, sans effets de bord, ce qui le rend élégant. Dans la discussion, il sert surtout à construire des compilateurs et des outils comme Pandoc. Pour le machine learning, Python avec des bibliothèques comme PyTorch reste le choix dominant.

- *gemma-base*: C'est une excellente question. La réponse courte est : **Oui, Haskell a un intérêt, mais pas dans le rôle que l'on lui attribue souvent (le développement direct de modèles de Deep Learning).**

L'intérêt de Haskell en IA réside moins dans la création de modèles neuronaux bruts (ce qui est le domaine
- *gemma-lora*: Haskell est jugé excellent pour l'IA, et il est difficile de voir son usage actuel : de gros acteurs comme Google, DeepMind, et plusieurs équipes de recherche sont activement dans la programmation ML en Haskell. La discussion ne donne pas encore de retour clair sur la place finale du langage.
- *gemma-lora-1ep*: Oui, Haskell est un langage puissant et syntaxiquement proche de C++, qui est souvent utilisé pour les applications d'accueil des utilisateurs. Il est aussi utilisé pour les applications d'accueil des utilisateurs.
- *galsenai-sft*: Un modèle est de la discussion, est un modèle de la discussion, et des modèles de la discussion, et des modèles de la discussion, et des modèles de la discussion, et des modèles de la discussion.
- *galsenai-sft-only*: Un membreent de la discussion, et de la discussion : de la discussion : les modèles de l'usage.
- *gemma-base-rag-bm25*: Le groupe n'a pas de discussion approfondie sur l'intérêt spécifique de Haskell en intelligence artificielle. Cependant, les participants ont exprimé une confusion concernant l'application de ce langage, notant qu'ils l'ont appris mais ne comprennent pas encore son usage en IA [1][2]. Bien que l'on 
- *gemma-base-rag-dense*: Le groupe reconnaît l'intérêt de Haskell en IA, bien que certains membres expriment une confusion quant à son usage spécifique dans ce domaine [1][2]. Ils ont noté que le langage possède des caractéristiques intéressantes, comme sa nature purement fonctionnelle qui n'implique pas d'effets secondaire
- *gemma-base-rag-hybrid*: Le groupe reconnaît l'élégance et la nature purement fonctionnelle de Haskell, soulignant qu'il n'y a pas d'effets secondaires dans ce langage [1][4]. Cependant, malgré l'apprentissage du langage, plusieurs participants expriment une confusion quant à son application spécifique dans le domaine de l'
- *gemma-base-rag-oracle*: Le groupe de discussion n'a pas abordé la question de savoir si Haskell a un intérêt en intelligence artificielle. Bien qu'un participant ait posé cette interrogation concernant l'usage du langage [1], aucun des extraits fournis ne propose de réponse ou de discussion sur ce sujet spécifique.
- *gemma-lora-rag-bm25*: [1][2] Haskell est jugé élégant, et son auteur ne comprend pas encore son intérêt dans l'IA. [3] Un membre cite Haskell comme son langage préféré, mais admet ne pas l'utiliser actuellement.
- *gemma-lora-rag-dense*: [1][2] Haskell est jugé difficile, et son intérêt pour l'IA reste une question pour l'utilisateur. [3] Un membre estime que l'essentiel reste de comprendre les bases de la programmation et l'architecture logicielle, qui sont centrales à l'IA. [4] Un membre suggère que plusieurs langages peuvent part
- *gemma-lora-rag-hybrid*: [1][2] Haskell est jugé élégant, et l'auteur ne comprend encore pleinement son intérêt dans l'IA. [2] Un membre suggère que Python est déjà bien établi, et que Haskell ne domine pas encore le domaine.
- *gemma-lora-rag-oracle*: [1] Haskell est jugé excellent, mais sa pratique est difficile, avec une préférence pour Python, Java et JavaScript. [1] Un membre ne voit pas encore son usage dans l'IA, malgré avoir appris le langage.
- *gemma-lora-1ep-rag-bm25*: Haskell est jugé élégant et élégant, et il est jugé plus élégant que Python, Java et JavaScript [4][3]. Un membre indique qu'il ne comprend pas encore l'intérêt de ce langage pour l'ère de l'IA, même après l'avoir appris [1][2]. Un autre membre suggère que l'usage de l'outil est peu utile pour l'ère
- *gemma-lora-1ep-rag-dense*: Haskell est jugé plus élégant que Python, qui est jugé plus simple à apprendre, et il est jugé plus simple à comprendre que Julia [1][2]. Un membre estime que l'architecture logicielle et les bases de la programmation sont essentielles pour comprendre les bases de la programmation et surtout l'archi
- *gemma-lora-1ep-rag-hybrid*: Haskell est jugé élégant, et l'extrait [3] indique qu'il reste le langage préféré de l'auteur, même si l'on en voit peu d'usage. L'extrait [1] mentionne que l'on ne comprend pas encore son intérêt en IA, et l'extrait [2] souligne que l'on ne le voit pas encore en IA, mais que l'on en voit aussi beau
- *gemma-lora-1ep-rag-oracle*: Haskell est jugé un langage élégant, comme le montre son code, et il est jugé plus apprécié que les autres comme Python, Java et JavaScript [1]. Un membre estime que le langage est purement fonctionnel, sans side effects, et qu'il est jugé plus apprécié que les autres comme Python, Java et JavaScrip

**Reference 3:** Gemini est déjà très bon sur les langues peu dotées, mais son wolof ne sera pas aussi propre qu'avec un modèle dédié. Ça reste bien mieux que rien, et l'infrastructure est gérée par le fournisseur, donc pas besoin de déployer soi-même des GPU.

- *gemma-base*: C'est une excellente question qui touche au cœur du défi du traitement du langage naturel (NLP) pour les langues à faible ressource (low-resource languages).

La réponse n'est pas un simple oui ou non, mais dépend entièrement de **la tâche spécifique** que vous souhaitez accomplir et de la **qualité
- *gemma-lora*: Deux positions existent. D'un côté, un modèle généraliste comme Gemini est excellent en général, et on peut lui demander de l'aide pour des textes wolof. De l'autre, un modèle dédié est plus simple et plus robuste.
- *gemma-lora-1ep*: Un modèle généraliste comme Gemini peut être utilisé pour l'apprentissage des bases de la langue, comme les conjugaisons, les structures de phrases et les bases de la grammaire. Un modèle dédié, comme l'Odoo.ai ou l'Odoo.ai, est moins généraliste, mais peut offrir une meilleure compréhension des nua
- *galsenai-sft*: Un membre est un modèle de la discussion, mais il est un modèle comme une base de la discussion.
- *galsenai-sft-only*: Un membre explique.
- *gemma-base-rag-bm25*: Le groupe explore l'utilisation de modèles généralistes comme Gemini pour les langues à faible ressource, considérant qu'ils peuvent être une bonne solution [4]. Il existe cependant un débat sur la nécessité d'utiliser des modèles dédiés, car certains utilisateurs se demandent pourquoi ne pas privil
- *gemma-base-rag-dense*: Le groupe a exploré la possibilité d'utiliser un modèle généraliste comme Gemini pour le wolof, tout en reconnaissant l'existence de modèles spécialisés. Certains membres ont interrogé si Gemini pouvait parler couramment le wolof, à l'instar de modèles dédiés comme Awa ou Soynade, y compris en audio
- *gemma-base-rag-hybrid*: Le groupe a exploré la capacité de Gemini à gérer les langues à faible ressource, se demandant s'il pouvait parler couramment le wolof comme des modèles spécialisés tels que Awa ou Soynade [1][2]. Il a été noté que Gemini est considéré comme très performant pour ce type de langage, même en audio, et
- *gemma-base-rag-oracle*: Le groupe a soulevé la question de savoir si un modèle généraliste comme Gemini peut être utilisé pour le wolof ou s'il est préférable d'utiliser un modèle spécialisé [1]. Il a été suggéré que Gemini est déjà très performant pour les langues à faibles ressources [1]. La discussion a porté sur la pos
- *gemma-lora-rag-bm25*: [1][2] Gemini est déjà très capable en langue faible ressources, comme le wolof, et il est annoncé comme un modèle généraliste qui peut gérer la traduction et la génération de voix. [3] Un modèle XLS-R fine-tuné sur une base wolof, ou des modèles existants comme Soynades, sont également cités comme 
- *gemma-lora-rag-dense*: [1][2] indique que Gemini est déjà très capable en langue faible ressources, comme le wolof, et même en audio. [3] suggère à l'utilisateur de rester sur Soynades pour le moment, mais souligne que les modèles open source comme ceux cités ont déjà la capacité de gérer le français et l'영어 avec un ton s
- *gemma-lora-rag-hybrid*: [1][2] indique que Gemini est déjà très capable en langue faible ressources, comme le wolof, et que son déploiement sur plusieurs modèles est possible via Google Cloud, sans nécessiter de GPU. [3] et [4] mettent l'accent sur l'intérêt de choisir un modèle dédié, comme XLS-R fine-tuné sur des données
- *gemma-lora-rag-oracle*: [1] indique que Gemini est déjà très capable en langue low-resource, et suggère qu'il peut parler couramment wolof, même en audio, avec une qualité correcte [1]. Un membre estime que ce modèle est déjà bien meilleur que rien, et que l'infrastructure locale n'impose aucune contrainte de GPU, ce qui f
- *gemma-lora-1ep-rag-bm25*: Un membre propose d'utiliser un modèle généraliste comme Gemini, qui est jugé très performant pour les langues à faible ressources, comme l'Awa ou Soynade, et propose aussi de combiner Google Translate ou DeepL avec Gemini 3.1 Preview TTS pour obtenir une traduction et une audio de qualité, [1][4]. 
- *gemma-lora-1ep-rag-dense*: Un membre estime que l'outil est déjà très performant pour les langues à faible ressources, comme l'est Gemini, et propose de l'utiliser pour l'apprentissage de l'écriture, comme l'on l'utilise avec Soynade. Un autre membre souligne que l'infrastructure est chez Google, ce qui évite de déployer des 
- *gemma-lora-1ep-rag-hybrid*: Un membre suggère d'utiliser Soynade, qui est open source et peut parler français et anglais avec un ton sénégalais, et propose de le déployer sur ses propres infrastructures [4]. Un autre membre estime que Gemini est déjà très performant pour les langues à faible ressources, comme l'Awa ou Soynade,
- *gemma-lora-1ep-rag-oracle*: Un membre suggère qu'un modèle généraliste comme Gemini est déjà très performant pour les langues à faible ressources, comme l' langue locale, et propose de l'utiliser pour l'apprentissage des bases de la langue, même si l'audio n'est pas parfait [1]. Un autre membre souligne que l'infrastructure d'

