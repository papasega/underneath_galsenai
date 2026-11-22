"""Keyword themes shared by the statistics scripts (regex per theme, matched case-insensitively)."""

THEMES = {
 'LLM / modèles (GPT, Claude, Gemini, Llama…)':r'\b(llm|gpt|claude|gemini|llama|opus|sonnet|haiku|mistral|qwen|deepseek|gemma|openai|anthropic|modèles?|models?)\b',
 'Agents / outils / Claude Code / Codex':r'\b(agents?|agentic|claude code|codex|cursor|mcp|skills?|tools?|outils?|copilot)\b',
 'Entraînement / fine-tuning / RL':r'\b(fine.?tun\w*|lora|sft|rlhf|grpo|dpo|training|entra[iî]n\w*|pretrain\w*|distill\w*|dataset|données|data)\b',
 'NLP langues locales (wolof…)':r'\b(wolof|pulaar|sérère|serere|nlp|langues?|traduction|tokeni\w*|asr|tts)\b',
 'Python / code / dev':r'\b(python|pytorch|tensorflow|code|github|git|api|docker|rust|mojo|jax|bug|script|package|framework)\b',
 'Carrière / emploi / études':r'\b(stage|emploi|job|recrut\w*|cv|master|thèse|these|doctorat|phd|bourse|formation|cours|université|ucad|ugb|école|ecole|carrière|salaire)\b',
 'Maths / théorie / recherche':r'\b(papier|paper|arxiv|maths?|math[ée]matiques|algèbre|gradient|transformers?|attention|théorie|recherche|benchmark)\b',
 'Infra / GPU / calcul':r'\b(gpu|cuda|h100|a100|supercalculateur|calcul|cloud|colab|kaggle|vram|inférence|inference|serveur)\b',
 'Événements / communauté / Sénégal':r'\b(meetup|hackathon|conférence|conference|événement|evenement|atelier|workshop|galsen|sénégal|senegal|dakar|communauté|afrique)\b',
}

SHORT = {
    "LLM / modèles (GPT, Claude, Gemini, Llama…)": "LLM / modèles",
    "Agents / outils / Claude Code / Codex": "Agents / outils",
    "Entraînement / fine-tuning / RL": "Entraînement / RL",
    "NLP langues locales (wolof…)": "NLP langues locales",
    "Python / code / dev": "Code / Python",
    "Carrière / emploi / études": "Carrière / études",
    "Maths / théorie / recherche": "Maths / recherche",
    "Infra / GPU / calcul": "Infra / GPU",
    "Événements / communauté / Sénégal": "Événements / communauté",
}
