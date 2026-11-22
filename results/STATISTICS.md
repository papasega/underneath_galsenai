# Statistics, top voices and word cloud

Computed on the **anonymized** chat (`scripts/chat_figures.py`, so safe to publish): 3383 messages, 103 members.
Voices are the random labels of the anonymized file (`U10`...), not names; they are stable only for that file.

**Themes** (share of messages mentioning each theme, keyword detection, a message can count for several):

| Theme | Share |
|---|---|
| LLM / modèles | 9.5 % |
| Agents / outils | 6.2 % |
| Code / Python | 3.9 % |
| NLP langues locales | 2.9 % |
| Événements / communauté | 2.8 % |
| Entraînement / RL | 2.7 % |
| Maths / recherche | 2.2 % |
| Carrière / études | 2.0 % |
| Infra / GPU | 1.5 % |

**Messages per month:** 04/26: 198 · 05/26: 55 · 06/26: 1558 · 07/26: 506 · 08/26: 189 · 09/26: 612 · 10/26: 212 · 11/26: 53. June alone holds almost half of the messages.

**Top voices** (the 10 most active write 84 % of the messages). "Theme" is where the voice is most over-represented
compared with the group; "distinctive words" are its most characteristic words (words used by at least 4 voices only,
so names and typos are mostly filtered out):

| Voice | Messages | Share | Words/msg | Theme | Distinctive words |
|---|---|---|---|---|---|
| U10 | 1075 | 31.8 % | 19 | Agents / outils | humanité, action, donner |
| U23 | 910 | 26.9 % | 16 | Code / Python | etait, ailleurs, plutot |
| U69 | 280 | 8.3 % | 9 | Agents / outils | apres, contexte, genre |
| U27 | 141 | 4.2 % | 22 | Infra / GPU | risque, expertise, experts |
| U65 | 109 | 3.2 % | 17 | Événements / communauté | scolaire, sénégalais, opportunités |
| U29 | 76 | 2.2 % | 18 | NLP langues locales | accord, question |
| U28 | 72 | 2.1 % | 10 | Carrière / études | dernière, master |
| U02 | 65 | 1.9 % | 16 | Agents / outils | tokens, jeunes, agent |
| U36 | 61 | 1.8 % | 23 | Carrière / études | tourne, internet, offline |
| U89 | 60 | 1.8 % | 56 | Entraînement / RL | output, images, conversation |

```bash
uv run --no-project --with wordcloud python scripts/chat_figures.py   # figures, report/stats_data.tex, results/chat_stats.json
```

Caveats: themes are keyword-based (false positives on "data", "code"); word clouds drop stop words, laughs, file names and
words that are usually capitalized (likely names), but the anonymizer cannot see every name in free text. Review the
cloud and the voice words before publishing, and keep an optional `data/private/redact_words.txt` (one name per line,
git-ignored) for names you spot.

## Raw statistics (private)

`python3 scripts/whatsapp_stats.py` writes `data/whatsapp_questions_reponses/stats_discussion.md` from the raw export (profile
names included, git-ignored): anonymize before sharing it.
