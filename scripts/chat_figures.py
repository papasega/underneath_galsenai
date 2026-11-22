"""Figures and statistics from the ANONYMIZED chat (safe to publish): theme shares, top voices, activity by month,
Senegal-shaped word cloud.

    uv run --no-project --with wordcloud python scripts/chat_figures.py [data/private/chat_anonymized.txt]

Writes report/stats_data.tex (pgfplots data for the report), assets/wordcloud.png, results/chat_stats.json.
Voices are the random labels (U01...) of the anonymized file: they are stable only for that file.
"""

import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from themes import SHORT, THEMES  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/private/chat_anonymized.txt"
LINE = re.compile(r"^(\d{4}-\d\d) \| (U\d+): (.*)")
STOP = set("""les des une que qui pour dans pas est sur avec plus par mais comme aussi tout tous cette ces son ses leur
être avoir fait faire peut peux sont était très bien donc alors ainsi quand même juste moi toi lui elles ils nous vous
elle mes tes mon ton vos nos aux dont où car puis après avant sans sous entre vers chez encore déjà ici cela ceci celui
the and for that this with you are have from not but can was will what your all has more about they one how just like
des l'on est-ce ça ca c'est j'ai n'est qu'il qu'on d'un d'une l'ia lien mention email numero nom membre omis médias
modifié message
été ont faut suis non oui pense dit dire etc hhh hhhh haha hahaha bcp deh rek meme même veux vois voir fois peu bon bonne
merci autres autre grand temps gens quoi vraiment comment joint webp fichier image jpg opus mp4 pdf vcf stk img ptt vid
dis veut vont sera soit avez avait vas vais chose choses quel quelques toujours encore depuis moins trop tant dont
beaucoup maintenant aujourd hui monde partie moment sais sinon yes yow bro hein elon raciste droite allah insha""".split())
PLACEHOLDERS = {"lien", "mention", "email", "numero", "nom", "membre"}

msgs: list[list[str]] = []
for raw in SRC.read_text(encoding="utf-8").splitlines():
    m = LINE.match(raw)
    if m:
        msgs.append([m[1], m[2], m[3]])
    elif msgs:
        msgs[-1][2] += "\n" + raw
N = len(msgs)
# Names the anonymizer cannot see are mostly written with a capital: drop words capitalized in >= 50% of uses.
_seen, _cap = collections.Counter(), collections.Counter()
for _, _, t in msgs:
    for tok in re.findall(r"[^\W\d_]{3,}", t):
        _seen[tok.lower()] += 1
        _cap[tok.lower()] += tok[0].isupper()
NAMELIKE = {w for w, n in _seen.items() if _cap[w] / n >= 0.5}
_private = ROOT / "data/private/redact_words.txt"  # optional, git-ignored: one name per line to keep out of the figures
if _private.exists():
    NAMELIKE |= {w.strip().lower() for w in _private.read_text(encoding="utf-8").splitlines() if w.strip()}
words = lambda t: [w for w in re.findall(r"[^\W\d_]{3,}", t.lower()) if w not in STOP and w not in NAMELIKE]

by_voice = collections.Counter(m[1] for m in msgs)
by_month = collections.Counter(m[0] for m in msgs)
cre = {k: re.compile(v, re.I) for k, v in THEMES.items()}
theme_n = collections.Counter()
voice_theme: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
voice_words: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
all_words: collections.Counter = collections.Counter()
voice_nwords: collections.Counter = collections.Counter()
for _, v, t in msgs:
    for k, r in cre.items():
        if r.search(t):
            theme_n[k] += 1
            voice_theme[v][k] += 1
    ws = words(t)
    all_words.update(ws)
    voice_words[v].update(ws)
    voice_nwords[v] += len(t.split())
total_words = sum(all_words.values())
voices_using = collections.Counter(w for c in voice_words.values() for w in c)  # in how many voices a word appears

top = by_voice.most_common(10)
rows = []
for v, c in top:
    t = voice_theme[v]
    # most over-represented theme: share in the voice's messages over share in the whole chat
    best = max(THEMES, key=lambda k: (t[k] / c) / (theme_n[k] / N) if theme_n[k] else 0)
    nv = sum(voice_words[v].values())
    dist = sorted(
        (w for w, n in voice_words[v].items() if n >= 4 and len(w) >= 5 and w not in PLACEHOLDERS and voices_using[w] >= 4),  # words used by >= 4 voices: skips names and typos
        key=lambda w: -math.log((voice_words[v][w] / nv) / (all_words[w] / total_words)),
    )[:3]
    rows.append({"voice": v, "messages": c, "share": c / N, "words_per_msg": voice_nwords[v] / c,
                 "theme": SHORT[best], "distinctive": dist})

# ---- pgfplots data for the report ----
def bars(name: str, items: list[tuple[str, float]], *, horizontal: bool = True, width: str = r"\linewidth", height: str = "4.4cm") -> str:
    coords = " ".join(f"({v:g},{{{k}}})" if horizontal else f"({{{k}}},{v:g})" for k, v in items)
    symb = ",".join(f"{{{k}}}" for k, _ in items)
    axis = "ybar" if not horizontal else "xbar"
    sym = f"symbolic {'y' if horizontal else 'x'} coords={{{symb}}}"
    extra = "y dir=reverse, xmin=0, ytick=data," if horizontal else "xtick=data, x tick label style={rotate=45,anchor=east}, ymin=0,"
    return (f"\\newcommand{{\\{name}}}{{\\begin{{tikzpicture}}\\begin{{axis}}[{axis}, width={width}, height={height}, bar width=5pt, "
            f"{sym}, {extra} enlarge y limits=0.07, enlarge x limits=0.07, axis lines*=left, tick style={{draw=none}}, "
            f"nodes near coords, nodes near coords style={{font=\\fontsize{{6}}{{7}}\\selectfont}}, "
            f"every node near coord/.append style={{/pgf/number format/fixed}}, "
            f"tick label style={{font=\\fontsize{{6.5}}{{7.5}}\\selectfont}}, ymajorgrids=false]"
            f"\\addplot[fill=DeepGreen,draw=none] coordinates {{{coords}}};\\end{{axis}}\\end{{tikzpicture}}}}\n")

themes = [(SHORT[k], 100 * c / N) for k, c in theme_n.most_common()]
months = sorted(by_month.items())
voices = [(r["voice"], r["messages"]) for r in rows]
(ROOT / "report/stats_data.tex").write_text(
    "% Generated by scripts/chat_figures.py from the anonymized chat. Do not edit.\n"
    + bars("themechart", [(k, round(v, 1)) for k, v in themes], width=r"0.27\linewidth", height="4.9cm")
    + bars("monthchart", [(m[5:] + "/" + m[2:4], c) for m, c in months], horizontal=False, width=r"0.34\linewidth", height="4.3cm")
    + bars("voicechart", voices, width=r"0.22\linewidth", height="5.4cm")
    + "\\newcommand{\\voicerows}{"
    + " \\\\\n".join(
        f"{r['voice']} & {r['messages']} & {r['share']:.1%} & {r['words_per_msg']:.0f} & {r['theme']} & "
        + ", ".join(r["distinctive"]) for r in rows).replace("%", r"\%")
    + " \\\\}\n"
    + f"\\newcommand{{\\statN}}{{{N}}}\\newcommand{{\\statVoices}}{{{len(by_voice)}}}"
    + f"\\newcommand{{\\statTop}}{{{round(100 * sum(c for _, c in top) / N)}\\%}}\n",
    encoding="utf-8")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/chat_stats.json").write_text(json.dumps(
    {"messages": N, "voices": len(by_voice), "themes": themes, "months": months, "top_voices": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- Senegal-shaped word cloud ----
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from wordcloud import WordCloud  # noqa: E402

path = next(l for l in (ROOT / "report/senegal.tex").read_text().splitlines() if "\\fill[DeepGreen]" in l)
pts = [(float(a), float(b)) for a, b in re.findall(r"\(([\d.]+),([\d.]+)\)", path)]
S = 110  # pixels per map unit
w, h = int(max(p[0] for p in pts) * S) + 20, int(max(p[1] for p in pts) * S) + 20
mask = Image.new("L", (w, h), 255)
ImageDraw.Draw(mask).polygon([(10 + x * S, h - 10 - y * S) for x, y in pts], fill=0)
freq = {k: v for k, v in all_words.items() if k not in PLACEHOLDERS and len(k) >= 3}
greens = ["#1e7a32", "#31ad48", "#14521f", "#50b963", "#001818"]
cloud = WordCloud(width=w, height=h, mask=np.array(mask), background_color="white", max_words=170, contour_width=4,
                  contour_color="#1e7a32", color_func=lambda *a, random_state=None, **k: greens[random_state.randint(0, 4)] if random_state else greens[0],
                  prefer_horizontal=0.9, relative_scaling=0.45, random_state=1).generate_from_frequencies(freq)
cloud.to_file(str(ROOT / "assets/wordcloud.png"))
print(f"{N} messages, {len(by_voice)} voices; top words:", [w for w, _ in all_words.most_common(12)])
