import re,glob,collections,sys
from pathlib import Path
D=str(Path(__file__).resolve().parent.parent/'data/whatsapp_questions_reponses')+'/'
src=open(glob.glob(D+'Discussion*.txt')[0],encoding='utf8').read().split('\n')
p=re.compile(r'^(\d\d)/(\d\d)/(\d{4}), (\d\d):(\d\d) - ([^:]+?): ?(.*)')
msgs=[]
for l in src:
    m=p.match(l)
    if m: msgs.append([f'{m[3]}-{m[2]}-{m[1]}',int(m[4]),m[6],m[7]])
    elif msgs: msgs[-1][3]+='\n'+l
THEMES={
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
cre={k:re.compile(v,re.I) for k,v in THEMES.items()}
tot=collections.Counter();words=collections.Counter();q=collections.Counter();links=collections.Counter();longm=collections.Counter()
days=collections.defaultdict(set);month=collections.Counter();hour=collections.Counter()
th=collections.defaultdict(collections.Counter);thtot=collections.Counter()
for d,h,s,t in msgs:
    tot[s]+=1;w=len(t.split());words[s]+=w;q[s]+='?' in t;links[s]+='http' in t;longm[s]+=w>=50
    days[s].add(d);month[d[:7]]+=1;hour[h]+=1
    for k,r in cre.items():
        if r.search(t): th[k][s]+=1;thtot[k]+=1
N=len(msgs);S=len(tot)
def lab(s): return s
o=[]
a=o.append
a('# Statistiques de la discussion WhatsApp « Questions/Réponses »\n')
a(f'Source : `data/whatsapp_questions_reponses/Discussion_WhatsApp_avec_Discussions_-_QuestionsRéponses.txt`. Période : {min(m[0] for m in msgs)} à {max(m[0] for m in msgs)}.\n')
a(f'- **{N}** messages, **{S}** expéditeurs distincts, **{len({m[0] for m in msgs})}** jours actifs, **{sum(words.values())}** mots.')
top10=sum(c for _,c in tot.most_common(10));a(f'- Les 10 plus actifs écrivent **{top10/N:.0%}** des messages ; le premier à lui seul **{tot.most_common(1)[0][1]/N:.0%}**.')
a(f'- Les {sum(1 for c in tot.values() if c<=5)} expéditeurs à 5 messages ou moins forment {sum(c for c in tot.values() if c<=5)/N:.1%} du volume.\n')
a('Les identifiants `TEL_xxxx` sont les pseudonymes déjà présents dans le fichier ; les noms affichés sont ceux du profil WhatsApp tels quels.\n')
a('## 1. Top 20 expéditeurs\n\n| # | Expéditeur | Messages | % | Mots | Mots/msg | Jours actifs | Questions (« ? ») | Liens | Longs (≥50 mots) |\n|---|---|---|---|---|---|---|---|---|---|')
for i,(s,c) in enumerate(tot.most_common(20),1):
    a(f'| {i} | {s} | {c} | {c/N:.1%} | {words[s]} | {words[s]/c:.1f} | {len(days[s])} | {q[s]} | {links[s]} | {longm[s]} |')
a('\n## 2. Qui parle de quoi (thèmes par mots-clés)\n')
a('Un message compte pour un thème s’il contient un de ses mots-clés (un message peut compter pour plusieurs thèmes). Méthode approximative : mots-clés listés dans le script, pas d’analyse sémantique.\n')
a('| Thème | Messages | % du total | Top 5 contributeurs (messages) |\n|---|---|---|---|')
for k,c in thtot.most_common():
    a(f'| {k} | {c} | {c/N:.1%} | '+', '.join(f'{s} ({n})' for s,n in th[k].most_common(5))+' |')
a('\n### Spécialisation : thème où chaque top contributeur est le plus sur-représenté\n\nPart du thème dans ses messages, comparée à la moyenne du groupe (indice 1,0 = moyenne). Contributeurs de 40 messages ou plus.\n\n| Expéditeur | Thème le plus sur-représenté | Part | Indice |\n|---|---|---|---|')
for s,c in tot.most_common():
    if c<40: break
    best=max(THEMES,key=lambda k:(th[k][s]/c)/(thtot[k]/N) if thtot[k] else 0)
    a(f'| {s} | {best} | {th[best][s]/c:.0%} | {(th[best][s]/c)/(thtot[best]/N):.2f} |')
a('\n## 3. Activité dans le temps\n\n| Mois | Messages |\n|---|---|')
for m,c in sorted(month.items()): a(f'| {m} | {c} |')
a('\n| Heure (locale du fichier) | Messages |\n|---|---|')
for h in range(24): a(f'| {h:02d}h | {hour[h]} |')
a('\n## 4. Limites\n\n- Thèmes par mots-clés : faux positifs (« data », « code », « modèle ») et faux négatifs (messages sans mot-clé, en wolof, ou en anglais hors liste).\n- « Questions » = messages contenant « ? », y compris les rhétoriques.\n- Les messages multi-lignes sont rattachés à leur expéditeur ; les messages de média omis comptent comme messages.\n- Un même humain peut apparaître sous plusieurs identifiants.')
open(D+'stats_discussion.md','w',encoding='utf8').write('\n'.join(o)+'\n')
print(N,S)
