"""Render measured v3 results, preserving corpus/procedural distinctions."""
import json
import csv
import random
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffuthink.v2.data import digest
from diffuthink.storypatch.corrections import read
from diffuthink.storypatch.context_evaluate import blend


def main():
    out=ROOT/'reports/storypatch-context-v3';report=json.loads((out/'evaluation.json').read_text())
    run=ROOT/'runs/storypatch-context-ranker-v3';info=json.loads((run/'best/ranker_info.json').read_text())
    assert digest(run/'best/model.safetensors')==report['binding']['ranker_sha256']
    for name in ['learning.jsonl']:shutil.copy2(run/name,out/name)
    shutil.copy2(run/'best/ranker_info.json',out/'ranker_info.json')
    shutil.copy2(ROOT/'data/processed/storypatch-context-v3/manifest.json',out/'corpus_manifest.json')
    # A reproducible stratified sample, with methods hidden and judgments blank.
    rows=blend(read(out/'test_predictions.jsonl'),info['blending']['weight']);rng=random.Random(193)
    review=[]
    for corpus in [True,False]:
        for clean in [True,False]:
            group=[r for r in rows if (r['case']['category']=='corpus')==corpus and r['case']['clean']==clean]
            review.extend(rng.sample(group,25))
    rng.shuffle(review);key={}
    with (out/'blind_review.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['id','original','selected','variant_A','variant_B','preferred_A_B_tie','reason'])
        for row in review:
            c=row['case'];base=min(row['candidates'],key=lambda x:x['local_nll'],default={}).get('replacement',row['original_fragment'])
            learned=max(row['candidates'],key=lambda x:x['ranker_score'],default={}).get('replacement',row['original_fragment'])
            options=[('nll',base),('combined',learned)];rng.shuffle(options)
            writer.writerow([c['id'],c['text'],row['original_fragment'],*[c['prefix']+r[1]+c['suffix'] for r in options],'',''])
            key[c['id']]=[r[0] for r in options]
    (out/'blind_review_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    (out/'DATA_LICENSE.md').write_text('''# Evaluation text attribution

This directory contains synthetic TinyStories excerpts and transformations.
Source: https://huggingface.co/datasets/roneneldan/TinyStories
Revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
TinyStories contributors retain their source rights. Dataset excerpts and
derived examples retain CDLA-Sharing-1.0 terms:
https://cdla.dev/sharing-1-0/
Procedural templates, code and model weights use Apache-2.0.
''',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['Corpus excerpts','Procedural templates'];domains=['corpus','procedural']
    for offset,key,label in [(-.18,'same_pool_error_top1_nll','NLL'),(.18,'same_pool_error_top1_ranker','Combined')]:
        ax[0].bar([i+offset for i in range(2)],[100*report['test'][d][key] for d in domains],width=.36,label=label)
    ax[0].set_xticks(range(2),labels);ax[0].set_ylim(0,100);ax[0].set_ylabel('Top1 reference match (%)');ax[0].legend()
    curve=[json.loads(s) for s in (run/'learning.jsonl').read_text().splitlines()]
    ax[1].plot([r['step'] for r in curve],[r['bce'] for r in curve]);ax[1].set_xlabel('Update');ax[1].set_ylabel('Validation BCE')
    fig.suptitle('StoryPatch v3: fixed candidate pools, synthetic references, no human ratings')
    fig.savefig(out/'results.png',dpi=160);plt.close(fig)
    table=[]
    for domain,title in [('corpus','Extraits du corpus'),('procedural','Gabarits procéduraux')]:
        r=report['test'][domain];old=r['nll_keep_or_replace'];new=r['learned_keep_or_replace']
        table.append(f"| {title} : première variante conforme | {r['same_pool_error_top1_nll']:.1%} | {r['same_pool_error_top1_ranker']:.1%} |")
        for label,key in [('réparation conseillée','error_corrected'),('original correct conservé','clean_preserved')]:
            table.append(f"| {title} : {label} | {old[key]:.1%} | {new[key]:.1%} |")
    legacy=report['legacy']['summary'];validation=info['calibration']['validation']
    text=f'''# StoryPatch v3 : classement contextuel

Un classeur de **{info['parameters']:,} paramètres** a été entraîné sur
**24 000 contextes**, dont 12 000 extraits variés du corpus. Il reprend le
backbone de notre propre DiffuThink, dont la lignée part d’une initialisation
aléatoire, puis apprend une nouvelle tâche de classement. Aucun poids externe.
Checkpoint retenu : **étape {info['step']}**, choisi par validation.

## Résultats sur 1 200 entrées / 600 contextes

Moitié extraits TinyStories, moitié gabarits. Chaque contexte possède une
version altérée et une version correcte. Toutes les étiquettes sont
synthétiques ; les extraits servent de références de reconstruction, sans
évaluation humaine de toutes les formulations possibles.

| Mesure | NLL | Classement combiné |
|---|---:|---:|
{chr(10).join(table)}

Sur les 36 anciens diagnostics réutilisés : **{legacy['nll']:.1%}** de
premières variantes acceptées avec la NLL, **{legacy['blended']:.1%}** avec
le classement combiné. Sur exactement les mêmes propositions, le précédent
classeur v2 obtient **{report['legacy']['previous_ranker_matched_pool']['top1']:.1%}**
(scores FP32 sur CPU). Ils ne constituent pas un test nouveau ou indépendant.

![Mesures et courbe](results.png)

La couverture des réponses de référence est de
**{report['test']['corpus']['error_candidate_coverage']:.1%}** dans les extraits
et **{report['test']['procedural']['error_candidate_coverage']:.1%}** dans les
gabarits. Reclasser ne peut pas créer une réponse absente des propositions.
Le générateur est identique dans toutes les comparaisons de cette expérience.

Le classement s’améliore, mais le conseil de remplacement est **très
conservateur** : il répare seulement
**{report['test']['all']['learned_keep_or_replace']['error_corrected']:.1%}** des
entrées altérées, contre **{report['test']['all']['nll_keep_or_replace']['error_corrected']:.1%}**
avec le conseil fondé sur la NLL seule. Les propositions restent consultables
et applicables manuellement. Sur les extraits du corpus, le taux de remplacement
hors référence atteint **{report['test']['corpus']['learned_keep_or_replace']['wrong_replacement_on_errors']:.1%}** :
la contrainte globale de validation ne se généralise pas à ce sous-groupe.

## Choix sur validation

Poids du score appris : **{info['blending']['weight']:g}** dans
`-NLL + poids × clip(score_appris, -8, 8)`. La validation compare séparément
les deux domaines, exigeant l’absence de régression de classement dans chacun.
Le score borné évite qu’une prédiction extrême domine sans limite la NLL.

Marge de remplacement : **{info['calibration']['margin_threshold']:g}** ;
score minimal : **{info['calibration']['score_floor']:g}**.
Sur les 480 entrées de calibrage : réparation {validation['error_corrected']:.1%},
conservation {validation['clean_preserved']:.1%}, remplacement hors référence
sur entrées altérées {validation['wrong_replacement_on_errors']:.1%}.
Ces contraintes ne garantissent pas les mêmes taux sur d’autres textes.

L’exploration NLL reste disponible par défaut. Le nouveau modèle s’essaie via
« Classement contextuel · expérimental ». Les résultats ne justifient pas
de prétendre à un correcteur général ou à une compréhension fiable.

Le fichier `evaluation.json` contient les résultats complets, les 36 diagnostics
et les empreintes. Les fichiers `*_predictions.jsonl` conservent toutes les
propositions. `selection.json` garde toute la grille essayée en validation.
`blind_review.csv` prépare 100 comparaisons avec ordre des méthodes masqué,
réparties entre domaines et entrées correctes/altérées. Les jugements restent
vides : aucune évaluation humaine n’a été effectuée.
[Protocole](../../docs/STORYPATCH_CONTEXT_V3.md) ·
[Attribution des extraits](DATA_LICENSE.md).
'''
    (out/'RESULTATS.md').write_text(text,encoding='utf-8');print(str(out/'RESULTATS.md'))


if __name__=='__main__':main()
