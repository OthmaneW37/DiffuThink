"""Bind measured evidence and draw the actual paired test results."""
import json
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffuthink.v2.data import digest


def main():
    report_dir=ROOT/'reports/storypatch-ranker-v2'
    report=json.loads((report_dir/'evaluation.json').read_text())
    ranker=ROOT/'runs/storypatch-ranker-v2/best'
    info=json.loads((ranker/'ranker_info.json').read_text())
    if digest(ranker/'model.safetensors')!=report['binding']['ranker_weights_sha256']:raise ValueError('Weights changed')
    old,new=report['end_to_end']['nll_keep_or_replace'],report['end_to_end']['learned_keep_or_replace']
    validation=info['calibration']['validation']
    passed=validation['balanced_success']>.5 and validation['clean_modified']<=.05+1e-9 and validation['wrong_replacement_on_errors']<=.1+1e-9
    selection={'generator':'runs/storypatch-13m/best','ranker':'runs/storypatch-ranker-v2/best',
        'generator_weights_sha256':report['binding']['generator_weights_sha256'],
        'ranker_weights_sha256':report['binding']['ranker_weights_sha256'],'enabled':passed,
        'rule':'Enable only if validation balanced success > always-keep (50%), clean modified <=5%, wrong error replacements <=10%. Test is descriptive only.',
        'validation':validation,'calibration':info['calibration'],'ui_default':'nll',
        'availability':'Experimental opt-in; legacy diagnostic ranking regresses.'}
    (report_dir/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    for version in ['v1','v2']:
        run=ROOT/f'runs/storypatch-ranker-{version}';destination=ROOT/f'reports/storypatch-ranker-{version}'
        shutil.copy2(run/'learning.jsonl',destination/'learning.jsonl')
        shutil.copy2(run/'best/ranker_info.json',destination/'ranker_info.json')
    corpus=ROOT/'data/processed/storypatch-corrections-mined-v2'
    shutil.copy2(corpus/'manifest.json',report_dir/'corpus_manifest.json')
    shutil.copy2(corpus/'mining.jsonl',report_dir/'mining.jsonl')
    # A frozen copy of test and calibration inputs makes outputs independently inspectable.
    for name in ['test.jsonl','validation.jsonl']:shutil.copy2(corpus/name,report_dir/name)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(left,right)=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['Repair altered input','Preserve clean input','Balanced success']
    keys=['error_corrected','clean_preserved','balanced_success']
    x=range(3)
    left.bar([i-.18 for i in x],[100*old[k] for k in keys],width=.36,label='Causal NLL')
    left.bar([i+.18 for i in x],[100*new[k] for k in keys],width=.36,label='Learned ranker + keep')
    left.set_xticks(list(x),labels,rotation=12);left.set_ylim(0,105);left.set_ylabel('Synthetic reference score (%)');left.legend(fontsize=8)
    for version in ['v1','v2']:
        curve=[json.loads(line) for line in (ROOT/f'runs/storypatch-ranker-{version}/learning.jsonl').read_text().splitlines()]
        right.plot([r['step'] for r in curve],[r['bce'] for r in curve],label=version)
    right.set_yscale('log');right.set_xlabel('Training update');right.set_ylabel('Validation BCE');right.legend()
    fig.suptitle('StoryPatch: 600 procedural cases / 300 paired contexts — no human rating')
    fig.savefig(report_dir/'results.png',dpi=160);plt.close(fig)
    e=report['end_to_end']; ci=report['paired_bootstrap']['percentile_95']
    table='\n'.join(f'| {label} | {100*old[key]:.1f}% | {100*new[key]:.1f}% |' for label,key in [
        ('Réparation des entrées altérées','error_corrected'),('Conservation exacte des entrées correctes','clean_preserved'),
        ('Réussite équilibrée','balanced_success'),('Remplacement hors référence sur entrées altérées','wrong_replacement_on_errors')])
    legacy=json.loads((report_dir/'legacy_probes.json').read_text())['summary'] if (report_dir/'legacy_probes.json').exists() else None
    legacy_text=(f"Sur les 36 anciens diagnostics : première variante acceptée {legacy['nll_top1']:.1%} avec la NLL, "
                 f"contre {legacy['learned_top1']:.1%} avec le classeur ; recommandation acceptée {legacy['recommended']:.1%}. "
                 "Ces probes ne servent ni au choix des poids ni au calibrage. Cette régression conduit à conserver "
                 "la NLL par défaut dans l’interface et à proposer le classeur comme mode expérimental explicite. "
                 "Voir `legacy_probes.json`." if legacy else 'Anciens diagnostics non mesurés.')
    text=f'''# StoryPatch : résultats du classeur appris

Un réseau indépendant de **{info['parameters']:,} paramètres**, initialisé
aléatoirement, classe les variantes du générateur DiffuThink de 13,44 M.
Le générateur et ses poids restent inchangés dans cette expérience.
Le checkpoint retenu est l’étape **{info['step']}**, choisie par BCE de validation.
Disponible en mode expérimental : **{'oui' if passed else 'non, échec des critères de validation'}**.
Le classement habituel NLL reste le mode par défaut de l’interface.

## Test figé : 600 cas, 300 contextes

300 versions altérées et 300 correctes ; six catégories de récits simples.
Étiquettes procédurales synthétiques, sans validation humaine. Les deux
méthodes reçoivent les mêmes propositions. Le classement NLL peut conseiller
de garder l’original si aucune proposition ne réduit sa NLL.

| Mesure | NLL + conservation | Classement appris + conservation |
|---|---:|---:|
{table}

L’intervalle descriptif de la différence de réussite équilibrée, bootstrap
apparié par contexte (1 000 tirages), est [{100*ci[0]:.1f}, {100*ci[1]:.1f}]
points. Il ne mesure pas la validité sur du texte libre.

Sans l’option de conservation, la première variante correspond à une réponse
acceptée dans **{e['same_pool_error_top1_nll']:.1%}** des entrées altérées avec
la NLL, contre **{e['same_pool_error_top1_ranker']:.1%}** avec le classeur.
La réponse acceptée est disponible quelque part dans les propositions dans
**{e['error_candidate_coverage']:.1%}** des cas. Le classeur ne peut pas créer
une réponse absente de cette liste. Préservation du contexte :
**{e['context_preserved']:.1%}**, par composition exacte des chaînes.

![Comparaison et courbes](results.png)

## Régression hors gabarits

{legacy_text}

## Décision et provenance

Seuils choisis sur 240 cas de validation, jamais sur le test : marge
**{info['calibration']['margin_threshold']:g}**, score minimal
**{info['calibration']['score_floor']:g}**. Le score est une préférence
apprise, sans interprétation comme probabilité de correction.
Validation : réparation {validation['error_corrected']:.1%}, conservation
{validation['clean_preserved']:.1%}, remplacement hors référence sur
entrées altérées {validation['wrong_replacement_on_errors']:.1%}.

12 000 contextes d’entraînement ; 2 699 propositions supplémentaires
extraites sur 1 200 de ces contextes et étiquetées avec une liste de réponses
fermée. Certaines formulations valides peuvent être étiquetées à tort.
Le corpus est procédural, avec gabarits séparés et contextes dédupliqués.
Les noms, mots et structures restent proches entre splits.

La première expérience, conservée dans `../storypatch-ranker-v1`, avait
appris à classer des contrastes trop simples. Son calibrage choisissait de
tout conserver. Le deuxième corpus et ses alternatives explicites ont été
figés avant le second entraînement ; les décisions d’amélioration sont
issues de la validation de la première expérience. Les 600 nouveaux cas
ne constituent pas une étude indépendante d’une équipe tierce.

`evaluation.json` contient les résultats par catégorie et les empreintes.
`test_predictions.jsonl` contient toutes les sorties. `selection.json`
documente la règle de déploiement fondée sur la validation.
`blind_review.csv` prépare une comparaison humaine masquée ; ses colonnes
de jugement sont **vides**. Aucun gain confirmé par des personnes n’est annoncé.

[Protocole et reproduction](../../docs/STORYPATCH_RANKER.md).
Développement et documentation assistés par IA. Résultat local ; cette étape
n’a pas téléversé de nouveau modèle sur Hugging Face ou GitHub.
'''
    (report_dir/'RESULTATS.md').write_text(text,encoding='utf-8');print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
