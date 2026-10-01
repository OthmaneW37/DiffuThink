"""Select on validation only, then render the complete held-out comparison."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from diffuthink.storypatch.data import load
from diffuthink.storypatch.train import span_evaluate
from diffuthink.v2.inference import load as load_model
from diffuthink.v2.hybrid import causal_evaluate


def main():
    os.chdir(ROOT)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precision = 'bf16' if device == 'cuda' else 'fp32'
    out = ROOT/'reports'/'storypatch-v2'
    specs = [('Published 13M', 'runs/stories-512/best', 'baseline'),
             ('Adapted 13M', 'runs/storypatch-13m/best', 'adapted13m'),
             ('Scratch 58M', 'runs/storypatch-58m-edit/best', 'scratch58m')]
    arrays, _ = load(ROOT/'data'/'processed'/'storypatch-v2')
    rows = []
    for label, directory, report_name in specs:
        model, _ = load_model(directory, device)
        span = span_evaluate(model, arrays['validation'], 256, precision)
        causal = causal_evaluate(model, arrays['validation'][0], 256, precision=precision)
        report = json.loads((out/f'{report_name}.json').read_text(encoding='utf-8'))
        info = json.loads((ROOT/directory/'training_info.json').read_text())
        rows.append({'label':label, 'model':directory, 'validation_span':span, 'validation_causal':causal,
                     'selection_score':span['nll']+.25*causal['nll'], 'report':report_name+'.json',
                     'test':{k:report[k] for k in ('span','causal','iterative_reconstruction','editor')},
                     'parameters':info['parameters'], 'weights_sha256':report['weights_sha256']})
        del model
    baseline = rows[0]
    eligible = [r for r in rows[1:] if r['selection_score'] < baseline['selection_score']
                and r['validation_span']['nll'] < baseline['validation_span']['nll']
                and r['validation_causal']['nll'] <= baseline['validation_causal']['nll']*1.1]
    selected = min(eligible, key=lambda r:r['selection_score']) if eligible else baseline
    selection = {'model':selected['model'], 'weights_sha256':selected['weights_sha256'],
                 'rule':'Lowest validation span NLL + .25 causal NLL; span improvement and <=10% causal NLL regression required',
                 'test_not_used_for_selection':True,
                 'validation':[{k:v for k,v in r.items() if k != 'test'} for r in rows]}
    (ROOT/'runs'/'storypatch-selected.json').write_text(json.dumps(selection, indent=2))
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    lines = ['# StoryPatch v2 — résultats mesurés', '',
             f"Modèle retenu sur validation : **{selected['label']}** (`{selected['model']}`).", '',
             'La sélection utilise uniquement 256 fenêtres de validation. Le score fixé avant entraînement est '
             '`NLL span + 0,25 × NLL causale`, avec une amélioration de reconstruction et au plus 10 % de régression de NLL causale.', '',
             'Le gain reste modeste : le premier choix accepté passe de 25/36 à 26/36, tandis que le top-3 passe de 29/36 à 28/36. '
             'La perplexité causale se dégrade légèrement. Le modèle de 58M reste inférieur avec le budget testé ; il n’est pas retenu pour la démo. '
             'La précision de reconstruction progresse, mais aucune amélioration générale ou statistiquement démontrée de cohérence n’est revendiquée.', '',
             '## Test indépendant', '',
             '| Mesure | Publié 13M | Adapté 13M | Depuis zéro 58M |',
             '|---|---:|---:|---:|']
    metrics = [
        ('NLL reconstruction (400 passages)', lambda t:f"{t['span']['nll']:.3f}"),
        ('Tokens reconstruits correctement', lambda t:f"{t['span']['token_accuracy']:.1%}"),
        ('Reconstruction parallèle exacte', lambda t:f"{t['span']['parallel_exact_match']:.1%}"),
        ('Reconstruction itérative exacte (64 passages)', lambda t:f"{t['iterative_reconstruction']['exact_match']:.1%}"),
        ('Perplexité causale (mêmes 400 fenêtres)', lambda t:f"{t['causal']['perplexity']:.3f}"),
        ('Premier choix accepté (36 cas)', lambda t:f"{t['editor']['top1']:.1%}"),
        ('Au moins un choix accepté dans les 3 premiers', lambda t:f"{t['editor']['top3']:.1%}"),
        ('Contexte conservé hors sélection', lambda t:f"{t['editor']['context_preserved']:.1%}"),
        ('Latence médiane éditeur (ms, GPU local)', lambda t:f"{t['editor']['median_latency_ms']:.0f}"),
    ]
    for label, metric in metrics:
        lines.append('| '+label+' | '+' | '.join(metric(r['test']) for r in rows)+' |')
    lines += ['', 'Les scores de reconstruction comparent la sortie au texte source : des variantes valides peuvent être comptées comme fausses. '
              'Les 36 cas sont des sondes synthétiques rédigées pour le diagnostic, sans revue humaine aveugle. '
              'Leur liste de réponses acceptées est incomplète. Aucun score général de cohérence ou de correction grammaticale n’est revendiqué.', '',
              'Une vérification CPU est conservée dans `adapted13m_cpu.json`. Elle a tourné pendant l’entraînement GPU ; '
              'sa latence inclut la concurrence pour les ressources et ne constitue pas une mesure de performance isolée. '
              'FP32 sur CPU et BF16 sur GPU peuvent donner des propositions et scores de sondes différents.', '',
              'Les valeurs de perplexité ci-dessus utilisent les mêmes fenêtres pour les trois modèles ; elles ne sont pas directement comparables '
              'au score historique de 6,147 sur des fenêtres de 192 tokens.', '',
              '## Tous les premiers choix, y compris les échecs', '',
              '| ID | Passage original | Publié 13M | Adapté 13M | Depuis zéro 58M |', '|---|---|---|---|---|']
    for index, case in enumerate(rows[0]['test']['editor']['samples']):
        values = []
        for row in rows:
            sample = row['test']['editor']['samples'][index]
            top = sample['candidates'][0]['replacement'] if sample['candidates'] else '(aucun)'
            values.append(top.replace('|','\\|')+(' ✓' if sample['top1'] else ''))
        lines.append('| '+case['id']+' | '+case['text'][case['start']:case['end']]+' | '+' | '.join(values)+' |')
    lines += ['', 'Les JSON associés contiennent les textes complets, toutes les propositions, les métriques par catégorie et les empreintes des fichiers.', '',
              '## Entraînement', '']
    for name in ('storypatch-13m','storypatch-58m','storypatch-58m-causal','storypatch-58m-edit'):
        status = json.loads((ROOT/'runs'/name/'status.json').read_text())
        info = json.loads((ROOT/'runs'/name/'best'/'training_info.json').read_text())
        lines.append(f"- `{name}` : {status['step']:,} mises à jour, {status['seconds']/60:.1f} minutes ; checkpoint retenu à l’étape {info['step']:,}. "
                     f"À ce checkpoint : {info['seen_tokens']:,} tokens présentés, {info['supervised_tokens']:,} cibles supervisées.")
    lines += ['', 'Le pilote 58M entraîné directement avec les deux objectifs a stagné près de 6 % de précision de reconstruction. '
              'Il a été interrompu après le checkpoint 2 500 ; quelques mises à jour ultérieures non sauvegardées ont été abandonnées. '
              'Ses poids ont ensuite suivi une consolidation causale puis une spécialisation sur les passages. '
              'Ce changement a été décidé sur validation, avant les scores test des candidats. Voir `pilot_notes.json`.', '',
              'Le modèle de 58M part de poids aléatoires ; celui de 13M poursuit notre propre modèle appris depuis zéro. '
              'Cette comparaison ne contrôle pas tout l’historique de calcul et ne permet pas d’attribuer un résultat à la taille seule.', '',
              '![Courbes de validation](learning_curves.png)', '',
              'Protocole : [STORYPATCH_V2.md](../../docs/STORYPATCH_V2.md). Texte source TinyStories : CDLA-Sharing-1.0. '
              'Code : Apache-2.0. Développement assisté par IA.']
    (out/'RESULTATS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for name, label in [('storypatch-13m','Adapted 13M'),('storypatch-58m','58M mixed pilot'),
                        ('storypatch-58m-causal','58M causal consolidation'),('storypatch-58m-edit','58M span adaptation')]:
        entries = [json.loads(line) for line in (ROOT/'runs'/name/'metrics.jsonl').read_text().splitlines()]
        entries = [r for r in entries if 'validation_span' in r]
        for ax, key in zip(axes, ('validation_span','validation_causal')):
            ax.plot([r['step'] for r in entries], [r[key]['nll'] for r in entries], label=label)
            ax.set_xlabel('Optimizer updates within each phase')
            ax.set_ylabel(key.replace('validation_', '').title()+' NLL (nats)')
            ax.grid(alpha=.2)
    axes[0].axhline(baseline['validation_span']['nll'], color='grey', linestyle='--',label='Published 13M')
    axes[1].axhline(baseline['validation_causal']['nll'], color='grey', linestyle='--',label='Published 13M')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'learning_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
