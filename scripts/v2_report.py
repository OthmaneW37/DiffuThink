"""Collect measured runs, plot validation curves and write a portfolio report."""
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output=Path("reports/v2")
    output.mkdir(parents=True,exist_ok=True)
    phases=[Path("runs/stories-v2"),Path("runs/stories-v2-refined")]
    reports=[]
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    offset=0
    for i,phase in enumerate(phases,1):
        report=json.loads((phase/"report.json").read_text())
        reports.append(report)
        shutil.copy2(phase/"report.json",output/f"phase{i}.json")
        shutil.copy2(phase/"metrics.jsonl",output/f"phase{i}_metrics.jsonl")
        rows=[json.loads(line) for line in (phase/"metrics.jsonl").read_text().splitlines()]
        validation=[r for r in rows if "validation_loss" in r]
        ax.plot([offset+r["step"] for r in validation],[r["validation_loss"] for r in validation],"o-",label=f"Phase {i}",linewidth=2,markersize=4)
        offset+=report["training"]["steps"]
    ax.set(xlabel="Cumulative optimizer updates",ylabel="Mean masked validation cross-entropy (nats)",title="DiffuThink-13M · measured validation learning curve")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output/"learning_curve.png",dpi=180)
    fig.savefig(output/"learning_curve.svg")
    plt.close(fig)
    final=reports[-1]
    benchmark=json.loads((output/"sampling_test.json").read_text())
    samples="\n\n".join(f"**Prompt :** {s['prompt']}\n\n> {s['text']}" for s in benchmark["samples"])
    table="\n".join(f"| {r['mask_rate']:.0%} | {r['accuracy']:.2%} | {r['top5_accuracy']:.2%} | {r['cross_entropy']:.3f} | {r['unigram_accuracy']:.2%} |" for r in final["test"])
    samplers="\n".join(f"| {r['policy']} | {r['masked_accuracy']:.2%} | {r['mean_forward_passes']:.2f} | {r['mean_latency_ms']:.1f} ms |" for r in benchmark["results"])
    total_tokens=sum(r["seen_tokens"] for r in reports)
    seconds=sum(r["seconds"] for r in reports)
    body=f'''# DiffuThink V2 — résultats mesurés

**13 337 280 paramètres, {offset:,} mises à jour, {total_tokens:,} tokens non-PAD présentés au réseau** (incluant les répétitions et tokens de contrôle), environ {seconds/60:.1f} minutes cumulées de boucle d'entraînement/validation. Ce temps exclut préparation et installation. GPU : {final['device_name']}, BF16. Le corpus contient 21 134 655 occurrences de tokens textuels avant répétitions, pas {total_tokens:,} tokens uniques.

Les deux phases proviennent de la même initialisation apprise depuis zéro. La seconde redémarre l'optimiseur à un learning rate plus faible. Les poids sélectionnés par validation proviennent de l'étape {final['best']['step']} de cette seconde phase. Les rapports JSON incluent toute la filiation.

![Courbe mesurée](learning_curve.png)

## Reconstruction sur test

512 fenêtres tenues à l'écart, masques reproductibles ; unité : sous-mot BPE.

| Masquage | Accuracy top-1 | Top-5 | CE en nats | Unigramme top-1 |
| --- | ---: | ---: | ---: | ---: |
{table}

Ces valeurs ne se comparent pas aux 89,1 % de la V1 (autres données, autres unités). Elles ne mesurent pas la cohérence des générations.

## Comparaison des politiques

{benchmark['examples']} fenêtres de test, mêmes masques à 50 %, température zéro. GPU BF16 après chauffe. Pas d'intervalle de confiance : les petits écarts ne constituent pas une preuve de supériorité.

| Politique | Accuracy masquée | Passes moyennes | Latence moyenne |
| --- | ---: | ---: | ---: |
{samplers}

**Bigramme bidirectionnel contextuel : {benchmark['bigram']['masked_accuracy']:.2%}** sur exactement les mêmes positions. Il ne voit que les voisins non masqués. Le coût de préparation du bigramme n'est pas inclus dans une comparaison de latence.

Le budget adaptatif est une heuristique et son gain doit être jugé sur ces mesures. Une passe peut être compétitive en reconstruction même si la démonstration utilise plusieurs passes. Aucun résultat d'état de l'art n'est revendiqué.

## Générations non filtrées

Prompts fixés à l'avance, seed 42, température 0,7, blocs de 16, au plus 64 nouveaux tokens. Toutes les sorties du benchmark sont reproduites ici, y compris celles qui sont maladroites. Aucun choix du meilleur parmi plusieurs essais.

{samples}

## Limites

Récits synthétiques anglais, vocabulaire et domaine limités ; répétitions, incohérences et erreurs grammaticales possibles. Déduplication exacte seulement. Les partitions sont disjointes par document, mais des récits similaires peuvent subsister. Une seule trajectoire d'entraînement en deux phases, pas une étude multigraines. Le benchmark de reconstruction n'est pas une évaluation humaine de qualité. Les rapports intermédiaires de test sont conservés ; ce travail reste exploratoire.

Le projet et le code ont été développés avec assistance IA. Les contributions à présenter sont le pipeline, les choix expliqués et les expériences réellement reproduites personnellement.
'''
    (output/"RESULTATS.md").write_text(body,encoding="utf-8")
    print(body[:500])


if __name__=="__main__":
    main()
