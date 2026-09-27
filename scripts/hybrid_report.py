"""Publish measured hybrid training curves and every fixed-prompt comparison."""
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    run=Path("runs/stories-hybrid")
    out=Path("reports/hybrid")
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((run/"report.json").read_text())
    initial=json.loads((run/"initial_validation.json").read_text())
    comparisons=json.loads((out/"continuations.json").read_text(encoding="utf-8"))
    rows=[json.loads(s) for s in (run/"metrics.jsonl").read_text().splitlines()]
    validation=[r for r in rows if "validation" in r]
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    ax.plot([r["step"] for r in validation],[r["validation"]["perplexity"] for r in validation],"o-",color="#147d64")
    ax.set(xlabel="Hybrid optimizer updates",ylabel="Causal validation perplexity",title="DiffuThink · learning continuation")
    ax.grid(alpha=.2)
    fig.savefig(out/"learning_curve.png",dpi=180)
    fig.savefig(out/"learning_curve.svg")
    plt.close(fig)
    for file in ("report.json","metrics.jsonl","settings.json","initial_validation.json"):
        shutil.copy2(run/file,out/file)
    metric=report["causal_test"]
    before_rep=sum(x["before"]["repeated_trigram_fraction"] for x in comparisons["examples"])/len(comparisons["examples"])
    after_rep=sum(x["after"]["repeated_trigram_fraction"] for x in comparisons["examples"])/len(comparisons["examples"])
    text=f'''# DiffuThink Hybrid — résultats

Nouvelle phase : **{report['settings']['steps']:,} mises à jour**, {report['seen_tokens']:,} tokens non-PAD présentés au réseau, {report['seconds']/60:.1f} minutes de boucle d'entraînement/validation sur le GPU local. Elle part des poids maison de la V2, sans poids externes. Architecture de 13 337 280 paramètres inchangée.

Le mode de continuation est **autorégressif**, le débruitage restant un objectif auxiliaire et un mode distinct pour compléter des trous. Cette amélioration ne doit pas être présentée comme un gain de génération par diffusion.

## Validation et test

Perplexité de validation au premier checkpoint causal (étape {validation[0]['step']}) : {validation[0]['validation']['perplexity']:.3f}. Dernier checkpoint : {validation[-1]['validation']['perplexity']:.3f}. Sélection par NLL de validation ; poids retenus à l'étape {report['best']['step']}.

Sur 512 fenêtres test : **NLL {metric['nll']:.4f} nats, perplexité {metric['perplexity']:.3f}, accuracy prochain token {metric['accuracy']:.2%}**, sur {metric['tokens']:,} cibles. Ce ne sont pas des mesures de cohérence humaine. La V2 n'avait pas été entraînée en causal ; on ne lui attribue pas artificiellement une perplexité comparable.

![Courbe d'apprentissage](learning_curve.png)

Fréquence moyenne de trigrammes de sortie répétés sur les 12 prompts : avant {before_rep:.2%}, après {after_rep:.2%}. Cet indicateur n'évalue ni la grammaire ni la logique du récit. Les algorithmes de génération ont changé et les sorties ont des longueurs différentes.

## Comparaison exhaustive des prompts fixés

Température 0,5, seed 42, au plus 96 nouveaux tokens. Les sorties brutes sont reproduites sans correction, sans choix du meilleur essai. Le cas utilisateur est un diagnostic connu.
'''
    for i,item in enumerate(comparisons["examples"],1):
        text+=f"\n### {i}. {item['prompt'].strip()}\n\n**Avant — diffusion par blocs**\n\n{item['before']['text']}\n\n**Après — causal sans pénalité de répétition**\n\n{item['after_raw']['text']}\n\n**Après — réglages recommandés**\n\n{item['after']['text']}\n"
    text+='\n## Limites\n\nPetit modèle anglais, corpus synthétique, contexte limité et possibles erreurs de cohérence. Pas de revue humaine aveugle, pas de preuve de perfection. Les métriques de débruitage après adaptation sont dans report.json pour vérifier le compromis entre les deux tâches.\n'
    (out/"RESULTATS.md").write_text(text,encoding="utf-8")
    print(json.dumps(metric,indent=2))


if __name__=="__main__":main()
