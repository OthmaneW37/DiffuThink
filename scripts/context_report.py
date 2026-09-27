"""Create the evidence bundle for the expanded story-context experiment."""
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    run=Path("runs/stories-512");out=Path("reports/context512")
    report=json.loads((run/"report.json").read_text(encoding="utf-8"))
    comparison=json.loads((out/"comparison.json").read_text(encoding="utf-8"))
    initial=json.loads((run/"initial_validation.json").read_text(encoding="utf-8"))
    rows=[json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
    validation=[r for r in rows if "validation" in r]
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    ax.plot([0]+[r["step"] for r in validation],[initial["perplexity"]]+[r["validation"]["perplexity"] for r in validation],"o-",color="#147d64")
    ax.set(xlabel="Additional optimizer updates",ylabel="Validation perplexity (512-token windows)",title="DiffuThink Story 512 · expanded corpus and context")
    ax.grid(alpha=.2)
    fig.savefig(out/"learning_curve.png",dpi=180);fig.savefig(out/"learning_curve.svg");plt.close(fig)
    for name in ("report.json","settings.json","metrics.jsonl","initial_validation.json"):
        shutil.copy2(run/name,out/name)
    if (run/"execution_notes.json").exists():
        shutil.copy2(run/"execution_notes.json",out/"execution_notes.json")
    manifest=report["best"]["manifest"];train=manifest["splits"]["train"]
    before=comparison["likelihood"]["before"];after=comparison["likelihood"]["after"]
    if before["tokens"] != after["tokens"]:
        raise ValueError("Likelihood target counts differ")
    text=f'''# DiffuThink Story 512 — expérience mesurée

Phase supplémentaire : **{report['settings']['steps']:,} mises à jour**, {report['seen_tokens']:,} tokens non-PAD présentés, {report['seconds']/60:.1f} minutes de boucle GPU et validation. Meilleur checkpoint sélectionné à l'étape {report['best']['step']}. Aucun poids externe.

Corpus : {train['documents']:,} récits synthétiques TinyStories contenant {train['text_tokens']:,} tokens de texte, avant les présentations répétées pendant l'entraînement. {train['complete_story_windows']/train['documents']:.2%} des récits tiennent entièrement dans une fenêtre. Les récits trop longs restent découpés ; EOS apparaît uniquement sur le dernier segment. Contexte 512 au lieu de 192 ; vocabulaire BPE identique ; mêmes documents de validation/test, nouvelle mise en fenêtres.

L'expérience combine corpus élargi, contexte étendu et davantage de calcul. Elle ne permet pas d'attribuer le gain à un seul de ces facteurs. L'objectif reste 90 % de lots causaux et 10 % de débruitage. La continuation est autorégressive.

## Comparaison contrôlée de vraisemblance

Les deux modèles sont évalués sur les **mêmes {comparison['protocol']['common_test_windows']} fenêtres test de 192 tokens**, avec exactement les mêmes cibles. Cette mesure isole une comparaison équitable des poids ; elle ne mesure pas l'avantage d'utiliser un contexte plus long en inférence.

| Mesure | Avant | Après |
|---|---:|---:|
| NLL, nats | {before['nll']:.4f} | {after['nll']:.4f} |
| Perplexité | {before['perplexity']:.3f} | {after['perplexity']:.3f} |
| Accuracy prochain token | {before['accuracy']:.2%} | {after['accuracy']:.2%} |
| Cibles | {before['tokens']:,} | {after['tokens']:,} |

La perplexité sur les nouvelles fenêtres de 512 tokens est publiée séparément dans report.json ; elle n'est pas directement comparable à l'ancien score sur 512 fenêtres de 192 tokens.

![Courbe d'apprentissage](learning_curve.png)

## Continuations : toutes les sorties

24 prompts × 2 seeds. Le premier prompt est le cas utilisateur déjà connu ; les autres ont été fixés avant inspection des nouveaux poids. Température 0,5 ; top-p 0,9 ; pénalité 1,12 ; interdiction des 4-grammes répétés ; plafond strict de 160 nouveaux tokens. Aucun choix du meilleur essai, aucune correction grammaticale, aucune marge de fin de phrase dans cette comparaison.

| Indicateur descriptif | Avant | Après |
|---|---:|---:|
'''
    labels={"mean_repeated_trigram_fraction":"Fraction moyenne de trigrammes répétés", "eos_rate":"Arrêt par EOS", "punctuated_ending_rate":"Sortie finissant par une ponctuation terminale"}
    for key,label in labels.items():
        text+=f"| {label} | {comparison['summary']['before'][key]:.2%} | {comparison['summary']['after'][key]:.2%} |\n"
    text+='\nCes indicateurs ne mesurent pas la cohérence sémantique. Une terminaison EOS ne prouve pas que le récit est satisfaisant. Pas de revue humaine aveugle ni de score de cohérence automatisé revendiqué.\n'
    for i,row in enumerate(comparison["examples"],1):
        text+=f"\n### {i}. Seed {row['seed']} — {row['prompt'].strip()}\n"
        for key,label in (("before","Avant"),("after","Après")):
            sample=row[key]
            text+=f"\n**{label}** — {sample['generated_tokens']} tokens, arrêt `{sample['stop_reason']}`\n\n{sample['text']}\n"
    text+='\n## Limites\n\nAnglais simple, corpus synthétique, possibles confusions de personnages, contradictions et répétitions. Les prompts ciblent certains défauts connus : ils ne constituent pas une certification générale de qualité. Le checkpoint est sélectionné sur la validation, pas sur ces sorties test. Le corpus est dédupliqué à l’identique après normalisation ; les quasi-doublons ne sont pas garantis absents.\n'
    (out/"RESULTATS.md").write_text(text,encoding="utf-8")
    print(json.dumps({"before":before,"after":after,"summary":comparison["summary"]},indent=2))


if __name__=="__main__":main()
