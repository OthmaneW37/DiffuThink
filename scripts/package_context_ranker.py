"""Package the measured context ranker, generator, source and attribution."""
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffuthink.v2.data import digest


def main():
    source=ROOT/'runs/storypatch-13m/best';ranker=ROOT/'runs/storypatch-context-ranker-v3/best'
    reports=ROOT/'reports/storypatch-context-v3';out=ROOT/'artifacts/StoryPatch-context-v3'
    report=json.loads((reports/'evaluation.json').read_text());info=json.loads((ranker/'ranker_info.json').read_text())
    if digest(source/'model.safetensors')!=report['binding']['generator_sha256']:raise ValueError('Generator changed')
    if digest(ranker/'model.safetensors')!=report['binding']['ranker_sha256']:raise ValueError('Ranker changed')
    for key in ['calibration','blending']:
        if info[key]!=report['selection'][key]:raise ValueError('Inference policy changed')
    if out.exists():raise ValueError('Use a fresh package directory')
    out.mkdir(parents=True)
    for name in ['config.json','model.safetensors','tokenizer.json','training_info.json']:shutil.copy2(source/name,out/name)
    shutil.copytree(ranker,out/'ranker')
    shutil.copytree(ROOT/'diffuthink',out/'diffuthink',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(reports,out/'reports/storypatch-context-v3')
    (out/'docs').mkdir();shutil.copy2(ROOT/'docs/STORYPATCH_CONTEXT_V3.md',out/'docs/STORYPATCH_CONTEXT_V3.md')
    for name in ['evaluation.json','selection.json']:shutil.copy2(reports/name,out/name)
    shutil.copy2(ROOT/'LICENSE',out/'LICENSE');shutil.copy2(reports/'DATA_LICENSE.md',out/'DATA_LICENSE.md')
    (out/'requirements.txt').write_text('torch>=2.2\ntokenizers>=0.22,<0.23\nsafetensors>=0.4\nrequests>=2.31\nnumpy>=1.26\n',encoding='utf-8')
    corpus=report['test']['corpus'];proc=report['test']['procedural'];legacy=report['legacy']['summary']
    (out/'README.md').write_text(f'''---
language:
- en
library_name: pytorch
pipeline_tag: fill-mask
license: apache-2.0
tags:
- text-editing
- from-scratch
- diffusion
- reranking
datasets:
- roneneldan/TinyStories
---
# StoryPatch v3 — contextual ranking

Select a few words in a short English story, compare alternatives, keep the
original or apply and undo an edit. Every character outside the selection is
preserved by string composition. Text is not sent to a remote generation API.

## Two models, one project

The generator has 13,439,680 parameters. The context ranker has
{info['parameters']:,} parameters and transfers its backbone from this project's
own generator, then learns a new classification head and span markers.
All weight lineage originates in DiffuThink's random initialization. The ranker
is **not** newly random-initialized end to end. No external pretrained weights
or teacher model are used. Both use the project's own BPE and a 512-token context.
This is custom PyTorch code, not a Transformers AutoModel checkpoint.

## Run

```sh
python -m pip install -r requirements.txt
python -m diffuthink.v2 serve --model . --ranker ranker --device cpu --port 7862
```

Open http://127.0.0.1:7862/. Select **Classement contextuel · expérimental**
to try the new ranker. Causal NLL exploration remains the default. The HTTP
rewrite API accepts `ranking: "learned"`. User edits remain manual.

```python
from diffuthink.v2.inference import load
from diffuthink.v2.editor import rewrite_span
from diffuthink.storypatch.ranker import Reranker
model, tokenizer = load('.', 'cpu')
ranker = Reranker.load('ranker', tokenizer)
text = 'Lily lost her favorite toy. She felt happy. Tears ran down her face.'
a = text.index('happy')
result = rewrite_span(model, tokenizer, text, a, a+5, ranker=ranker)
print(result['recommendation'])
```

## Training and evidence

24,000 training contexts: 12,000 procedural contrasts and 12,000 varied
TinyStories excerpts with mechanical corruptions. TinyStories is synthetic;
there are **no human preference labels**. The own-generator backbone is adapted
for up to 3,200 updates. Selected checkpoint: update {info['step']}, selected
by validation BCE on 1,200 contexts.

The final preference score combines causal NLL and bounded learned compatibility:
`-NLL + {info['blending']['weight']:g} * clip(learned_score, -8, 8)`.
The weight is selected on validation with no top1 loss in either domain.
Separate validation thresholds determine when to recommend preserving the
original. These scores are not probabilities of correctness.

The paired test has 1,200 inputs across 600 contexts. Half the inputs are clean
and half altered. Each method receives identical generated candidates.

| Top1 reference match on altered inputs | NLL | Combined |
|---|---:|---:|
| Corpus excerpts (300 contexts) | {corpus['same_pool_error_top1_nll']:.1%} | {corpus['same_pool_error_top1_ranker']:.1%} |
| Procedural templates (300 contexts) | {proc['same_pool_error_top1_nll']:.1%} | {proc['same_pool_error_top1_ranker']:.1%} |
| Legacy probes (36, reused) | {legacy['nll']:.1%} | {legacy['blended']:.1%} |

Original corpus spans are reconstruction references, not exhaustive lists of
valid edits. Exact reference matching can reject good paraphrases. The
procedural and legacy tests are reused regression diagnostics, not new
independent evidence. No general grammar or reasoning claim is made.
Complete results and failures: [report](reports/storypatch-context-v3/RESULTATS.md).
All candidates, hashes and validation trials are included.

The keep/replace policy is conservative: it recommends an accepted repair on
only {report['test']['all']['learned_keep_or_replace']['error_corrected']:.1%} of altered inputs,
versus {report['test']['all']['nll_keep_or_replace']['error_corrected']:.1%} for NLL alone,
while preserving {report['test']['all']['learned_keep_or_replace']['clean_preserved']:.1%}
of clean inputs. Better ranking does not imply more successful replacement
recommendations. All proposals remain available for manual review.

## Limitations and licensing

This remains a small English story model. It can suggest incorrect, awkward or
contradictory edits, or preserve an original mistake. It cannot recover a
reference that the generator did not propose. Ranker training excerpts are
shorter than the maximum inference context; long-context generalization is not
established. No blind human evaluation has been performed.

Source text and derived TinyStories examples retain CDLA-Sharing-1.0 terms;
see [DATA_LICENSE.md](DATA_LICENSE.md) and the pinned source revision in
`training_info.json`. Code, weights and procedural templates use Apache-2.0.
Development, experiments and documentation were substantially AI-assisted.
Reproduction: [French protocol](docs/STORYPATCH_CONTEXT_V3.md).

This is a local release candidate. Packaging does not publish it to a service.
''',encoding='utf-8')
    checks={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(str(out))


if __name__=='__main__':main()
