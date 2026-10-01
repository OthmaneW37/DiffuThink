"""Create a standalone model+ranker release from fingerprinted local evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffuthink.v2.data import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/StoryPatch-ranked');args=p.parse_args()
    reports=ROOT/'reports/storypatch-ranker-v2'
    selected=json.loads((reports/'selection.json').read_text())
    if not selected['enabled']:raise ValueError('Ranker did not pass validation gates')
    source=ROOT/selected['generator'];ranker=ROOT/selected['ranker']
    for directory,key in [(source,'generator_weights_sha256'),(ranker,'ranker_weights_sha256')]:
        if digest(directory/'model.safetensors')!=selected[key]:raise ValueError('Evaluated checkpoint changed')
    info=json.loads((ranker/'ranker_info.json').read_text())
    if info['calibration']!=selected['calibration']:raise ValueError('Calibration changed')
    out=ROOT/args.output
    if out.exists():raise ValueError('Choose a fresh bundle output')
    out.mkdir(parents=True)
    for name in ['config.json','model.safetensors','tokenizer.json','training_info.json']:shutil.copy2(source/name,out/name)
    shutil.copytree(ranker,out/'ranker')
    shutil.copytree(ROOT/'diffuthink',out/'diffuthink',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for version in ['v1','v2']:shutil.copytree(ROOT/f'reports/storypatch-ranker-{version}',out/f'reports/storypatch-ranker-{version}')
    (out/'docs').mkdir();shutil.copy2(ROOT/'docs/STORYPATCH_RANKER.md',out/'docs/STORYPATCH_RANKER.md')
    shutil.copy2(ROOT/'LICENSE',out/'LICENSE');shutil.copy2(reports/'evaluation.json',out/'evaluation.json')
    shutil.copy2(reports/'selection.json',out/'selection.json')
    (out/'requirements.txt').write_text('torch>=2.2\ntokenizers>=0.22,<0.23\nsafetensors>=0.4\nrequests>=2.31\nnumpy>=1.26\n',encoding='utf-8')
    result=json.loads((reports/'evaluation.json').read_text())['end_to_end'];nll=result['nll_keep_or_replace'];ranked=result['learned_keep_or_replace']
    (out/'README.md').write_text(f'''---
language:
- en
library_name: pytorch
pipeline_tag: fill-mask
license: apache-2.0
tags:
- from-scratch
- text-editing
- diffusion
- reranking
datasets:
- roneneldan/TinyStories
---
# StoryPatch: generate, compare, or keep

An inspectable editor for short English stories. Select 1–12 whole words,
compare local alternatives, keep the original or apply and undo a change.
All surrounding characters remain exactly unchanged by string composition.
No external pretrained model, remote generation API or instruction model is used.

The 13,439,680-parameter generator has weight lineage originating from random
initialization in DiffuThink, trained on synthetic TinyStories and adapted for
whole-word reconstruction. The independent 3,528,769-parameter ranker starts
from random weights and learns contextual candidate preferences on procedural
contrasts. It is a custom PyTorch architecture, not a Transformers AutoModel
checkpoint or a generic fill-mask pipeline. The scores are not correctness
probabilities. A validation-calibrated threshold can recommend keeping the original.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m diffuthink.v2 serve --model . --ranker ranker --device cpu --port 7862
```

Open http://127.0.0.1:7862/. Six denoising attempts see both sides of a span.
The generator proposes; the ranker compares. Replacements remain manual.
Local causal NLL is the UI default. Select "Correction ciblée · expérimental"
to try the learned ranker. The HTTP API uses `ranking: "learned"` to opt in.
The generation laboratory is available at `/lab`.

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

## Measured local evidence

600 procedural editing inputs across 300 paired contexts, half altered and
half already correct. Exact bounded synthetic answers, **no human evaluation**.
The generator's candidate pool is identical for both ranking methods.

| Synthetic measure | NLL + keep | Learned ranker + keep |
|---|---:|---:|
| Repair altered input | {nll['error_corrected']:.1%} | {ranked['error_corrected']:.1%} |
| Preserve clean input exactly | {nll['clean_preserved']:.1%} | {ranked['clean_preserved']:.1%} |
| Balanced success | {nll['balanced_success']:.1%} | {ranked['balanced_success']:.1%} |

Candidate coverage on altered inputs is {result['error_candidate_coverage']:.1%}.
On the 36 legacy authored probes, top1 acceptance regresses from 72.2% (NLL)
to 52.8% (learned). This is why the new ranking is an explicit experimental mode.
This bounds repair achievable by reordering these candidates. Tests use held-out
template combinations in six familiar categories, not free-form user stories.
Do not interpret these figures as general grammar, reasoning or factual accuracy.
All outputs, hashes, failures and category scores are in
[the results](reports/storypatch-ranker-v2/RESULTATS.md). Selection uses validation
only; see `selection.json`. A blind review CSV is provided with unfilled judgments.

## Training and limits

Own 8,192-token BPE; both models have 512-token contexts. Ranker training uses
12,000 procedural contexts, plus 2,699 generator proposals mined from 1,200
training contexts only. Labels use a closed accepted-answer list and can reject
valid alternatives. No human corrections or external teacher are claimed.
Ranker checkpoint: step {info['step']}, chosen by validation BCE.
Weights, tokenization and evaluation are linked by SHA-256.

The first ranker trial failed to pass useful replacement thresholds and is
retained for inspection. The final model can still prefer incorrect or
ungrammatical suggestions, especially outside this narrow training domain.
Original preservation is always available. Texts are not saved to application
files or sent to another model; hosted inference processes them on its server.
The single-process demo is intended for exploration, not high concurrency.

[French protocol](docs/STORYPATCH_RANKER.md). Generator lineage and TinyStories
source revision are in `training_info.json`. Code, weights and procedural corpus
use Apache-2.0; original TinyStories text retains CDLA-Sharing-1.0 terms.
Development and documentation were substantially AI-assisted. No scientific
novelty claim. This folder is a local release candidate; packaging does not upload it.
''',encoding='utf-8')
    checks={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(str(out))


if __name__=='__main__':main()
