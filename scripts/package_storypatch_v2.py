"""Package the validation-selected checkpoint with matching evidence; no upload."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffuthink.v2.data import digest
from diffuthink.v2.model import Denoiser


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='artifacts/StoryPatch-v2')
    args = p.parse_args()
    reports = ROOT/'reports'/'storypatch-v2'
    selected = json.loads((reports/'selection.json').read_text())
    source = ROOT/selected['model']
    output = ROOT/args.output
    if output.exists():
        raise ValueError('Choose a new output directory')
    if digest(source/'model.safetensors') != selected['weights_sha256']:
        raise ValueError('Selected checkpoint changed after evaluation')
    entry = next(r for r in selected['validation'] if r['model'] == selected['model'])
    evaluation = json.loads((reports/entry['report']).read_text(encoding='utf-8'))
    if evaluation['weights_sha256'] != selected['weights_sha256']:
        raise ValueError('Test evidence does not match checkpoint')
    model = Denoiser.from_pretrained(source)
    info = json.loads((source/'training_info.json').read_text())
    output.mkdir(parents=True)
    for name in ('config.json','model.safetensors','tokenizer.json','training_info.json'):
        shutil.copy2(source/name, output/name)
    shutil.copy2(ROOT/'LICENSE', output/'LICENSE')
    shutil.copytree(ROOT/'diffuthink', output/'diffuthink', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(reports/entry['report'], output/'evaluation.json')
    shutil.copy2(reports/'selection.json', output/'selection.json')
    for name in ('baseline.json','adapted13m.json','scratch58m.json','adapted13m_cpu.json','pilot_notes.json','environment.json'):
        shutil.copy2(reports/name, output/name)
    shutil.copy2(reports/'RESULTATS.md', output/'RESULTATS.md')
    shutil.copy2(reports/'learning_curves.png', output/'learning_curves.png')
    shutil.copy2(ROOT/'docs'/'STORYPATCH_V2.md', output/'PROTOCOL.md')
    # Keep local report links valid in the standalone bundle.
    result_text = (output/'RESULTATS.md').read_text(encoding='utf-8')
    (output/'RESULTATS.md').write_text(result_text.replace('../../docs/STORYPATCH_V2.md', 'PROTOCOL.md'), encoding='utf-8')
    (output/'requirements.txt').write_text('torch>=2.2\ntokenizers>=0.22,<0.23\nsafetensors>=0.4\nrequests>=2.31\nnumpy>=1.26\n')
    params = sum(p.numel() for p in model.parameters())
    edit_manifest = info.get('editing_manifest')
    editing_description = (
        f"Preparation retains {edit_manifest['splits']['train']['windows']:,} filtered training windows "
        'and up to eight whole-word mask positions per window. These are reconstruction targets, not '
        'human-labeled corrections. This phase alternates causal and span reconstruction updates.'
        if edit_manifest else 'The baseline was retained; the candidate editing updates did not meet validation gates.'
    )
    card = f'''---
language:
- en
library_name: pytorch
pipeline_tag: fill-mask
tags:
- diffusion
- from-scratch
- text-editing
license: apache-2.0
datasets:
- roneneldan/TinyStories
---
# StoryPatch v2

An inspectable English story editor powered by {params:,} parameters. All weight
lineage originates from random initialization in this project; no external
pretrained model or generation API is used. Custom PyTorch checkpoint, not
compatible with Transformers AutoModel or the generic fill-mask pipeline.

## Use locally

Install `requirements.txt`, then run this command from the downloaded directory:

```sh
python -m diffuthink.v2 serve --model . --device cpu --port 7861
```

Open http://127.0.0.1:7861/ and select 1–12 words in a simple English story.
Six denoising attempts use both sides of the selection. Causal local NLL ranks
the candidates. All characters outside the selection are preserved by composition.

```python
from diffuthink.v2.inference import load
from diffuthink.v2.editor import rewrite_span
model, tokenizer = load('.', 'cpu')
text = 'Lily lost her favorite toy. She felt happy. Tears ran down her face.'
start = text.index('happy')
result = rewrite_span(model, tokenizer, text, start, start + len('happy'))
for candidate in result['candidates']:
    print(candidate['replacement'])
```

## Measured evidence

- 400 held-out whole-word spans: NLL {evaluation['span']['nll']:.3f}; token accuracy {evaluation['span']['token_accuracy']:.1%}.
- 36 authored diagnostic cases: first accepted candidate {evaluation['editor']['top1']:.1%}; top-three coverage {evaluation['editor']['top3']:.1%}.
- Causal perplexity on the same 400 held-out windows: {evaluation['causal']['perplexity']:.3f}.

These reference metrics can reject valid alternative answers. The authored probes
are not a general grammar or coherence benchmark. No blind human evaluation is
claimed. Full samples, settings and weight fingerprints are in `evaluation.json`.
Checkpoint choice uses validation only; see `selection.json` and `RESULTATS.md`.
The selected 13M adaptation improves reference reconstruction modestly. First-choice
accepted probes change from 25/36 to 26/36, top-three coverage falls from 29/36 to
28/36, and causal perplexity worsens slightly. The larger 58M experiment was not
selected. This is not a demonstrated general improvement in semantic coherence.

## Data and training

Context: {model.config.max_length} tokens. Own 8,192-token BPE. Source corpus:
500,000 synthetic TinyStories training documents, with separate validation/test
documents. {editing_description} Learned positions and RoPE are both retained.

Exact lineage, token counts, checkpoint step and data hashes: `training_info.json`.
Source dataset: https://huggingface.co/datasets/roneneldan/TinyStories,
revision `{info['manifest']['revision']}`. Dataset text and derived evaluation
snippets retain CDLA-Sharing-1.0 terms; code and weights use Apache-2.0.

## Limits

Small English story model, not an instruction-following assistant. Suggestions can
be ungrammatical, contradictory, repetitive or worse than the original. Local
likelihood is not a semantic verifier. No learned reranker, general reasoning or
scientific novelty claim. Development was substantially AI-assisted.

This folder is a local release candidate; packaging does not upload it.
'''
    (output/'README.md').write_text(card, encoding='utf-8')
    checks = {f.relative_to(output).as_posix(): digest(f) for f in output.rglob('*') if f.is_file()}
    (output/'SHA256SUMS.json').write_text(json.dumps(checks, indent=2))
    print(output)


if __name__ == '__main__':
    main()
