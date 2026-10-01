"""Evaluate fixed reconstruction masks and the actual interactive editor."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from diffuthink.v2.data import digest
from diffuthink.v2.editor import rewrite_span
from diffuthink.v2.inference import load as load_model, denoise
from diffuthink.v2.hybrid import causal_evaluate
from diffuthink.v2.model import PAD, MASK
from .data import load
from .train import span_evaluate


def normalize(text):
    return ' '.join(text.casefold().strip(' .!?\n\t').split())


def run(args):
    torch.set_num_threads(4)
    arrays, manifest = load(args.data)
    model, tokenizer = load_model(args.model, args.device)
    if digest(Path(args.model)/'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Benchmark requires the identical frozen tokenizer')
    precision = 'bf16' if args.device == 'cuda' else 'fp32'
    report = {'model': args.model, 'weights_sha256': digest(Path(args.model)/'model.safetensors'),
              'device': args.device, 'precision': precision, 'torch_version': str(torch.__version__),
              'data_manifest': manifest, 'split': args.split, 'editor_attempts': 6,
              'challenge_sha256': digest(args.challenges), 'seed': 42,
              'limitations': 'Exact reference metrics miss valid alternatives. Authored challenges are diagnostic, not a general grammar or coherence score.'}
    report['span'] = span_evaluate(model, arrays[args.split], 400, precision)
    report['causal'] = causal_evaluate(model, arrays[args.split][0], 400, precision=precision)
    print(json.dumps({'span': report['span'], 'causal': report['causal']}), flush=True)
    samples = []
    array, spans = arrays[args.split]
    for index in range(min(64, len(array))):
        clean = [int(t) for t in array[index] if t != PAD]
        a, b = map(int, spans[index, 0])
        masked = clean[:a]+[MASK]*(b-a)+clean[b:]
        result = denoise(model, tokenizer, masked, steps=12, temperature=0, seed=42, precision=precision, allow_eos=False)
        pred = result['tokens'][a:b]
        samples.append({'index': index, 'reference': tokenizer.decode(clean[a:b]),
                        'prediction': tokenizer.decode(pred), 'exact': pred == clean[a:b],
                        'context_preserved': result['tokens'][:a] == clean[:a] and result['tokens'][b:] == clean[b:]})
    report['iterative_reconstruction'] = {'samples': samples, 'examples': len(samples),
        'exact_match': sum(s['exact'] for s in samples)/len(samples),
        'context_preservation': sum(s['context_preserved'] for s in samples)/len(samples),
        'oracle_length': True, 'note': 'Reference span length supplied; this is not the editor candidate metric.'}
    challenges = json.loads(Path(args.challenges).read_text())['cases']
    outputs = []
    for case in challenges:
        result = rewrite_span(model, tokenizer, case['text'], case['start'], case['end'], precision=precision)
        accepted = {normalize(x) for x in case['accepted']}
        candidates = result['candidates']
        outputs.append({**case, 'top1': bool(candidates and normalize(candidates[0]['replacement']) in accepted),
                        'top3': any(normalize(c['replacement']) in accepted for c in candidates[:3]),
                        'context_preserved': all(c['text'] == result['prefix']+c['replacement']+result['suffix'] for c in candidates),
                        'empty': not candidates, 'latency_ms': result['latency_ms'],
                        'candidates': [{k: v for k, v in c.items() if k != 'trace'} for c in candidates]})
        print(json.dumps({'case': case['id'], 'top1': outputs[-1]['top1'], 'top': candidates[0]['replacement'] if candidates else None}), flush=True)
    def summarize(rows):
        return {key: sum(row[key] for row in rows)/len(rows) for key in ('top1','top3','context_preserved','empty')}
    report['editor'] = {'examples': len(outputs), **summarize(outputs),
                        'median_latency_ms': float(np.median([r['latency_ms'] for r in outputs])),
                        'by_category': {c: summarize([r for r in outputs if r['category'] == c]) for c in sorted({r['category'] for r in outputs})},
                        'samples': outputs}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report['editor'].items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', default='data/processed/storypatch-v2')
    p.add_argument('--challenges', default='reports/storypatch-v2/challenges.json')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--split', choices=['validation','test'], default='test')
    run(p.parse_args())
