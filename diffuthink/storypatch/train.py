"""Resumable mixed causal / whole-word-span training, without external weights."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from diffuthink.v2.model import Config, Denoiser, PAD, BOS, MASK
from diffuthink.v2.train import amp, atomic_save
from diffuthink.v2.hybrid import causal_evaluate
from .data import load


def span_batch(clean, ranges, rng=None, partial=False):
    positions = torch.arange(clean.shape[1], device=clean.device)[None]
    selected = (positions >= ranges[:, :1]) & (positions < ranges[:, 1:])
    if partial:
        # Some words have already been committed by iterative inference.
        keep = torch.rand(clean.shape, device=clean.device, generator=rng) < .5
        keep.scatter_(1, ranges[:, :1], True)
        selected &= keep
    if not selected.any(-1).all() or (selected & (clean.eq(PAD) | clean.eq(BOS))).any():
        raise ValueError('Invalid supervised spans')
    noise = selected.sum(-1).float() / (clean.ne(PAD) & clean.ne(BOS)).sum(-1)
    return clean.masked_fill(selected, MASK), selected, noise


@torch.inference_mode()
def span_evaluate(model, pair, limit=400, precision='bf16', batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    total = correct = exact = examples = 0
    loss = 0.
    array, spans = pair
    for start in range(0, min(limit, len(array)), batch_size):
        stop = min(start+batch_size, limit, len(array))
        clean = torch.tensor(np.array(array[start:stop]), dtype=torch.long, device=device)
        clean = clean[:, :int(clean.ne(PAD).sum(-1).max())]
        ranges = torch.tensor(np.array(spans[start:stop, 0]), dtype=torch.long, device=device)
        noisy, selected, noise = span_batch(clean, ranges)
        with amp(device.type, precision):
            logits = model(noisy, noise, selected)
        target = clean[selected]
        matched = logits.argmax(-1).eq(target)
        loss += F.cross_entropy(logits.float(), target, reduction='sum').item()
        correct += int(matched.sum())
        offset = 0
        for count in selected.sum(-1).tolist():
            exact += int(matched[offset:offset+count].all())
            offset += count
        total += target.numel()
        examples += len(clean)
    return {'nll': loss/total, 'token_accuracy': correct/total, 'parallel_exact_match': exact/examples,
            'targets': total, 'examples': examples,
            'meaning': 'Exact reference reconstruction; valid alternative wording can score zero'}


def train(args):
    if min(args.steps, args.batch_size, args.eval_every, args.eval_samples, args.warmup) < 1 or args.steps <= args.warmup:
        raise ValueError('Positive counts and steps > warmup required')
    if args.span_every < 0:
        raise ValueError('span-every must be nonnegative')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    arrays, manifest = load(args.data)
    source_manifest = manifest['source_manifest']
    out = Path(args.output)
    if (out/'last.pt').exists() and not args.resume:
        raise ValueError('Existing run; use --resume')
    out.mkdir(parents=True, exist_ok=True)
    config = Config(vocab_size=source_manifest['vocab_size'], width=args.width, heads=8,
                    layers=args.layers, max_length=source_manifest['max_length'])
    lineage = None
    if args.source:
        lineage = json.loads((Path(args.source)/'training_info.json').read_text())
        if lineage.get('pretrained_weights') is not False:
            raise ValueError('Only our own from-scratch lineage is accepted')
        if lineage['manifest']['tokenizer_sha256'] != manifest['tokenizer_sha256']:
            raise ValueError('Cannot change vocabulary while reusing weights')
        model = Denoiser.from_pretrained(args.source, args.device)
        config = model.config
        if config.max_length != source_manifest['max_length']:
            raise ValueError('Context mismatch')
    else:
        model = Denoiser(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.1)
    sampling = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=args.device).manual_seed(args.seed+1)
    settings = {k: v for k, v in vars(args).items() if k not in ('resume', 'stop_after')}
    step = seen = supervised = 0
    elapsed, best = 0., float('inf')
    if args.resume:
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        restored_settings = dict(state['settings'])
        restored_settings.setdefault('span_every', 2)
        restored_settings.setdefault('selection_mode', 'mixed')
        if restored_settings != settings or state['manifest'] != manifest:
            raise ValueError('Recovery settings/data mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        sampling.set_state(state['sampling_rng'])
        noise_rng.set_state(state['noise_rng'])
        step, seen, supervised, elapsed, best = (state[k] for k in ('step','seen_tokens','supervised_tokens','seconds','best'))
    (out/'settings.json').write_text(json.dumps(settings, indent=2))
    (out/'config.json').write_text(json.dumps(asdict(config), indent=2))
    # Uniform random anchors + wrapped offsets preserve marginal uniform sampling.
    train_rows, spans = arrays['train']
    lengths = np.count_nonzero(train_rows, axis=1)
    order = np.argsort(lengths, kind='stable')
    started = time.perf_counter()
    end = min(args.steps, args.stop_after) if args.stop_after else args.steps
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()), 'from_step': step,
                      'to_step': end, 'source': args.source, 'device': args.device}), flush=True)
    while step < end:
        step += 1
        model.train()
        anchor = int(torch.randint(len(order), (1,), generator=sampling))
        offsets = torch.randint(min(2048, len(order)), (args.batch_size,), generator=sampling).numpy()
        indices = order[(anchor+offsets) % len(order)]
        clean = torch.tensor(np.array(train_rows[indices, :int(lengths[indices].max())]), dtype=torch.long, device=args.device)
        lr = args.lr * min(1., step/args.warmup) * (.1+.9*.5*(1+math.cos(math.pi*max(0, step-args.warmup)/(args.steps-args.warmup))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        editing = args.span_every > 0 and step % args.span_every == 0
        with amp(args.device, args.precision):
            if editing:
                choices = torch.randint(8, (args.batch_size,), generator=sampling).numpy()
                ranges = torch.tensor(np.array(spans[indices, choices]), dtype=torch.long, device=args.device)
                noisy, selected, noise = span_batch(clean, ranges, noise_rng, partial=(step//args.span_every) % 2 == 0)
                logits = model(noisy, noise, selected)
                targets = clean[selected]
            else:
                targets = clean[:, 1:]
                selected = targets.ne(PAD)
                logits = model(clean[:, :-1], torch.zeros(len(clean), device=args.device), selected, causal=True, right_padded=True)
                targets = targets[selected]
            loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        seen += int(clean.ne(PAD).sum())
        supervised += targets.numel()
        if step == 1 or step % 100 in (0, 1):
            row = {'step': step, 'objective': 'span' if editing else 'causal', 'loss': loss.item(),
                   'lr': lr, 'seen_tokens': seen, 'supervised_tokens': supervised,
                   'seconds': elapsed+time.perf_counter()-started}
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == end:
            metric = span_evaluate(model, arrays['validation'], args.eval_samples, args.precision)
            causal = causal_evaluate(model, arrays['validation'][0], args.eval_samples, precision=args.precision)
            # Fixed before training, not chosen by looking at test samples.
            score = causal['nll'] if args.selection_mode == 'causal' else metric['nll'] + .25*causal['nll']
            row = {'step': step, 'validation_span': metric, 'validation_causal': causal, 'selection_score': score}
            if score < best:
                best = score
                model.save_pretrained(out/'best')
                shutil.copy2(Path(args.data)/'tokenizer.json', out/'best'/'tokenizer.json')
                info = {'step': step, 'parameters': sum(p.numel() for p in model.parameters()),
                        'seen_tokens': seen, 'supervised_tokens': supervised, 'pretrained_weights': False,
                        'manifest': source_manifest, 'editing_manifest': manifest, 'previous_phase': lineage,
                        'continuation_mode': 'autoregressive',
                        'objective': f'Span update every {args.span_every} steps (0 = causal only), alternating full and partially revealed word spans; all other updates causal',
                        'selection_mode': args.selection_mode,
                        'selection_score': score, 'validation': row}
                (out/'best'/'training_info.json').write_text(json.dumps(info, indent=2))
            atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'sampling_rng': sampling.get_state(), 'noise_rng': noise_rng.get_state(),
                         'step': step, 'seen_tokens': seen, 'supervised_tokens': supervised,
                         'seconds': elapsed+time.perf_counter()-started, 'best': best,
                         'settings': settings, 'manifest': manifest}, out/'last.pt')
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
    (out/'status.json').write_text(json.dumps({'completed': step == args.steps, 'step': step,
        'planned_steps': args.steps, 'seconds': elapsed+time.perf_counter()-started,
        'best_selection_score': best}, indent=2))
    print(f'Saved {out}; completed={step == args.steps}', flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed/storypatch-v2')
    p.add_argument('--output', default='runs/storypatch-58m')
    p.add_argument('--source', help='Optional continuation of our own smaller scratch-trained model')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    for key, value in [('width',512), ('layers',12), ('steps',12000), ('batch-size',24),
                       ('warmup',400), ('eval-every',500), ('eval-samples',256), ('seed',48)]:
        p.add_argument('--'+key, type=int, default=value)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--span-every', type=int, default=2, help='0 = causal only; otherwise one span update every N steps')
    p.add_argument('--selection-mode', choices=['mixed','causal'], default='mixed')
    p.add_argument('--stop-after', type=int, help='Save recoverable state early without changing the LR schedule')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
