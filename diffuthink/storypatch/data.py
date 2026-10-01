"""Whole-word editing examples from existing, fingerprinted document splits.

Targets are original text, not human-verified corrections. No external teacher.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from diffuthink.v2.data import digest, load_arrays
from diffuthink.v2.model import BOS, EOS, PAD

WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")


def word_spans(text, tokenizer, rng, count=8):
    """Token ranges enclosing complete words, with visible left and right context."""
    encoded = tokenizer.encode(text)
    words = list(WORD.finditer(text))
    spans = []
    for _ in range(count * 20):
        if len(words) < 4:
            break
        start = int(rng.integers(1, len(words)-1))
        length = int(rng.integers(1, min(12, len(words)-1-start)+1))
        a, b = words[start].start(), words[start+length-1].end()
        positions = [i for i, (x, y) in enumerate(encoded.offsets) if y > a and x < b]
        if not positions or not 1 <= len(positions) <= 16:
            continue
        first, last = positions[0], positions[-1]
        # Token edges can contain whitespace, but may not consume visible punctuation/letters.
        x, y = encoded.offsets[first][0], encoded.offsets[last][1]
        if text[x:a].strip() or text[b:y].strip():
            continue
        span = (first+1, last+2)  # BOS offset; end exclusive
        if span not in spans:
            spans.append(span)
        if len(spans) == count:
            break
    return encoded.ids, spans


def quality(text):
    words = text.lower().split()
    if len(words) < 8 or '\ufffd' in text or any(ord(c) < 32 and not c.isspace() for c in text):
        return False
    grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    return not grams or 1-len(set(grams))/len(grams) <= .20


def normalized(text):
    return ' '.join(text.lower().split())


def prepare(source, output, seed=4801):
    out = Path(output)
    if out.exists():
        raise ValueError('Choose a new output directory; frozen datasets are never overwritten')
    arrays, source_manifest = load_arrays(source)
    tokenizer = Tokenizer.from_file(str(Path(source)/'tokenizer.json'))
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True)
    tokenizer.save(str(out/'tokenizer.json'))
    manifest = {'version': 1, 'source': str(source), 'source_manifest': source_manifest,
                'seed': seed, 'spans_per_window': 8, 'max_words': 12, 'max_span_tokens': 16,
                'tokenizer_sha256': digest(out/'tokenizer.json'), 'splits': {},
                'supervision': 'Self-supervised reconstruction; original spans are not verified corrections',
                'deduplication': 'Normalized exact window hashes across splits; no general near-deduplication claim'}
    seen = set()
    # Reserve held-out windows first; reject any normalized train matches.
    for split in ('test', 'validation', 'train'):
        rows, ranges, examples = [], [], []
        stats = {'rejected_quality': 0, 'rejected_duplicate': 0, 'rejected_alignment': 0}
        raw = out/f'{split}.raw'
        with raw.open('wb') as stream:
            for index, row in enumerate(arrays[split]):
                ids = [int(x) for x in row if x not in (PAD, BOS, EOS)]
                text = tokenizer.decode(ids)
                if not quality(text):
                    stats['rejected_quality'] += 1
                    continue
                key = hashlib.sha256(normalized(text).encode()).hexdigest()
                if key in seen:
                    stats['rejected_duplicate'] += 1
                    continue
                encoded, spans = word_spans(text, tokenizer, rng)
                if encoded != ids or not spans:
                    stats['rejected_alignment'] += 1
                    continue
                seen.add(key)
                # Keep original sequence and EOS labels exactly, including long-story segments.
                np.asarray(row, dtype=np.uint16).tofile(stream)
                rows.append(index)
                ranges.append(spans + [spans[-1]]*(8-len(spans)))
                if split != 'train' and len(examples) < 400:
                    a, b = spans[0]
                    examples.append({'id': f'{split}-{index}', 'source_window': index,
                                     'start_token': a, 'end_token': b,
                                     'reference': tokenizer.decode(ids[a-1:b-1]), 'text': text})
                if index and index % 25000 == 0:
                    print(json.dumps({'split': split, 'processed': index, 'accepted': len(rows)}), flush=True)
        shape = (len(rows), source_manifest['max_length'])
        if not rows:
            raise ValueError(f'No valid {split} examples')
        mapped = np.memmap(raw, dtype=np.uint16, mode='r', shape=shape)
        np.save(out/f'{split}.npy', mapped)
        del mapped
        raw.unlink()
        np.save(out/f'{split}_spans.npy', np.asarray(ranges, dtype=np.uint16))
        np.save(out/f'{split}_source_rows.npy', np.asarray(rows, dtype=np.int32))
        if examples:
            (out/f'{split}_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
        stats.update(windows=len(rows), candidate_spans=len(rows)*8)
        stats['files'] = {f'{split}{suffix}.npy': digest(out/f'{split}{suffix}.npy')
                          for suffix in ('', '_spans', '_source_rows')}
        manifest['splits'][split] = stats
        print(json.dumps({'split': split, **stats}), flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if digest(directory/'tokenizer.json') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    arrays = {}
    for split, info in manifest['splits'].items():
        for name, expected in info['files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Data checksum mismatch: {name}')
        arrays[split] = (np.load(directory/f'{split}.npy', mmap_mode='r'),
                         np.load(directory/f'{split}_spans.npy', mmap_mode='r'))
    return arrays, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/processed/stories-512')
    parser.add_argument('--output', default='data/processed/storypatch-v2')
    args = parser.parse_args()
    prepare(args.source, args.output)
