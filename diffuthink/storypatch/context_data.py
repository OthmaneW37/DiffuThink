"""Natural TinyStories passages plus procedural contrasts, with source split fidelity.

Natural means corpus excerpts rather than templates. TinyStories itself is
synthetic. Original wording is a reconstruction target, not a human preference.
"""
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from .data import load, quality, WORD, normalized
from .corrections import read
from diffuthink.v2.model import PAD,BOS,EOS


def negatives(fragment,rng):
    words=fragment.split();result=[]
    if len(words)>1:result.append(' '.join(reversed(words)))
    result += [fragment+' '+words[-1], ' '.join(['the']*len(words))]
    choices=['house','tree','happy','sad','she','he','were','was','went','blue','little','because','with','not']
    result.append(' '.join([rng.choice(choices)]+words[1:]))
    if len(fragment)>3:
        i=rng.randrange(1,len(fragment)-1)
        result.append(fragment[:i]+fragment[i+1]+fragment[i]+fragment[i+2:])
    return [v for v in dict.fromkeys(result) if v!=fragment and v.strip()]


def build(source='data/processed/storypatch-v2',procedural='data/processed/storypatch-corrections-mined-v2',
          output='data/processed/storypatch-context-v3',seed=9821):
    out=Path(output)
    if out.exists():raise ValueError('Frozen corpus exists; choose a fresh output')
    arrays,source_manifest=load(source);rng=random.Random(seed)
    tokenizer=Tokenizer.from_file(str(Path(source)/'tokenizer.json'))
    out.mkdir(parents=True);seen=set();prepared={}
    for split,count in [('test',300),('validation',600),('train',12000)]:
        candidates=list(range(len(arrays[split][0])));rng.shuffle(candidates);rows=[]
        for source_index in candidates:
            tokens=arrays[split][0][source_index]
            full=tokenizer.decode([int(t) for t in tokens if t not in (PAD,BOS,EOS)])
            sentences=re.split(r'(?<=[.!?])\s+',full)
            # Complete consecutive sentences when possible; do not truncate in mid-word.
            start=rng.randrange(max(1,len(sentences)-1));text=''
            for sentence in sentences[start:start+4]:
                proposal=(text+' '+sentence).strip()
                if len(tokenizer.encode(proposal).ids)>128:break
                text=proposal
            words=list(WORD.finditer(text))
            if len(words)<12 or not quality(text):continue
            i=rng.randrange(2,len(words)-2);length=rng.choice([1,1,2,3])
            j=min(len(words)-2,i+length-1);a,b=words[i].start(),words[j].end()
            fragment=text[a:b];bad=negatives(fragment,rng)
            if not bad:continue
            key=hashlib.sha256(normalized(text).encode()).hexdigest()
            if key in seen:continue
            seen.add(key)
            rows.append({'id':f'{split}-corpus-{source_index}','category':'corpus','source_window':source_index,
                'context_sha256':key,'prefix':text[:a],'suffix':text[b:],'accepted':[fragment],'negative':bad,
                'label_provenance':'Original corpus span vs mechanical corruptions; alternative valid wording may be rejected.'})
            if len(rows)==count:break
        if len(rows)!=count:raise ValueError(f'Only {len(rows)} natural contexts for {split}; expected {count}')
        proc=read(Path(procedural)/f'{split}.jsonl')
        if split=='validation':proc=proc[:600]
        # Interleave so a prefix of the validation split includes both domains.
        combined=[]
        for i in range(max(len(rows),len(proc))):
            if i<len(rows):combined.append(rows[i])
            if i<len(proc):combined.append({**proc[i],'id':proc[i]['id']+'-procedural'})
        prepared[split]=combined
        print(json.dumps({'split':split,'corpus':len(rows),'procedural':len(proc)}),flush=True)
    manifest={'version':3,'seed':seed,'source_manifest':source_manifest,'procedural_manifest':json.loads((Path(procedural)/'manifest.json').read_text()),
        'provenance':'Mixed procedural contrasts and synthetic TinyStories excerpts; no human preference labels.',
        'license':'Corpus excerpts CDLA-Sharing-1.0; procedural material Apache-2.0.',
        'deduplication':'Exact normalized natural excerpts across splits; original source document splits retained. No near-duplicate claim.',
        'splits':{}}
    for split,rows in prepared.items():
        path=out/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        manifest['splits'][split]={'contexts':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':build()
