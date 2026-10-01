"""Add generator proposals from TRAIN contexts only to the contrast corpus.

Labels use the bounded procedural answer list. Alternatives absent from that
list can be wrongly rejected; provenance and every proposal are retained.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from .corrections import read
from .evaluate_ranker import accepted
from diffuthink.v2.editor import rewrite_span
from diffuthink.v2.inference import load


def main(args):
    torch.set_num_threads(4);source=Path(args.data);out=Path(args.output)
    if out.exists():raise ValueError('Choose a fresh mining output')
    out.mkdir(parents=True)
    model,tokenizer=load(args.model,args.device);train=read(source/'train.jsonl')
    mined=[];additional=0
    for i,c in enumerate(train[:args.contexts]):
        word=c['negative'][0];text=c['prefix']+word+c['suffix'];a=len(c['prefix'])
        result=rewrite_span(model,tokenizer,text,a,a+len(word),seed=701+i,precision='bf16' if args.device=='cuda' else 'fp32')
        proposals=[x['replacement'] for x in result['candidates']]
        new=[p for p in proposals if not accepted(c,p) and p not in c['negative']]
        c['negative']+=new;additional+=len(new)
        mined.append({'context_id':c['id'],'proposals':proposals,'added_negative':new,
                      'label_rule':'Not in bounded explicit synthetic answer list; may reject valid alternatives.'})
        if (i+1)%100==0:print(json.dumps({'training_contexts':i+1,'added_negatives':additional}),flush=True)
    (out/'train.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in train),encoding='utf-8')
    for split in ['validation','test']:shutil.copy2(source/f'{split}.jsonl',out/f'{split}.jsonl')
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['splits']['train']['sha256']=hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest()
    manifest['mining']={'contexts':args.contexts,'added_negatives':additional,'source_corpus':json.loads((source/'manifest.json').read_text()),
        'generator_sha256':hashlib.sha256((Path(args.model)/'model.safetensors').read_bytes()).hexdigest(),
        'seed':701,'label_rule':'Closed-world exact accepted list; not human judged. Training split only.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'mining.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in mined),encoding='utf-8')
    print(str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/storypatch-corrections-v2')
    p.add_argument('--output',default='data/processed/storypatch-corrections-mined-v2');p.add_argument('--model',default='runs/storypatch-13m/best')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda']);p.add_argument('--contexts',type=int,default=1200)
    main(p.parse_args())
