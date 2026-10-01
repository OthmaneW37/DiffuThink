"""Validation-only decision calibration, then a frozen matched-pool test.

The reference language is synthetic and closed-world: these scores do not stand
in for human judgments. Both classifiers receive identical generated candidates.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from .corrections import read, editing_cases
from .ranker import Reranker
from .train_ranker import encode_groups, validation
from diffuthink.v2.editor import rewrite_span
from diffuthink.v2.inference import load


def normalized(text):return text.strip().casefold().strip(' .!?')


def accepted(case,word):return normalized(word) in {normalized(x) for x in case['accepted']}


def decision(row,threshold,floor):
    best=max(row['candidates'],key=lambda c:c['ranker_score'],default=None)
    if best and best['ranker_score']-row['original_score']>threshold and best['ranker_score']>=floor:
        return best['replacement']
    return row['original_fragment']


def metrics(rows,chooser):
    error=[r for r in rows if not r['case']['clean']];clean=[r for r in rows if r['case']['clean']]
    corrected=sum(accepted(r['case'],chooser(r)) for r in error)
    unchanged=sum(chooser(r)==r['original_fragment'] for r in clean)
    bad=sum(chooser(r)!=r['original_fragment'] and not accepted(r['case'],chooser(r)) for r in error)
    return {'error_cases':len(error),'clean_cases':len(clean),'error_corrected':corrected/len(error),
        'clean_preserved':unchanged/len(clean),'balanced_success':.5*(corrected/len(error)+unchanged/len(clean)),
        'wrong_replacement_on_errors':bad/len(error),'clean_modified':1-unchanged/len(clean)}


def calibrate(rows):
    # Predeclared grid and harm gates; correctness is only synthetic reference matching.
    margins=[0,.25,.5,1,1.5,2,3,4,6,8,12,1000000.]
    floors=[-1000000.,-4,-2,0,1,2,3,4,6,8,1000000.]
    best=None
    for margin in margins:
        for floor in floors:
            m=metrics(rows,lambda r:decision(r,margin,floor))
            if m['clean_modified']>.05+1e-9 or m['wrong_replacement_on_errors']>.10+1e-9:continue
            key=(m['balanced_success'],-m['clean_modified'],-m['wrong_replacement_on_errors'],margin,floor)
            if best is None or key>best[0]:best=(key,margin,floor,m)
    return {'margin_threshold':best[1],'score_floor':best[2],'validation':best[3],
        'method':'Grid search on validation only: maximize balanced success subject to <=5% clean edits and <=10% wrong edits on errors.',
        'margin_grid':margins,'score_grid':floors,'contexts':len(rows)//2,
        'limitations':'Finite synthetic validation; thresholds are not guarantees on user text.'}


def generate_rows(model,tokenizer,ranker,contexts,device,path):
    path=Path(path);rows=[]
    # A completed cache is tied to both model hashes by the caller's manifest.
    if path.exists():return read(path)
    temporary=path.with_suffix('.jsonl.tmp')
    with temporary.open('w',encoding='utf-8') as file:
        for index,case in enumerate(editing_cases(contexts)):
            result=rewrite_span(model,tokenizer,case['text'],case['start'],case['end'],seed=301+index//2,
                                precision='bf16' if device=='cuda' else 'fp32')
            a,b=result['selection']['start'],result['selection']['end']
            scores=ranker.scores([(case['text'],a,b)]+[(c['text'],a,a+len(c['replacement'])) for c in result['candidates']])
            candidates=[{k:c[k] for k in ('replacement','text','local_nll','seed','masked_tokens')} for c in result['candidates']]
            for c,s in zip(candidates,scores[1:]):c['ranker_score']=s
            row={'case':case,'original_fragment':result['selection']['text'],'original_nll':result['original_local_nll'],
                'original_score':scores[0],'candidates':candidates,'latency_ms':result['latency_ms'],
                'context_preserved':all(c['text']==result['prefix']+c['replacement']+result['suffix'] for c in candidates)}
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush();rows.append(row)
            if (index+1)%40==0:print(json.dumps({'file':str(path),'cases':index+1}),flush=True)
    temporary.replace(path);return rows


def nll_choice(row):
    best=min(row['candidates'],key=lambda c:c['local_nll'],default=None)
    return best['replacement'] if best and best['local_nll']<row['original_nll'] else row['original_fragment']


def evaluate(rows,calibration):
    margin,floor=calibration['margin_threshold'],calibration['score_floor']
    learned=lambda r:decision(r,margin,floor)
    errors=[r for r in rows if not r['case']['clean']]
    top1=lambda r,key,reverse:next(iter(sorted(r['candidates'],key=lambda c:c[key],reverse=reverse)),{}).get('replacement','')
    return {'nll_keep_or_replace':metrics(rows,nll_choice),'learned_keep_or_replace':metrics(rows,learned),
        'same_pool_error_top1_nll':sum(accepted(r['case'],top1(r,'local_nll',False)) for r in errors)/len(errors),
        'same_pool_error_top1_ranker':sum(accepted(r['case'],top1(r,'ranker_score',True)) for r in errors)/len(errors),
        'error_candidate_coverage':sum(any(accepted(r['case'],c['replacement']) for c in r['candidates']) for r in errors)/len(errors),
        'context_preserved':sum(r['context_preserved'] for r in rows)/len(rows)}


def bootstrap(rows,calibration,seed=71):
    # Paired clean/error contexts, not 600 independent observations.
    rng=random.Random(seed);pairs=[rows[i:i+2] for i in range(0,len(rows),2)]
    deltas=[]
    for _ in range(1000):
        sample=[r for _ in pairs for r in rng.choice(pairs)]
        before=metrics(sample,nll_choice)['balanced_success']
        after=metrics(sample,lambda r:decision(r,calibration['margin_threshold'],calibration['score_floor']))['balanced_success']
        deltas.append(after-before)
    deltas.sort()
    return {'metric':'learned minus NLL balanced success','paired_contexts':len(pairs),'resamples':1000,
            'percentile_95':[deltas[25],deltas[974]],'limitations':'Within this procedural test; does not measure real-world validity.'}


def main(args):
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data);manifest=json.loads((data/'manifest.json').read_text())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    binding={'generator_weights_sha256':digest(Path(args.model)/'model.safetensors'),
             'ranker_weights_sha256':digest(Path(args.ranker)/'model.safetensors'),'corpus':manifest,
             'device':args.device,'generator_precision':'bf16' if args.device=='cuda' else 'fp32',
             'ranker_precision':'fp32','seed':301,'validation_contexts':args.validation_contexts}
    if (out/'binding.json').exists() and json.loads((out/'binding.json').read_text())!=binding:
        raise ValueError('Cache belongs to different weights/data/settings')
    (out/'binding.json').write_text(json.dumps(binding,indent=2))
    for split in ['validation','test']:
        if digest(data/f'{split}.jsonl')!=manifest['splits'][split]['sha256']:raise ValueError('Frozen evaluation changed')
    model,tokenizer=load(args.model,args.device);ranker=Reranker.load(args.ranker,tokenizer,args.device)
    valid=read(data/'validation.jsonl')
    validation_rows=generate_rows(model,tokenizer,ranker,valid[:args.validation_contexts],args.device,out/'validation_predictions.jsonl')
    calibration=calibrate(validation_rows)
    info_path=Path(args.ranker)/'ranker_info.json';info=json.loads(info_path.read_text());info['calibration']=calibration
    info_path.write_text(json.dumps(info,indent=2),encoding='utf-8');ranker.info=info
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    print(json.dumps({'calibration':calibration}),flush=True)
    test=read(data/'test.jsonl')
    pool=validation(ranker.model,encode_groups(test,tokenizer,512),test,args.device)
    test_rows=generate_rows(model,tokenizer,ranker,test,args.device,out/'test_predictions.jsonl')
    report={'version':1,'binding':binding,'reference_pool_test':pool,'end_to_end':evaluate(test_rows,calibration),
        'paired_bootstrap':bootstrap(test_rows,calibration),
        'categories':{cat:evaluate([r for r in test_rows if r['case']['category']==cat],calibration) for cat in sorted({c['category'] for c in test})},
        'limitations':['Rule-generated references, not human validated.', 'Held-out template combinations within six familiar categories.',
                       'Accepted alternatives are incomplete.', 'Paired clean/error cases are clustered by context.',
                       'Top1 conditional candidate preference differs from a cautious keep-or-replace recommendation.']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rng=random.Random(591); mapping={}
    with (out/'blind_review.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['id','original','selected','suggestion_A','suggestion_B','winner_A_B_tie','reason'])
        for r in test_rows:
            options=[('nll',nll_choice(r)),('ranker',decision(r,calibration['margin_threshold'],calibration['score_floor']))];rng.shuffle(options)
            c=r['case'];writer.writerow([c['id'],c['text'],r['original_fragment'],*[c['prefix']+x[1]+c['suffix'] for x in options],'',''])
            mapping[c['id']]=[x[0] for x in options]
    (out/'blind_review_key.json').write_text(json.dumps(mapping,indent=2))
    print(json.dumps(report['end_to_end']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/storypatch-corrections-v1')
    p.add_argument('--model',default='runs/storypatch-13m/best');p.add_argument('--ranker',default='runs/storypatch-ranker-v1/best')
    p.add_argument('--output',default='reports/storypatch-ranker-v1');p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--validation-contexts',type=int,default=120);main(p.parse_args())
