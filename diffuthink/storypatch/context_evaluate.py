"""Freeze a bounded NLL/compatibility blend on validation before testing."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from .corrections import read
from .ranker import Reranker, blended_score
from .evaluate_ranker import generate_rows, calibrate, evaluate, accepted, bootstrap
from .train_ranker import encode_groups, validation
from .benchmark import cases
from diffuthink.v2.inference import load
from diffuthink.v2.editor import rewrite_span


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blend(rows,weight):
    output=copy.deepcopy(rows)
    for r in output:
        r['raw_original_score']=r['original_score']
        r['original_score']=blended_score(r['original_nll'],r['original_score'],weight)
        for c in r['candidates']:
            c['raw_ranker_score']=c['ranker_score']
            c['ranker_score']=blended_score(c['local_nll'],c['ranker_score'],weight)
    return output


def top1(rows):
    errors=[r for r in rows if not r['case']['clean']]
    return sum(accepted(r['case'],max(r['candidates'],key=lambda c:c['ranker_score'],default={}).get('replacement','')) for r in errors)/len(errors)


def choose_blend(rows):
    # Both domains must meet their NLL validation top1 score; zero is a valid fallback.
    domains={'corpus':[r for r in rows if r['case']['category']=='corpus'],
             'procedural':[r for r in rows if r['case']['category']!='corpus']}
    baseline={name:top1(blend(group,0.)) for name,group in domains.items()}
    trials=[]
    for weight in [0.,.05,.1,.2,.4,.8,1.6]:
        scores={name:top1(blend(group,weight)) for name,group in domains.items()}
        passes=all(scores[name]+1e-12>=baseline[name] for name in domains)
        trials.append({'weight':weight,'top1_by_domain':scores,'passes':passes})
    eligible=[t for t in trials if t['passes']]
    selected=max(eligible,key=lambda t:(sum(t['top1_by_domain'].values()),-t['weight']))
    return {'weight':selected['weight'],'formula':'-local_nll + weight * clip(learned_score, -8, 8)',
            'selection':'Validation top1 improvement with no top1 loss in either corpus or procedural domain; ties prefer lower weight.',
            'baseline':baseline,'trials':trials}


def main():
    torch.set_num_threads(4)
    data=Path('data/processed/storypatch-context-v3');out=Path('reports/storypatch-context-v3');out.mkdir(parents=True,exist_ok=True)
    generator=Path('runs/storypatch-13m/best');directory=Path('runs/storypatch-context-ranker-v3/best')
    manifest=json.loads((data/'manifest.json').read_text())
    binding={'generator_sha256':digest(generator/'model.safetensors'),'ranker_sha256':digest(directory/'model.safetensors'),
        'corpus':manifest,'validation_contexts':240,'precision':'bf16 generator / fp32 ranker','device':'cuda','seed':301}
    binding_path=out/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:raise ValueError('Cache binding mismatch')
    binding_path.write_text(json.dumps(binding,indent=2),encoding='utf-8')
    for split in ['validation','test']:
        if digest(data/f'{split}.jsonl')!=manifest['splits'][split]['sha256']:raise ValueError('Corpus changed')
    model,tokenizer=load(generator,'cuda');ranker=Reranker.load(directory,tokenizer,'cuda')
    valid=read(data/'validation.jsonl')[:240]
    validation_rows=generate_rows(model,tokenizer,ranker,valid,'cuda',out/'validation_predictions.jsonl')
    blending=choose_blend(validation_rows);calibration=calibrate(blend(validation_rows,blending['weight']))
    ranker.info['blending']=blending;ranker.info['calibration']=calibration
    (directory/'ranker_info.json').write_text(json.dumps(ranker.info,indent=2),encoding='utf-8')
    (out/'selection.json').write_text(json.dumps({'blending':blending,'calibration':calibration},indent=2),encoding='utf-8')
    print(json.dumps({'blend':blending,'calibration':calibration}),flush=True)
    # Test is first scored only after the complete inference policy is fixed.
    test=read(data/'test.jsonl');raw=generate_rows(model,tokenizer,ranker,test,'cuda',out/'test_predictions.jsonl')
    mixed=blend(raw,blending['weight'])
    groups={'all':mixed,'corpus':[r for r in mixed if r['case']['category']=='corpus'],
            'procedural':[r for r in mixed if r['case']['category']!='corpus']}
    results={name:evaluate(group,calibration) for name,group in groups.items()}
    legacy=[]
    for i,c in enumerate(cases()):
        r=rewrite_span(model,tokenizer,c['text'],c['start'],c['end'],seed=42+i*13,precision='bf16',ranker=ranker)
        base=min(r['candidates'],key=lambda x:x['local_nll'],default={}).get('replacement','')
        new=next(iter(r['candidates']),{}).get('replacement','')
        legacy.append({'case':c,'nll':base,'blended':new,'recommendation':r['recommendation'],
                       'candidates':[{k:x[k] for k in ['replacement','local_nll','ranker_score','raw_ranker_score']} for x in r['candidates']]})
    legacy_metrics={key:sum(accepted(r['case'],r[key]) for r in legacy)/len(legacy) for key in ['nll','blended']}
    previous=Reranker.load('runs/storypatch-ranker-v2/best',tokenizer,'cpu');previous_correct=0
    for row in legacy:
        c=row['case'];a,b=c['start'],c['end'];pool=row['candidates']
        texts=[(c['text'][:a]+r['replacement']+c['text'][b:],a,a+len(r['replacement'])) for r in pool]
        scores=previous.scores(texts) if texts else []
        selected=pool[max(range(len(scores)),key=lambda i:scores[i])]['replacement'] if scores else ''
        row['previous_ranker_top1']=selected;previous_correct+=accepted(c,selected)
    previous_metrics={'top1':previous_correct/len(legacy),'device':'cpu','precision':'fp32',
        'weights_sha256':digest('runs/storypatch-ranker-v2/best/model.safetensors'),
        'candidate_pool':'exact stored candidate pool shared with NLL and v3; no regeneration'}
    report={'binding':binding,'selection':{'blending':blending,'calibration':calibration},'test':results,
        'bootstrap':bootstrap(mixed,calibration),'legacy':{'summary':legacy_metrics,'rows':legacy,'previous_ranker_matched_pool':previous_metrics},
        'reference_pool_test':validation(ranker.model,encode_groups(test,tokenizer,512),test,'cuda'),
        'limits':['TinyStories is synthetic. No human preference labels.', 'Corpus metric is original-span reconstruction, not semantic correctness.',
                  'Procedural cases are the previous benchmark reused for regression tracking.', 'Legacy probes are repeated diagnostics, not a new unseen test.']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'test':results,'legacy':legacy_metrics}),flush=True)


if __name__=='__main__':main()
