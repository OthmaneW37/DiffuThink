"""Legacy diagnostics outside the new procedural templates; never tune on these."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diffuthink.storypatch.benchmark import cases
from diffuthink.storypatch.evaluate_ranker import accepted
from diffuthink.storypatch.ranker import Reranker
from diffuthink.v2.editor import rewrite_span
from diffuthink.v2.inference import load


def main():
    torch.set_num_threads(4);model,tokenizer=load(ROOT/'runs/storypatch-13m/best','cuda')
    ranker=Reranker.load(ROOT/'runs/storypatch-ranker-v2/best',tokenizer,'cuda');rows=[]
    for i,case in enumerate(cases()):
        result=rewrite_span(model,tokenizer,case['text'],case['start'],case['end'],seed=42+i*13,precision='bf16',ranker=ranker)
        nll=min(result['candidates'],key=lambda c:c['local_nll'],default={}).get('replacement','')
        learned=next(iter(result['candidates']),{}).get('replacement','')
        recommended=result['recommendation']['replacement']
        rows.append({'case':case,'nll_top1':nll,'learned_top1':learned,'recommended':recommended,
                     'recommendation':result['recommendation'],
                     'candidates':[{k:c[k] for k in ['replacement','local_nll','ranker_score']} for c in result['candidates']]})
    summary={key:sum(accepted(r['case'],r[key]) for r in rows)/len(rows) for key in ['nll_top1','learned_top1','recommended']}
    payload={'summary':summary,'cases':len(rows),'rows':rows,'settings':{'seed':42,'seed_increment':13,'precision':'bf16',
        'ranker_precision':'fp32','device':'cuda','role':'Legacy diagnostic probes only, no checkpoint or threshold selection.'},
        'limitations':'Authored synthetic probes with incomplete exact answers; no human evaluation.'}
    (ROOT/'reports/storypatch-ranker-v2/legacy_probes.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
