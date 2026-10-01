"""Train a random-initialized ranker on frozen rule-generated contrasts."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from diffuthink.v2.model import Config
from .corrections import read
from .ranker import SpanRanker, encode_candidate, collate


def encode_groups(contexts,tokenizer,max_length):
    return [[encode_candidate(tokenizer,c['prefix']+word+c['suffix'],len(c['prefix']),len(c['prefix'])+len(word),max_length)
             for word in c['accepted']+c['negative']] for c in contexts]


@torch.inference_mode()
def validation(model,groups,contexts,device,batch=128):
    model.eval(); rows=[r for group in groups for r in group]; scores=[]
    for start in range(0,len(rows),batch):scores.extend(model(*collate(rows[start:start+batch],device)).cpu().tolist())
    pos=[];neg=[]; wins=0;offset=0
    for group,c in zip(groups,contexts):
        values=scores[offset:offset+len(group)];offset+=len(group);p=len(c['accepted'])
        pos+=values[:p];neg+=values[p:]
        wins+=max(values[:p])>max(values[p:])
    logits=torch.tensor(pos+neg); labels=torch.tensor([1.]*len(pos)+[0.]*len(neg))
    return {'pool_top1':wins/len(groups),'bce':float(F.binary_cross_entropy_with_logits(logits,labels)),
            'positive_mean':sum(pos)/len(pos),'negative_mean':sum(neg)/len(neg),'contexts':len(groups)}


def run(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=random.Random(args.seed); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(args.tokenizer)
    manifest=json.loads((Path(args.data)/'manifest.json').read_text())
    for split in ['train','validation']:
        actual=hashlib.sha256((Path(args.data)/f'{split}.jsonl').read_bytes()).hexdigest()
        if manifest['splits'][split]['sha256']!=actual:raise ValueError('Frozen corpus changed')
    train=read(Path(args.data)/'train.jsonl');valid=read(Path(args.data)/'validation.jsonl')
    config=Config(vocab_size=tokenizer.get_vocab_size(),width=args.width,heads=4,layers=3,max_length=512,architecture='storypatch_span_ranker_v1')
    model=SpanRanker(config).to(args.device)
    groups=encode_groups(train,tokenizer,512);vgroups=encode_groups(valid,tokenizer,512)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    settings=vars(args).copy();settings.pop('resume',None)
    info={'architecture':config.architecture,'initialized_from':'random; no external or generator weights',
          'parameters':sum(p.numel() for p in model.parameters()),'settings':settings,'corpus':manifest,
          'tokenizer_sha256':hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
          'loss':'BCE(candidate compatibility) + 0.5 softplus(negative score - positive score)',
          'calibration':{'margin_threshold':1000000.,'score_floor':1000000.,'status':'not calibrated'}}
    start=0; best=math.inf;stale=0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['settings']!=settings:raise ValueError('Resume settings differ')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['step'];best=state['best'];stale=state['stale']
    begun=time.perf_counter();losses=[]
    print(json.dumps({'parameters':info['parameters'],'training_contexts':len(train),'device':args.device}),flush=True)
    for step in range(start+1,args.steps+1):
        model.train();rows=[]
        for _ in range(args.batch_pairs):
            i=rng.randrange(len(groups));p=len(train[i]['accepted'])
            rows += [groups[i][rng.randrange(p)],groups[i][rng.randrange(p,len(groups[i]))]]
        scores=model(*collate(rows,args.device)); positive=scores[0::2];negative=scores[1::2]
        labels=torch.arange(len(rows),device=args.device).remainder(2).eq(0).float()
        loss=F.binary_cross_entropy_with_logits(scores,labels)+.5*F.softplus(negative-positive).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        warm=min(1.,step/100);decay=.1+.9*.5*(1+math.cos(math.pi*step/args.steps))
        for group in optimizer.param_groups:group['lr']=args.lr*warm*decay
        optimizer.step();losses.append(float(loss.detach()))
        if step%args.eval_every==0 or step==args.steps:
            metrics=validation(model,vgroups,valid,args.device)
            record={'step':step,'train_loss':sum(losses)/len(losses),'elapsed_seconds':time.perf_counter()-begun,**metrics}
            losses=[]
            with (out/'learning.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if metrics['bce']<best:
                best=metrics['bce'];stale=0
                model.save(out/'best',{**info,'step':step,'validation':metrics})
            else:stale+=1
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'python_rng':rng.getstate(),
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else None,
                   'step':step,'best':best,'stale':stale,'settings':settings}
            torch.save(state,out/'resume.pt.tmp');(out/'resume.pt.tmp').replace(out/'resume.pt')
            if stale>=6:print('Early stop: six validation checks without improvement',flush=True);break
    print(str(out/'best'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/storypatch-corrections-v1')
    p.add_argument('--tokenizer',default='runs/storypatch-13m/best/tokenizer.json')
    p.add_argument('--output',default='runs/storypatch-ranker-v1')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--batch-pairs',type=int,default=48)
    p.add_argument('--width',type=int,default=192);p.add_argument('--lr',type=float,default=.0007)
    p.add_argument('--eval-every',type=int,default=200);p.add_argument('--seed',type=int,default=73)
    p.add_argument('--resume');run(p.parse_args())
