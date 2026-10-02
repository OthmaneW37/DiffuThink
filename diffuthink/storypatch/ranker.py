"""Bidirectional span compatibility scorer with project-local weight lineage.

Scores are learned preferences, not probabilities of semantic correctness.
"""
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch import nn
from diffuthink.model import Block, RMSNorm
from diffuthink.v2.model import Config, BOS, EOS, PAD


class SpanRanker(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.width,padding_idx=PAD)
        self.segment=nn.Embedding(2,config.width)
        if config.architecture=='storypatch_context_ranker_v2':
            self.position=nn.Embedding(config.max_length,config.width)
            self.time_mlp=nn.Sequential(nn.Linear(config.width,config.width),nn.SiLU(),nn.Linear(config.width,config.width))
        self.blocks=nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm=RMSNorm(config.width)
        self.dropout=nn.Dropout(.1)
        self.head=nn.Sequential(nn.Linear(config.width*2,config.width),nn.SiLU(),nn.Linear(config.width,1))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if getattr(module,'bias',None) is not None: nn.init.zeros_(module.bias)

    @classmethod
    def from_own_generator(cls,path):
        from diffuthink.v2.model import Denoiser
        source=Denoiser.from_pretrained(path)
        model=cls(replace(source.config,architecture='storypatch_context_ranker_v2'))
        for name in ['embedding','position','time_mlp','blocks','norm']:
            getattr(model,name).load_state_dict(getattr(source,name).state_dict())
        nn.init.zeros_(model.segment.weight)
        return model

    def hidden(self,tokens,selected):
        if tokens.ndim!=2 or tokens.shape!=selected.shape or tokens.shape[1]>self.config.max_length:
            raise ValueError('Invalid ranker input shape')
        valid=tokens.ne(PAD); span=selected & valid
        if not span.any(-1).all(): raise ValueError('Every candidate needs a selected token')
        x=self.dropout(self.embedding(tokens)+self.segment(selected.long()))
        time=torch.zeros(tokens.shape[0],self.config.width,device=tokens.device,dtype=x.dtype)
        if hasattr(self,'position'):
            x=x+self.position(torch.arange(tokens.shape[1],device=tokens.device))
            half=self.config.width//2
            # Exactly the generator's zero-noise conditioning: sin(0), cos(0).
            time=self.time_mlp(torch.cat([time[:,:half],torch.ones_like(time[:,:half])],-1))
        for block in self.blocks: x=block(x,time,valid)
        return self.norm(x)

    def forward(self,tokens,selected):
        x=self.hidden(tokens,selected);valid=tokens.ne(PAD);span=selected & valid
        def pool(mask): return (x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.head(self.dropout(torch.cat([pool(span),pool(valid)],-1))).squeeze(-1)

    def save(self,path,metadata):
        from safetensors.torch import save_file
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        (path/'config.json').write_text(json.dumps(asdict(self.config),indent=2),encoding='utf-8')
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(path/'model.safetensors'))
        (path/'ranker_info.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def encode_candidate(tokenizer,text,start,end,max_length):
    encoded=tokenizer.encode(text)
    indices=[i for i,(a,b) in enumerate(encoded.offsets) if b>start and a<end]
    if not indices: raise ValueError('Candidate contains no selected tokens')
    # No context is silently cropped: a refusal is preferable to an incomplete score.
    ids=[BOS]+encoded.ids+[EOS]
    if len(ids)>max_length: raise ValueError('Text exceeds ranker context')
    selected=[False]*len(ids)
    for i in indices:selected[i+1]=True
    return ids,selected


def collate(rows,device):
    length=max(len(ids) for ids,_ in rows)
    tokens=torch.full((len(rows),length),PAD,dtype=torch.long,device=device)
    selected=torch.zeros_like(tokens,dtype=torch.bool)
    for i,(ids,mask) in enumerate(rows):
        tokens[i,:len(ids)]=torch.tensor(ids,device=device)
        selected[i,:len(ids)]=torch.tensor(mask,device=device)
    return tokens,selected


def blended_score(nll,learned,weight):
    """Bound learned influence; weight zero recovers exactly the NLL ordering."""
    return -nll+weight*max(-8.,min(8.,learned))


class Reranker:
    def __init__(self,model,tokenizer,info):
        self.model=model.eval(); self.tokenizer=tokenizer; self.info=info

    @classmethod
    def load(cls,path,tokenizer,device='cpu'):
        from safetensors.torch import load_file
        path=Path(path); info=json.loads((path/'ranker_info.json').read_text())
        # Bind the exact BPE, not only its vocabulary size.
        actual=hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        if actual!=info['tokenizer_sha256']: raise ValueError('Ranker tokenizer fingerprint mismatch')
        model=SpanRanker(Config(**json.loads((path/'config.json').read_text())))
        model.load_state_dict(load_file(str(path/'model.safetensors')))
        return cls(model.to(device),tokenizer,info)

    @torch.inference_mode()
    def scores(self,texts):
        rows=[encode_candidate(self.tokenizer,t,a,b,self.model.config.max_length) for t,a,b in texts]
        device=next(self.model.parameters()).device
        # FP32 keeps recommendation thresholds independent of generator precision.
        return self.model(*collate(rows,device)).float().cpu().tolist()

    def rank(self,result):
        a,b=result['selection']['start'],result['selection']['end']
        candidates=result['candidates']
        try:
            values=self.scores([(result['original'],a,b)]+[(c['text'],a,a+len(c['replacement'])) for c in candidates])
        except ValueError as exc:
            result['recommendation']={'action':'keep','reason':'ranker_unavailable','detail':str(exc)}
            return result
        blending=self.info.get('blending')
        if blending is not None:
            result['original_raw_ranker_score']=values[0]
            for c,value in zip(candidates,values[1:]):c['raw_ranker_score']=value
            values=[blended_score(result['original_local_nll'],values[0],blending['weight'])]+[
                blended_score(c['local_nll'],s,blending['weight']) for c,s in zip(candidates,values[1:])]
            result['blending']=blending
        original=values[0]; result['original_ranker_score']=original
        for c,score in zip(candidates,values[1:]):c['ranker_score']=score;c['ranker_margin']=score-original
        candidates.sort(key=lambda c:c['ranker_score'],reverse=True)
        calibration=self.info.get('calibration',{})
        threshold=calibration.get('margin_threshold',float('inf'))
        floor=calibration.get('score_floor',float('inf'))
        best=candidates[0] if candidates else None
        edit=bool(best and best['ranker_margin']>threshold and best['ranker_score']>=floor)
        result['recommendation']={'action':'replace' if edit else 'keep',
            'replacement':best['replacement'] if edit else result['selection']['text'],
            'reason':'learned_preference' if edit else 'insufficient_margin',
            'margin_threshold':threshold,'score_floor':floor}
        result['method']='Bidirectional denoising, then independently trained span compatibility ranking; validation-calibrated keep/replace decision.'
        if blending is not None:result['method']='Bidirectional denoising, then bounded learned compatibility combined with causal NLL; validation-calibrated keep/replace decision.'
        result['limitation']='Synthetic training domain; learned scores are preferences, not verified correctness or confidence probabilities.'
        return result
