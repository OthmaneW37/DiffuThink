"""Bounded, inspectable span rewriting with exact preservation of surrounding text."""
import math
import time
import torch
from torch.nn import functional as F
from .model import BOS,EOS,MASK
from .inference import denoise
from .train import amp


@torch.inference_mode()
def local_nll(model,tokenizer,text,start,end,precision="fp32"):
    encoded=tokenizer.encode(text)
    if len(encoded.ids)+1>model.config.max_length:
        raise ValueError("Text exceeds model context")
    positions=[i for i,(a,b) in enumerate(encoded.offsets) if b>start and a<end]
    if not positions:raise ValueError("Selection has no tokens")
    last=min(len(encoded.ids),positions[-1]+1+16)
    device=next(model.parameters()).device
    tokens=torch.tensor([[BOS]+encoded.ids],device=device)
    targets=torch.tensor(encoded.ids+[EOS],device=device)
    selected=torch.zeros_like(tokens,dtype=torch.bool)
    selected[0,positions[0]:last]=True
    with amp(device.type,precision):
        logits=model(tokens,torch.zeros(1,device=device),selected,causal=True,right_padded=True)
    return float(F.cross_entropy(logits.float(),targets[selected[0]]))


@torch.inference_mode()
def rewrite_span(model,tokenizer,text,start,end,seed=42,precision="fp32"):
    if not isinstance(text,str) or not 1<=len(text)<=3000:
        raise ValueError("Use a short English text (1..3000 characters)")
    if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(text):
        raise ValueError("Select a passage inside the text")
    if not isinstance(seed,int) or not 0<=seed<2**63-16:
        raise ValueError("Invalid seed")
    original=text[start:end]
    a=start+len(original)-len(original.lstrip())
    b=end-(len(original)-len(original.rstrip()))
    if a>=b or len(text[a:b].split())>12:
        raise ValueError("Select 1 to 12 words")
    if (a>0 and text[a-1].isalnum() and text[a].isalnum()) or (b<len(text) and text[b-1].isalnum() and text[b].isalnum()):
        raise ValueError("Select complete words")
    if len(tokenizer.encode(text).ids)>model.config.max_length-32:
        raise ValueError("Text too long; shorten it before editing")
    prefix,suffix=text[:a],text[b:]
    left=tokenizer.encode(prefix.rstrip()).ids
    right=tokenizer.encode(suffix).ids
    count=len(tokenizer.encode((' ' if prefix else '')+text[a:b]).ids)
    if not 1<=count<=16:raise ValueError("Select a shorter passage (at most 16 subword tokens)")
    before=local_nll(model,tokenizer,text,a,b,precision)
    candidates=[];seen={text[a:b]};started=time.perf_counter()
    # Six deterministic attempts; all unique candidates are retained with disclosed ranking.
    lengths=[count,count,count,max(1,count-1),min(16,count+1),count]
    for attempt,length in enumerate(lengths):
        ids=[BOS]+left+[MASK]*length+right+[EOS]
        result=denoise(model,tokenizer,ids,steps=12,temperature=0 if attempt==0 else .8,
                       seed=seed+attempt,precision=precision,allow_eos=False)
        fragment=tokenizer.decode(result["tokens"][1+len(left):1+len(left)+length]).strip()
        if not fragment or fragment in seen or '\ufffd' in fragment or len(fragment.split())>14:
            continue
        seen.add(fragment)
        rewritten=prefix+fragment+suffix
        score=local_nll(model,tokenizer,rewritten,a,a+len(fragment),precision)
        trace=[]
        for row in result["trace"]:
            # The raw model trace is shown for research inspection, not used to rewrite context.
            trace.append({"step":row["step"],"remaining":row["remaining"],"model_text":row["text"]})
        candidates.append({"replacement":fragment,"text":rewritten,"local_nll":score,
            "delta_nll":score-before,"seed":seed+attempt,"masked_tokens":length,"trace":trace})
    candidates.sort(key=lambda c:c["local_nll"])
    return {"original":text,"selection":{"start":a,"end":b,"text":text[a:b]},
        "prefix":prefix,"suffix":suffix,"original_local_nll":before,"candidates":candidates,
        "attempts":len(lengths),"latency_ms":(time.perf_counter()-started)*1000,
        "method":"Bidirectional denoising; rank by causal NLL of the span and up to 16 following tokens.",
        "limitation":"Ranking reflects this model's preferences, not verified grammar or semantic correctness."}
