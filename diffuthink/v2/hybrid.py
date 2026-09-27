"""Continue our own scratch-trained weights with causal + denoising objectives.

The continuation path is explicitly autoregressive, not diffusion sampling.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import Denoiser, PAD, BOS, EOS, MASK, UNK, corrupt
from .data import load_arrays
from .train import amp, atomic_save, evaluate


def banned_completions(ids, n):
    """Tokens that would repeat an n-gram already present in this context."""
    if n < 1 or len(ids) < n - 1:
        return set()
    prefix = tuple(ids[-(n-1):]) if n > 1 else ()
    return {ids[i+n-1] for i in range(len(ids)-n+1) if tuple(ids[i:i+n-1]) == prefix}


@torch.inference_mode()
def causal_evaluate(model, array, limit=256, batch_size=16, precision="bf16"):
    device = str(next(model.parameters()).device).split(":")[0]
    model.eval()
    loss = correct = count = 0
    for start in range(0, min(limit, len(array)), batch_size):
        clean = torch.tensor(np.array(array[start:min(start+batch_size, limit)]), dtype=torch.long, device=device)
        inputs, targets = clean[:, :-1], clean[:, 1:]
        selected = targets.ne(PAD)
        with amp(device, precision):
            logits = model(inputs, torch.zeros(len(inputs), device=device), selected, causal=True, right_padded=True)
        loss += F.cross_entropy(logits.float(), targets[selected], reduction="sum").item()
        correct += int(logits.argmax(-1).eq(targets[selected]).sum())
        count += int(selected.sum())
    return {"nll": loss/count, "perplexity": math.exp(loss/count), "accuracy": correct/count, "tokens": count}


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=96, temperature=0.5, seed=42,
             top_p=0.9, repetition_penalty=1.12, no_repeat_ngram=4, precision="bf16",
             finish_sentence_tokens=0, **unused):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    if not 1 <= max_new_tokens <= 256 or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
        raise ValueError("Invalid generation length or temperature")
    if not 0 < top_p <= 1 or repetition_penalty < 1 or not 0 <= seed < 2**63:
        raise ValueError("Invalid sampling settings")
    if not 0 <= no_repeat_ngram <= 10:
        raise ValueError("no_repeat_ngram must be 0..10")
    if not isinstance(finish_sentence_tokens, int) or not 0 <= finish_sentence_tokens <= 64:
        raise ValueError("Sentence extension must be 0..64 tokens")
    device = str(next(model.parameters()).device).split(":")[0]
    ids = [BOS] + tokenizer.encode(prompt.rstrip()).ids
    if len(ids) >= model.config.max_length:
        raise ValueError(f"Prompt must be shorter than {model.config.max_length} tokens")
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    output, trace = list(ids), []
    stop_reason = "length"
    started = time.perf_counter()
    for step in range(max_new_tokens + finish_sentence_tokens):
        context = [BOS] + output[1:][-(model.config.max_length-1):]
        tokens = torch.tensor([context], device=device)
        with amp(device, precision):
            selected = torch.zeros_like(tokens, dtype=torch.bool)
            selected[0,-1] = True
            logits = model(tokens, torch.zeros(1, device=device), selected, causal=True, right_padded=True)[0].float()
        logits[[PAD, BOS, MASK, UNK]] = -torch.inf
        if step < 8:
            logits[EOS] = -torch.inf
        recent = torch.tensor(sorted(set(output[-64:])), device=device)
        logits[recent] = torch.where(logits[recent] < 0, logits[recent]*repetition_penalty, logits[recent]/repetition_penalty)
        banned = banned_completions(output, no_repeat_ngram)
        if banned:
            logits[list(banned)] = -torch.inf
        if temperature == 0:
            token = int(logits.argmax())
        else:
            values, indices = (logits/temperature).sort(descending=True)
            probs = values.softmax(-1)
            remove = probs.cumsum(-1) - probs >= top_p
            probs[remove] = 0
            token = int(indices[torch.multinomial(probs, 1, generator=generator)])
        if token == EOS:
            stop_reason = "eos"
            break
        output.append(token)
        trace.append({"step": step+1, "remaining": None, "text": tokenizer.decode(output), "mode": "autoregressive"})
        if finish_sentence_tokens and step+1 >= max_new_tokens and sentence_ended(trace[-1]["text"]):
            stop_reason = "sentence_boundary"
            break
    return {"text": tokenizer.decode(output), "tokens": output, "trace": trace, "seed": seed,
            "stop_reason": stop_reason, "generated_tokens": len(output)-len(ids),
            "forward_passes": step+1, "latency_ms": (time.perf_counter()-started)*1000, "mode": "autoregressive",
            "sampling": {"temperature": temperature, "top_p": top_p, "repetition_penalty": repetition_penalty,
                         "no_repeat_ngram": no_repeat_ngram, "finish_sentence_tokens": finish_sentence_tokens}}


def sentence_ended(text):
    # A transparent display heuristic, not a grammar correction or proof of a complete story.
    return text.rstrip().rstrip('\"\'”’').endswith((".", "!", "?"))


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arrays, manifest = load_arrays(args.data)
    lineage = json.loads((Path(args.source)/"training_info.json").read_text())
    if lineage["pretrained_weights"] is not False:
        raise ValueError("Expected our own scratch-trained checkpoint")
    if lineage["manifest"] != manifest:
        previous = lineage["manifest"]
        if not getattr(args, "expand_data", False):
            raise ValueError("Changed corpus requires --expand-data")
        for key in ("dataset", "revision", "tokenizer_sha256", "vocab_size"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Incompatible expanded data: {key}")
        for split in ("validation", "test"):
            if previous["splits"][split]["documents_sha256"] != manifest["splits"][split]["documents_sha256"]:
                raise ValueError("Held-out documents must stay identical")
    if (out/"last.pt").exists() and not args.resume:
        raise ValueError("Run exists; use --resume")
    model = Denoiser.from_pretrained(args.source, args.device)
    if manifest["max_length"] != model.config.max_length:
        if not getattr(args, "expand_data", False):
            raise ValueError("Context change requires --expand-data")
        model.extend_context(manifest["max_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    sampling = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=args.device).manual_seed(args.seed+1)
    settings = vars(args).copy()
    settings.pop("resume")
    step, seen, elapsed, best = 0, 0, 0., float("inf")
    if args.resume:
        state = torch.load(out/"last.pt", map_location="cpu", weights_only=True)
        if state["settings"] != settings or state["manifest"] != manifest:
            raise ValueError("Recovery settings/data mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampling.set_state(state["sampling_rng"])
        noise_rng.set_state(state["noise_rng"])
        step, seen, elapsed, best = state["step"],state["seen_tokens"],state["seconds"],state["best"]
    else:
        initial = causal_evaluate(model, arrays["validation"], args.eval_samples)
        (out/"initial_validation.json").write_text(json.dumps(initial,indent=2))
    (out/"settings.json").write_text(json.dumps(settings,indent=2))
    start=time.perf_counter()
    for step in range(step+1,args.steps+1):
        model.train()
        lr = args.lr*min(1.,step/args.warmup)*(0.1+0.9*0.5*(1+math.cos(math.pi*max(0.,(step-args.warmup)/(args.steps-args.warmup)))))
        for group in optimizer.param_groups: group["lr"]=lr
        indices=torch.randint(len(arrays["train"]),(args.batch_size,),generator=sampling).numpy()
        clean=torch.tensor(np.array(arrays["train"][indices]),dtype=torch.long,device=args.device)
        optimizer.zero_grad(set_to_none=True)
        # Nine causal batches followed by one denoising batch. No future leakage
        # in the causal path, no separate pretrained teacher or reranker.
        denoise_every = getattr(args, "denoise_every", 10)
        denoising=denoise_every > 0 and step%denoise_every==0
        with amp(args.device,"bf16"):
            if denoising:
                inputs,selected,noise=corrupt(clean,noise_rng)
                logits=model(inputs,noise,selected)
                targets=clean[selected]
            else:
                targets=clean[:,1:]
                selected=targets.ne(PAD)
                logits=model(clean[:,:-1],torch.zeros(len(clean),device=args.device),selected,causal=True,right_padded=True)
                targets=targets[selected]
            loss=F.cross_entropy(logits.float(),targets)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError("Non-finite gradient")
        optimizer.step()
        seen+=int(clean.ne(PAD).sum())
        if step==1 or step%100 in (0,1):
            row={"step":step,"objective":"denoising" if denoising else "causal","loss":loss.item(),"lr":lr,"seen_tokens":seen,"seconds":elapsed+time.perf_counter()-start}
            with (out/"metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            metric=causal_evaluate(model,arrays["validation"],args.eval_samples)
            row={"step":step,"validation":metric}
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            if metric["nll"]<best:
                best=metric["nll"]
                model.save_pretrained(out/"best")
                shutil.copy2(Path(args.data)/"tokenizer.json",out/"best"/"tokenizer.json")
                info={"step":step,"validation_loss":best,"parameters":sum(p.numel() for p in model.parameters()),
                      "seen_tokens":seen,"pretrained_weights":False,"manifest":manifest,"previous_phase":lineage,
                      "continuation_mode":"autoregressive","objective":f"Denoising every {denoise_every} updates (0 = causal only)",
                      "context_extension":manifest["max_length"] != lineage["manifest"]["max_length"]}
                (out/"best"/"training_info.json").write_text(json.dumps(info,indent=2))
            atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"sampling_rng":sampling.get_state(),
                         "noise_rng":noise_rng.get_state(),"step":step,"seen_tokens":seen,"seconds":elapsed+time.perf_counter()-start,
                         "best":best,"settings":settings,"manifest":manifest},out/"last.pt")
            print(json.dumps(row),flush=True)
    model=Denoiser.from_pretrained(out/"best",args.device)
    counts=np.ones(model.config.vocab_size,dtype=float)
    for chunk_start in range(0,len(arrays["train"]),4096):
        counts+=np.bincount(np.asarray(arrays["train"][chunk_start:chunk_start+4096]).reshape(-1),minlength=model.config.vocab_size)
    counts[[PAD,BOS]]=1
    frequencies=torch.tensor(counts/counts.sum(),dtype=torch.float32)
    report={"settings":settings,"best":json.loads((out/"best"/"training_info.json").read_text()),"seen_tokens":seen,
            "seconds":elapsed+time.perf_counter()-start,"causal_test":causal_evaluate(model,arrays["test"],512),
            "denoising_test":evaluate(model,arrays["test"],frequencies,device=args.device,precision="bf16",limit=512)}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print("Completed "+str(out),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="runs/stories-v2-refined/best")
    p.add_argument("--data",default="data/processed/stories-v2")
    p.add_argument("--output",default="runs/stories-hybrid")
    p.add_argument("--device",default="cuda",choices=["cpu","cuda"])
    p.add_argument("--steps",type=int,default=16000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=0.0003)
    p.add_argument("--warmup",type=int,default=300)
    p.add_argument("--eval-every",type=int,default=1000)
    p.add_argument("--eval-samples",type=int,default=256)
    p.add_argument("--seed",type=int,default=44)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--expand-data",action="store_true",help="Extend context/corpus, requiring same vocabulary and held-out documents")
    p.add_argument("--denoise-every",type=int,default=10,help="0 for causal only; otherwise every Nth update")
    args=p.parse_args()
    if args.steps<=args.warmup or min(args.batch_size,args.eval_every,args.eval_samples,args.warmup)<1 or args.denoise_every < 0:
        p.error("Invalid training counts")
    train(args)


if __name__=="__main__": main()
