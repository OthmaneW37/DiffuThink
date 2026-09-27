"""Fixed prompts, raw before/after outputs; no best-of selection or LLM judge."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from diffuthink.v2.inference import load,continue_text
from diffuthink.v2.hybrid import generate

PROMPTS=[
    "One day a man was under a tree and ",
    "Once upon a time, a little girl found",
    "Tom opened the door and saw",
    "The woman stood on the balcony and",
    "A small bird was afraid to fly. One day,",
    "Lily lost her red ball. She looked under the bed and",
    "The rain stopped, so the children",
    "A boy wanted to help his mother, but",
    "The dog saw a cat near the river. It",
    "Anna opened the box. Inside, she found",
    "Ben was sad because his friend",
    "The old man planted a tree. Every morning,",
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="runs/stories-v2-refined/best")
    p.add_argument("--after",default="runs/stories-hybrid/best")
    p.add_argument("--output",default="reports/hybrid/continuations.json")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    torch.set_num_threads(4)
    rows=[{"prompt":prompt} for prompt in PROMPTS]
    for key,path,fn in [("before",args.before,continue_text),("after_raw",args.after,generate),("after",args.after,generate)]:
        model,tokenizer=load(path,"cuda")
        for row in rows:
            # Same temperature/seed/budget; algorithm changed and is named.
            options = {"repetition_penalty":1.,"no_repeat_ngram":0} if key == "after_raw" else {}
            result=fn(model,tokenizer,row["prompt"],max_new_tokens=96,temperature=0.5,seed=args.seed,precision="bf16",**options)
            encoded_prompt = row["prompt"] if key == "before" else row["prompt"].rstrip()
            tokens=result["tokens"][len(tokenizer.encode(encoded_prompt).ids)+1:]
            triples=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
            row[key]={"text":result["text"],"generated_tokens":len(tokens),"latency_ms":result["latency_ms"],
                      "repeated_trigram_fraction":1-len(set(triples))/len(triples) if triples else 0}
            print(key,repr(row["prompt"]),result["text"],flush=True)
        del model
        torch.cuda.empty_cache()
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"seed":args.seed,"temperature":0.5,"max_new_tokens":96,
        "before_mode":"blockwise diffusion","after_mode":"autoregressive, top_p=.9, repetition_penalty=1.12, no_repeat_ngram=4",
        "after_raw_mode":"autoregressive, same top-p, without repetition penalties",
        "limitations":"Diagnostic prompts, including user failure; not a blinded human evaluation. Seeds are not comparable random streams across algorithms.",
        "examples":rows},indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":main()
