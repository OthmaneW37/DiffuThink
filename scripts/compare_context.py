"""Compare the 192-token release with the expanded corpus/context checkpoint.

Likelihood uses identical legacy windows. Samples have identical decoding budgets.
The wider-context test report is separate and must not be compared as if windowing
were unchanged. No automatic semantic-coherence score is claimed.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from diffuthink.v2.inference import load
from diffuthink.v2.hybrid import causal_evaluate, generate, sentence_ended
from diffuthink.v2.data import digest

# Fixed before examining the new checkpoint. First prompt is the known user case.
PROMPTS = [
    "One day a man was under a tree and ",
    "Mia put her blue cup on the table. When she came back,",
    "Sam could not swim. He stood beside the deep pond and",
    "The little bird hurt its wing. It could not fly, so",
    "Lucy gave her last cookie to her brother. Now she",
    "It was raining outside. The children decided to stay inside and",
    "The old man reached the top of the hill. He sat down to rest and",
    "Ben lost his red hat in the garden. His sister helped him",
    "The box was empty. Lily looked inside and",
    "A small dog was afraid of the loud thunder. Its owner",
    "Tom promised to return the toy before dinner. When the sun went down,",
    "Anna planted a seed and watered it every day. After a few weeks,",
    "The girl could not reach the shelf. She asked her father to",
    "A rabbit found a carrot. Instead of eating it alone,",
    "Max broke his friend's toy by accident. He felt sorry and",
    "The snow melted in the warm sun. The children",
    "Two friends built a small boat. They put it on the water and",
    "Sara heard a kitten crying behind the fence. She",
    "The baker had no flour left. To make more bread, he",
    "A boy found a lost key on the path. He wanted to",
    "The little bear was tired after a long walk. When he got home,",
    "Dad turned off the light. The room became dark, and",
    "Nora finished her drawing. She showed it to her mother, who",
    "The children cleaned up all their toys. At last,",
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--before",default="runs/stories-hybrid/best")
    p.add_argument("--after",default="runs/stories-512/best")
    p.add_argument("--output",default="reports/context512/comparison.json")
    p.add_argument("--device",default="cuda",choices=["cpu","cuda"])
    p.add_argument("--phase",default="both",choices=["before","after","both"])
    args=p.parse_args()
    torch.set_num_threads(4)
    data=Path("data/processed/stories-v2/test.npy")
    array=np.load(data,mmap_mode="r")
    rows=[{"prompt":prompt,"seed":seed} for prompt in PROMPTS for seed in (42,7)]
    result={"protocol":{"common_test_array_sha256":digest(data),"common_test_windows":len(array),
        "temperature":.5,"max_new_tokens":160,"finish_sentence_tokens":0,
        "seeds":[42,7],"prompt_count":len(PROMPTS),"device":args.device,
        "before_weights_sha256":digest(Path(args.before)/"model.safetensors"),
        "latency_note":"Timings are not a speed benchmark; other GPU work may run concurrently.",
        "note":"First prompt is a known diagnostic. Other prompts fixed before inspecting new weights. All raw samples retained. No blind human coherence assessment."},
        "likelihood":{},"summary":{},"weights":{},"examples":rows}
    precision="bf16" if args.device=="cuda" else "fp32"
    if args.phase=="after":
        previous=json.loads(Path(args.output).read_text(encoding="utf-8"))
        if previous["protocol"]!=result["protocol"]:
            raise ValueError("Baseline protocol differs from this run")
        if [(r["prompt"],r["seed"]) for r in previous["examples"]]!=[(r["prompt"],r["seed"]) for r in rows]:
            raise ValueError("Prompt list differs")
        result=previous;rows=result["examples"]
    for label,path in (("before",args.before),("after",args.after)):
        if args.phase!="both" and args.phase!=label:continue
        model,tokenizer=load(path,args.device)
        result["weights"][label]=digest(Path(path)/"model.safetensors")
        info=json.loads((Path(path)/"training_info.json").read_text(encoding="utf-8"))
        legacy=json.loads(Path("data/processed/stories-v2/manifest.json").read_text(encoding="utf-8"))
        if info["manifest"]["tokenizer_sha256"]!=legacy["tokenizer_sha256"]:
            raise ValueError("Cannot compare incompatible tokenizations")
        result["likelihood"][label]=causal_evaluate(model,array,limit=len(array),precision=precision)
        for i,row in enumerate(rows):
            sample=generate(model,tokenizer,row["prompt"],max_new_tokens=160,temperature=.5,
                            seed=row["seed"],precision=precision,finish_sentence_tokens=0)
            ids=sample["tokens"][len(tokenizer.encode(row["prompt"].rstrip()).ids)+1:]
            grams=[tuple(ids[j:j+3]) for j in range(max(0,len(ids)-2))]
            row[label]={k:sample[k] for k in ("text","stop_reason","generated_tokens","latency_ms")}
            row[label]["repeated_trigram_fraction"]=1-len(set(grams))/len(grams) if grams else 0
            row[label]["ends_with_punctuation"]=sentence_ended(sample["text"])
            print(f"{label}: {i+1}/{len(rows)}",flush=True)
        result["summary"][label]={
            "mean_repeated_trigram_fraction":sum(r[label]["repeated_trigram_fraction"] for r in rows)/len(rows),
            "eos_rate":sum(r[label]["stop_reason"]=="eos" for r in rows)/len(rows),
            "punctuated_ending_rate":sum(r[label]["ends_with_punctuation"] for r in rows)/len(rows),
            "mean_generated_tokens":sum(r[label]["generated_tokens"] for r in rows)/len(rows)}
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    print(json.dumps({"likelihood":result["likelihood"],"summary":result["summary"]},indent=2))


if __name__=="__main__":main()
