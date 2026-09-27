"""Sampler ablations on fixed held-out subword masks; no selection on test."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import load_arrays
from .inference import load, denoise, continue_text
from .model import PAD, BOS, EOS, corrupt


def run(args):
    if args.examples < 1:
        raise ValueError("examples must be positive")
    torch.set_num_threads(4)
    model, tokenizer = load(args.model, args.device)
    arrays, manifest = load_arrays(args.data)
    info = json.loads((Path(args.model) / "training_info.json").read_text())
    if info["manifest"] != manifest:
        raise ValueError("Benchmark corpus differs from training")
    # Validation is the default while developing sampling policies.
    array = arrays[args.split][:args.examples]
    clean = torch.tensor(np.array(array), dtype=torch.long)
    noisy, selected, _ = corrupt(clean, torch.Generator().manual_seed(314), mode="random", rate=0.5)
    schedules = [("one_pass", 1, "confidence"), ("left_to_right", 12, "left_to_right"),
                 ("confidence", 12, "confidence"), ("adaptive", 12, "adaptive")]
    denoise(model, tokenizer, [BOS, 3, EOS], temperature=0, precision=args.precision)
    results = []
    for name, steps, policy in schedules:
        correct = total = exact = passes = 0
        latencies = []
        for truth, source, mask in zip(clean, noisy, selected):
            size = int(truth.ne(PAD).sum())
            result = denoise(model, tokenizer, source[:size].tolist(), steps=steps, temperature=0,
                             policy=policy, precision=args.precision)
            prediction = torch.tensor(result["tokens"])
            correct += int(prediction[mask[:size]].eq(truth[:size][mask[:size]]).sum())
            total += int(mask.sum())
            exact += int(prediction.eq(truth[:size]).all())
            passes += result["forward_passes"]
            latencies.append(result["latency_ms"])
        results.append({"policy": name, "masked_accuracy": correct/total, "exact_match": exact/len(clean),
                        "mean_forward_passes": passes/len(clean), "mean_latency_ms": float(np.mean(latencies)),
                        "p95_latency_ms": float(np.percentile(latencies,95)), "masked_tokens": total})
    prompts = ["Once upon a time, a little girl found", "Tom opened the door and saw", "The woman stood on the balcony and",
               "A small bird was afraid to fly. One day,"]
    samples = [{"prompt": prompt, **continue_text(model,tokenizer,prompt,max_new_tokens=64,steps=12,
                temperature=0.7,seed=42,precision=args.precision)} for prompt in prompts]
    from .baselines import bigram_accuracy
    baseline = bigram_accuracy(arrays["train"], noisy.numpy(), clean.numpy(), selected.numpy(), model.config.vocab_size)
    report = {"split": args.split, "examples": len(clean), "mask_seed":314,"mask_rate":0.5,"temperature":0,
              "device":args.device,"precision":args.precision,"checkpoint_step":info["step"],
              "tokenizer_sha256":manifest["tokenizer_sha256"],"results":results,"bigram":baseline,"samples":samples}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)
