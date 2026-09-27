import argparse


def main():
    p = argparse.ArgumentParser(description="DiffuThink v2 — from scratch")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", default="data/processed/stories-v2")
    prep.add_argument("--revision", default="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")
    prep.add_argument("--tokenizer", help="Reuse our existing BPE vocabulary for weight continuity")
    for key, value in [("stories",100000),("heldout",1000),("tokenizer-stories",50000),("vocab-size",8192),("length",192)]:
        prep.add_argument("--"+key, type=int, default=value)
    train = sub.add_parser("train")
    train.add_argument("--data", default="data/processed/stories-v2")
    train.add_argument("--output", default="runs/stories-v2")
    train.add_argument("--resume")
    train.add_argument("--initialize-from",help="Start a new optimization phase from our own trained weights")
    train.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    train.add_argument("--precision", choices=["fp32","bf16"], default="bf16")
    train.add_argument("--lr", type=float, default=0.0006)
    for key, value in [("steps",6000),("batch-size",16),("accumulation",2),("width",320),("heads",8),("layers",6),
                       ("eval-every",500),("eval-samples",128),("test-samples",512),("threads",4),("seed",42),("warmup",300)]:
        train.add_argument("--"+key,type=int,default=value)
    sample = sub.add_parser("sample")
    sample.add_argument("--model", default="runs/stories-512/best")
    sample.add_argument("--generation-mode", choices=["auto", "causal", "diffusion"], default="auto")
    sample.add_argument("--prompt", default="Once upon a time, a little girl found")
    sample.add_argument("--suffix", default=None)
    sample.add_argument("--missing-tokens", type=int, default=3)
    sample.add_argument("--new-tokens", type=int, default=64)
    sample.add_argument("--finish-sentence-tokens",type=int,default=0,help="Optional extra causal tokens (0..64) to reach terminal punctuation")
    sample.add_argument("--block-size", type=int, default=16)
    sample.add_argument("--steps", type=int, default=16)
    sample.add_argument("--temperature", type=float, default=0.5)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--model",default="runs/stories-v2-refined/best")
    benchmark.add_argument("--data",default="data/processed/stories-v2")
    benchmark.add_argument("--output",default="reports/v2/sampling_validation.json")
    benchmark.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    benchmark.add_argument("--precision",choices=["fp32","bf16"],default="bf16")
    benchmark.add_argument("--split",choices=["validation","test"],default="validation")
    benchmark.add_argument("--examples",type=int,default=64)
    serve = sub.add_parser("serve")
    serve.add_argument("--model",default="runs/stories-512/best")
    serve.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    serve.add_argument("--port",type=int,default=7861)
    export = sub.add_parser("export")
    export.add_argument("--model",default="runs/stories-512/best")
    export.add_argument("--report",default="runs/stories-512/report.json")
    export.add_argument("--output",default="artifacts/DiffuThink-Story-512")
    export.add_argument("--comparison",help="Optional measured comparison JSON bound to these weights")
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args)
    elif args.command == "train":
        from .train import train as run
        run(args)
    elif args.command == "benchmark":
        from .benchmark import run
        run(args)
    elif args.command == "serve":
        from .demo import serve as run
        run(args)
    elif args.command == "export":
        from .export import export as run
        run(args)
    else:
        import json
        import torch
        from .inference import load, continue_text, infill
        torch.set_num_threads(4)
        model, tokenizer = load(args.model, args.device)
        options = dict(steps=args.steps, temperature=args.temperature, seed=args.seed)
        if args.suffix is None:
            from pathlib import Path
            info = json.loads((Path(args.model)/"training_info.json").read_text())
            use_causal = args.generation_mode == "causal" or (args.generation_mode == "auto" and info.get("continuation_mode") == "autoregressive")
            if use_causal:
                if info.get("continuation_mode") != "autoregressive":
                    raise ValueError("This checkpoint was not trained for causal generation")
                from .hybrid import generate
                result = generate(model, tokenizer, args.prompt, max_new_tokens=args.new_tokens,
                                  finish_sentence_tokens=args.finish_sentence_tokens, **options)
            else:
                result = continue_text(model, tokenizer, args.prompt, max_new_tokens=args.new_tokens, block_size=args.block_size, **options)
        else:
            result = infill(model, tokenizer, args.prompt, args.suffix, missing_tokens=args.missing_tokens, **options)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
