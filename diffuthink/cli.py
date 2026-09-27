import argparse
import json
import torch
from .data import make_corpus
from .engine import train, evaluate_checkpoint, load_checkpoint
from .model import generate


def main():
    parser = argparse.ArgumentParser(description="Train and explain a small diffusion text model")
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("prepare", help="Generate the original controlled English grammar corpus")
    corpus.add_argument("--output", default="data/synthetic.txt")
    training = sub.add_parser("train")
    training.add_argument("--data", default="data/synthetic.txt")
    training.add_argument("--output", default="runs/demo")
    for name, default in [("steps", 600), ("batch-size", 32), ("eval-every", 100), ("seed", 42), ("width", 64), ("heads", 4), ("layers", 2), ("length", 64), ("threads", 4)]:
        training.add_argument("--" + name, type=int, default=default)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", default="data/synthetic.txt")
    sampling = sub.add_parser("sample")
    sampling.add_argument("--template", default="The cat ~~~~~ in the garden today.")
    sampling.add_argument("--steps", type=int, default=12)
    sampling.add_argument("--temperature", type=float, default=0.0)
    sampling.add_argument("--seed", type=int, default=42)
    serving = sub.add_parser("serve")
    serving.add_argument("--port", type=int, default=7860)
    for cmd in (evaluation, sampling, serving):
        cmd.add_argument("--checkpoint", default="runs/demo/best.pt")
        cmd.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        cmd.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        print(f"Wrote {make_corpus(args.output)} unique sentences to {args.output}")
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    else:
        torch.set_num_threads(args.threads)
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if args.command == "sample":
            print(json.dumps(generate(model, args.template, args.steps, args.temperature, args.seed), indent=2, ensure_ascii=False))
        else:
            from .demo import serve
            serve(model, checkpoint, args.port)


if __name__ == "__main__":
    main()
