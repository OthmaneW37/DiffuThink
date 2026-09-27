"""Training, deterministic held-out evaluation, and portable checkpoints."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from .data import load_data
from .model import DiffuThink, ModelConfig, PAD, corrupt, masked_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DiffuThink(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(model, examples, frequencies, seed=1234, batch_size=32):
    device = next(model.parameters()).device
    model.eval()
    rows = []
    for rate in (0.25, 0.5, 0.75, 1.0):
        rng = torch.Generator(device=device).manual_seed(seed)
        loss_sum, correct, baseline_loss, baseline_correct, count = 0., 0, 0., 0, 0
        for clean in examples.split(batch_size):
            clean = clean.to(device)
            noisy, selected = corrupt(clean, torch.full((len(clean),), rate, device=device), rng)
            logits = model(noisy, torch.full((len(clean),), rate, device=device))[selected]
            targets = clean[selected]
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            correct += logits.argmax(-1).eq(targets).sum().item()
            baseline_loss += -frequencies.to(device)[targets].log().sum().item()
            baseline_correct += targets.eq(int(frequencies.argmax())).sum().item()
            count += targets.numel()
        rows.append({"mask_rate": rate, "masked_bytes": count, "cross_entropy": loss_sum / count,
                     "accuracy": correct / count, "unigram_cross_entropy": baseline_loss / count,
                     "unigram_accuracy": baseline_correct / count})
    return rows


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.lr <= 0:
        raise ValueError("steps, batch size, eval interval and learning rate must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(args.width, args.heads, args.layers, args.length)
    data, manifest = load_data(args.data, cfg.max_length, args.seed)
    model = DiffuThink(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Laplace smoothing: the baseline assigns nonzero probability to every byte.
    counts = torch.bincount(data["train"][data["train"].ne(PAD)], minlength=256).float() + 1
    frequencies = counts / counts.sum()
    sampling_rng = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    history, best = [], float("inf")
    start = time.perf_counter()
    initial = evaluate(model, data["validation"], frequencies)
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(len(data["train"]), (args.batch_size,), generator=sampling_rng)
        clean = data["train"][indices].to(device)
        noise = 0.05 + 0.95 * torch.rand(len(clean), device=device, generator=noise_rng)
        noisy, selected = corrupt(clean, noise, noise_rng)
        loss = masked_loss(model(noisy, noise), clean, selected)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, data["validation"], frequencies)
            score = sum(row["cross_entropy"] for row in metrics) / len(metrics)
            entry = {"step": step, "train_loss": loss.item(), "validation_loss": score,
                     "gradient_norm": float(grad_norm), "seconds": time.perf_counter() - start}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            checkpoint = {"config": asdict(cfg), "model": model.state_dict(), "step": step,
                          "manifest": manifest, "frequencies": frequencies, "seed": args.seed}
            # Inference checkpoints intentionally omit optimizer state; no resume claim.
            torch.save(checkpoint, output / "last.pt")
            if score < best:
                best = score
                torch.save(checkpoint, output / "best.pt")
            write_json(output / "history.json", history)
    model, checkpoint = load_checkpoint(output / "best.pt", device)
    report = {"task": "Held-out byte reconstruction on deduplicated line splits",
              "config": asdict(cfg), "parameters": sum(p.numel() for p in model.parameters()),
              "device": device, "torch": str(torch.__version__), "python": platform.python_version(),
              "data": manifest, "training": {"seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
              "learning_rate": args.lr, "threads": args.threads, "best_step": checkpoint["step"], "seconds": time.perf_counter() - start},
              "initial_validation": initial, "final_validation": evaluate(model, data["validation"], frequencies),
              "test": evaluate(model, data["test"], frequencies), "history": history,
              "limitations": ["Synthetic grammar; same grammar shared across splits", "Fixed length byte infilling; no EOS or chat instruction training",
                              "Masked reconstruction CE is not autoregressive perplexity", "Unigram is a weak baseline; no claim against pretrained LMs"]}
    write_json(output / "report.json", report)
    print(f"Saved {output / 'best.pt'} and {output / 'report.json'}", flush=True)


def evaluate_checkpoint(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    data, manifest = load_data(args.data, model.config.max_length, checkpoint["seed"])
    if manifest != checkpoint["manifest"]:
        raise ValueError("Dataset differs from the training manifest; evaluation aborted")
    result = evaluate(model, data["test"], checkpoint["frequencies"])
    print(json.dumps(result, indent=2))
