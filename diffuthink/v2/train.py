"""Resumable GPU training with fixed validation masks and atomic recovery state."""
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from .data import load_arrays
from .model import Config, Denoiser, PAD, BOS, corrupt


def amp(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" and precision == "bf16" else nullcontext()


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, array, frequencies, *, device, precision, limit=256, batch_size=8):
    model.eval()
    reports = []
    for rate in (0.15, 0.5, 0.85):
        rng = torch.Generator(device=device).manual_seed(2718)
        total = correct = top5 = 0
        loss = baseline_loss = 0.
        baseline_correct = 0
        for start in range(0, min(limit, len(array)), batch_size):
            clean = torch.tensor(np.array(array[start:min(start+batch_size,limit)]), dtype=torch.long, device=device)
            noisy, selected, noise = corrupt(clean, rng, mode="random", rate=rate)
            with amp(device, precision):
                logits = model(noisy, noise, selected)
            targets = clean[selected]
            logits = logits.float()
            loss += F.cross_entropy(logits, targets, reduction="sum").item()
            correct += logits.argmax(-1).eq(targets).sum().item()
            top5 += logits.topk(5, -1).indices.eq(targets[:, None]).any(-1).sum().item()
            total += targets.numel()
            baseline_loss += -frequencies.to(device)[targets].log().sum().item()
            baseline_correct += targets.eq(int(frequencies.argmax())).sum().item()
        reports.append({"mask_rate": rate, "targets": total, "cross_entropy": loss / total,
                        "accuracy": correct / total, "top5_accuracy": top5 / total,
                        "unigram_cross_entropy": baseline_loss / total, "unigram_accuracy": baseline_correct / total})
    return reports


def train(args):
    if min(args.steps, args.batch_size, args.accumulation, args.eval_every, args.threads, args.eval_samples) < 1:
        raise ValueError("Training counts must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA/ROCm unavailable; choose --device cpu")
    if args.precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported; choose --precision fp32")
    arrays, manifest = load_arrays(args.data)
    config = Config(vocab_size=manifest["vocab_size"], width=args.width, heads=args.heads, layers=args.layers,
                    max_length=manifest["max_length"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise ValueError("Run already exists; use --resume or a new output directory")
    run_config = {k: v for k, v in vars(args).items() if k not in ("resume", "command")}
    model = Denoiser(config).to(device)
    lineage = None
    if getattr(args, "initialize_from", None):
        if args.resume:
            raise ValueError("Choose resume OR initialize-from")
        source = Path(args.initialize_from)
        lineage = json.loads((source / "training_info.json").read_text())
        if lineage["manifest"] != manifest or lineage.get("pretrained_weights") is not False:
            raise ValueError("Initialization must use this corpus and a from-scratch DiffuThink checkpoint")
        prior = Denoiser.from_pretrained(source)
        if prior.config != config:
            raise ValueError("Initialization model configuration differs")
        model.load_state_dict(prior.state_dict())
        del prior
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sampling = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    counts = np.bincount(np.asarray(arrays["train"]).reshape(-1), minlength=config.vocab_size).astype(np.float64) + 1
    counts[[PAD, BOS]] = 1
    frequencies = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    step, seen_tokens, elapsed, best = 0, 0, 0., float("inf")
    if args.resume:
        if not (out / "best" / "model.safetensors").exists():
            raise ValueError("Resume in the original output directory containing best/ and last.pt")
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["manifest"] != manifest or state["model_config"] != asdict(config):
            raise ValueError("Resume data/model mismatch")
        for key in ("seed", "batch_size", "accumulation", "lr", "steps", "warmup", "precision", "device"):
            if state["run_config"][key] != run_config[key]:
                raise ValueError(f"Resume requires unchanged {key}; schedule must remain identical")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampling.set_state(state["sampling_rng"])
        noise_rng.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        step, seen_tokens, elapsed, best = state["step"], state["seen_tokens"], state["elapsed"], state["best"]
        lineage = state.get("lineage")
        del state
    else:
        initial = evaluate(model, arrays["validation"], frequencies, device=device, precision=args.precision, limit=args.eval_samples)
        (out / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    start = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"parameters": parameters, "device": device, "precision": args.precision, "start_step": step, "target_steps": args.steps}), flush=True)
    log = out / "metrics.jsonl"
    def save_state():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "model_config": asdict(config), "manifest": manifest, "run_config": run_config,
                     "step": step, "seen_tokens": seen_tokens, "elapsed": elapsed + time.perf_counter() - start,
                     "best": best, "sampling_rng": sampling.get_state(), "noise_rng": noise_rng.get_state(),
                     "lineage": lineage,
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None}, out / "last.pt")
    try:
        while step < args.steps:
            model.train()
            lr_scale = min(1., (step + 1) / max(1, args.warmup))
            progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
            lr = args.lr * lr_scale * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for _ in range(args.accumulation):
                indices = torch.randint(len(arrays["train"]), (args.batch_size,), generator=sampling).numpy()
                clean = torch.tensor(np.array(arrays["train"][indices]), dtype=torch.long, device=device)
                noisy, selected, noise = corrupt(clean, noise_rng)
                with amp(device, args.precision):
                    logits = model(noisy, noise, selected)
                    loss = F.cross_entropy(logits.float(), clean[selected])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; previous recovery checkpoint is preserved")
                (loss / args.accumulation).backward()
                loss_value += loss.item() / args.accumulation
                seen_tokens += int(clean.ne(PAD).sum())
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise RuntimeError("Non-finite gradient; previous checkpoint is preserved")
            optimizer.step()
            step += 1
            if step == 1 or step % 25 == 0:
                wall = elapsed + time.perf_counter() - start
                row = {"step": step, "loss": loss_value, "lr": lr, "gradient_norm": float(gradient),
                       "seen_tokens": seen_tokens, "seconds": wall, "tokens_per_second": seen_tokens / wall}
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, arrays["validation"], frequencies, device=device, precision=args.precision, limit=args.eval_samples)
                score = sum(r["cross_entropy"] for r in metrics) / len(metrics)
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "validation": metrics, "validation_loss": score}) + "\n")
                if score < best:
                    best = score
                    model.save_pretrained(out / "best")
                    shutil.copy2(Path(args.data) / "tokenizer.json", out / "best" / "tokenizer.json")
                    (out / "best" / "training_info.json").write_text(json.dumps({"step": step, "validation_loss": score,
                        "parameters": parameters, "seen_tokens": seen_tokens, "pretrained_weights": False,
                        "manifest": manifest, "previous_phase": lineage}, indent=2), encoding="utf-8")
                save_state()
                print(json.dumps({"step": step, "validation_loss": score, "best": best}), flush=True)
    except KeyboardInterrupt:
        # An interruption in a partial step is not an exact resume point.
        print("Interrupted. Resume from the previous completed evaluation checkpoint.", flush=True)
        return
    model = Denoiser.from_pretrained(out / "best", device)
    report = {"status": "completed", "model_config": asdict(config), "parameters": parameters,
              "training": run_config, "seen_tokens": seen_tokens, "seconds": elapsed + time.perf_counter() - start,
              "torch_version": str(torch.__version__), "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "best": json.loads((out / "best" / "training_info.json").read_text()),
              "test": evaluate(model, arrays["test"], frequencies, device=device, precision=args.precision, limit=args.test_samples),
              "limitations": ["English synthetic stories, not a general chatbot", "No comparison to v1 byte accuracy: tokenizer and corpus differ",
                              "Custom masked-denoising objective, not a claimed diffusion likelihood bound"]}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training complete: {out / 'report.json'}", flush=True)
