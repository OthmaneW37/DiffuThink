"""Confidence-guided subword denoising and blockwise continuation."""
import math
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from .model import Denoiser, PAD, BOS, EOS, MASK, UNK
from .train import amp


def load(directory, device="cpu"):
    return Denoiser.from_pretrained(directory, device), Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))


@torch.inference_mode()
def denoise(model, tokenizer, ids, *, steps=16, temperature=0.7, seed=42, top_k=40, precision="fp32", policy="confidence", threshold=0.85, allow_eos=True):
    if not ids or len(ids) > model.config.max_length:
        raise ValueError(f"Input exceeds {model.config.max_length} tokens")
    if steps < 1 or not math.isfinite(temperature) or temperature < 0 or top_k < 1:
        raise ValueError("Invalid sampling parameters")
    if policy not in ("confidence", "left_to_right", "adaptive") or not 0 < threshold <= 1:
        raise ValueError("Invalid policy or threshold")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in 0..2^63-1")
    device = str(next(model.parameters()).device).split(":")[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.tensor([ids], device=device)
    fixed = tokens.ne(MASK)
    count = int((~fixed).sum())
    iterations = min(steps, count)
    trace = []
    model.eval()
    started = time.perf_counter()
    for step in range(iterations):
        unknown = tokens.eq(MASK)
        rate = unknown.sum(-1).float() / (tokens.ne(PAD) & tokens.ne(BOS)).sum(-1).clamp_min(1)
        with amp(device, precision):
            logits = model(tokens, rate)[0].float()
        logits[:, [PAD, BOS, MASK, UNK]] = -torch.inf
        if not allow_eos:
            logits[:, EOS] = -torch.inf
        # Ranking always uses the untempered distribution. Using T≈0 here
        # would turn every argmax confidence into 1 and destroy its ordering.
        confidence_probs = logits.softmax(-1)
        if temperature == 0:
            candidates = logits.argmax(-1)
        else:
            values, indices = logits.topk(min(top_k, logits.shape[-1]), -1)
            choice = torch.multinomial((values / temperature).softmax(-1), 1, generator=generator)
            candidates = indices.gather(-1, choice).squeeze(-1)
        confidence = confidence_probs.gather(-1, candidates[:, None]).squeeze(-1).masked_fill(~unknown[0], -1)
        # Cosine schedule keeps more holes early and commits all by the last pass.
        remaining = int(unknown.sum())
        target = min(remaining - 1, math.floor(count * math.cos((step + 1) / iterations * math.pi / 2)))
        if step == iterations - 1:
            target = 0
        commit = remaining - max(0, target)
        if policy == "adaptive":
            commit = max(commit, int((confidence >= threshold).sum()))
        ranking = confidence if policy != "left_to_right" else -torch.arange(len(ids), device=device).float().masked_fill(~unknown[0], float("inf"))
        positions = ranking.topk(commit).indices
        tokens[0, positions] = candidates[positions]
        trace.append({"step": step + 1, "remaining": int(tokens.eq(MASK).sum()),
                      "text": tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)})
        if not tokens.eq(MASK).any():
            break
    result = tokens[0].tolist()
    return {"text": tokenizer.decode(result), "tokens": result, "trace": trace,
            "forward_passes": len(trace), "latency_ms": (time.perf_counter() - started) * 1000,
            "seed": seed}


def infill(model, tokenizer, prefix, suffix, missing_tokens=3, **kwargs):
    if not 1 <= missing_tokens <= 64:
        raise ValueError("missing_tokens must be 1..64")
    ids = [BOS] + tokenizer.encode(prefix).ids + [MASK] * missing_tokens + tokenizer.encode(suffix).ids + [EOS]
    return denoise(model, tokenizer, ids, **kwargs)


def continue_text(model, tokenizer, prompt, max_new_tokens=64, block_size=16, **kwargs):
    if not 1 <= max_new_tokens <= 256 or not 1 <= block_size <= 64:
        raise ValueError("max_new_tokens: 1..256; block_size: 1..64")
    context = [BOS] + tokenizer.encode(prompt).ids
    if len(context) + min(block_size, max_new_tokens) > model.config.max_length:
        raise ValueError("Prompt too long for this model context")
    output, trace, passes, latency = list(context), [], 0, 0.
    base_seed = kwargs.pop("seed", 42)
    for offset in range(0, max_new_tokens, block_size):
        length = min(block_size, max_new_tokens - offset)
        # Keep BOS as a boundary when a continuation exceeds the context window.
        window = [BOS] + output[1:][-(model.config.max_length - length - 1):]
        result = denoise(model, tokenizer, window + [MASK] * length, seed=base_seed + offset, **kwargs)
        new = result["tokens"][-length:]
        passes += result["forward_passes"]
        latency += result["latency_ms"]
        trace.extend({**row, "block": offset // block_size + 1} for row in result["trace"])
        if EOS in new:
            output.extend(new[:new.index(EOS)])
            break
        output.extend(new)
    return {"text": tokenizer.decode(output), "tokens": output, "trace": trace, "forward_passes": passes,
            "latency_ms": latency, "seed": base_seed}
