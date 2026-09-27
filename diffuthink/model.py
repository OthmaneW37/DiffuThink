"""Bidirectional Transformer: RMSNorm, RoPE, SwiGLU and noise conditioning.

These are packaged versions of the concepts explored in experiments/.
No pretrained weights or high-level Transformer implementation are used.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, MASK = 256, 257
VOCAB_SIZE = 258


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens) -> str:
    return bytes(int(t) for t in tokens if 0 <= int(t) < 256).decode("utf-8", errors="replace")


@dataclass
class ModelConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    max_length: int = 64

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.max_length) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width / heads must be an even integer for RoPE")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def rope(x):
    length, dim = x.shape[-2:]
    freq = 10000 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = torch.outer(torch.arange(length, device=x.device), freq)
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w = cfg.width
        self.heads = cfg.heads
        self.norm1, self.norm2 = RMSNorm(w), RMSNorm(w)
        self.time = nn.Linear(w, w, bias=False)
        self.qkv = nn.Linear(w, 3 * w, bias=False)
        self.out = nn.Linear(w, w, bias=False)
        self.gate = nn.Linear(w, 4 * w, bias=False)
        self.up = nn.Linear(w, 4 * w, bias=False)
        self.down = nn.Linear(4 * w, w, bias=False)

    def forward(self, x, time, valid, causal=False):
        x = x + self.time(time)[:, None]
        b, n, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q, k, v = [z.reshape(b, n, self.heads, w // self.heads).transpose(1, 2) for z in (q, k, v)]
        # Bidirectional: visible bytes on both sides can inform a missing byte.
        if causal and valid is None:
            # Caller guarantees right padding; supervised tokens cannot see future PADs.
            y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        else:
            attention_mask = valid[:, None, None, :]
            if causal:
                attention_mask = attention_mask & torch.ones(n, n, device=x.device, dtype=torch.bool).tril()
            y = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=attention_mask)
        x = x + self.out(y.transpose(1, 2).reshape(b, n, w))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class DiffuThink(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB_SIZE, config.width, padding_idx=PAD)
        self.time_mlp = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.head = nn.Linear(config.width, 256, bias=False)

    def forward(self, tokens, noise):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.max_length:
            raise ValueError("Expected [batch, length] within max_length")
        half = self.config.width // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=tokens.device) / max(half - 1, 1))
        angles = noise[:, None] * 1000 * freq
        time = self.time_mlp(torch.cat((angles.sin(), angles.cos()), -1))
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, time, tokens.ne(PAD))
        return self.head(self.norm(x))


def corrupt(clean, noise, generator=None):
    """Bernoulli masking, with at least one target per nonempty example."""
    valid = clean.ne(PAD)
    if not valid.any(-1).all():
        raise ValueError("Empty examples cannot be corrupted")
    selected = (torch.rand(clean.shape, device=clean.device, generator=generator) < noise[:, None]) & valid
    missing = ~selected.any(-1)
    # Choose a valid position uniformly, without assuming padding placement.
    fallback = torch.rand(clean.shape, device=clean.device, generator=generator).masked_fill(~valid, -1).argmax(-1)
    selected[missing, fallback[missing]] = True
    return clean.masked_fill(selected, MASK), selected


def masked_loss(logits, clean, selected):
    return F.cross_entropy(logits[selected], clean[selected])


@torch.inference_mode()
def generate(model, template: str, steps=12, temperature=0.8, seed=42):
    """Each ~ represents ONE missing byte. Commit most confident bytes first.

    The fixed context is never overwritten. This is a heuristic iterative
    unmasking sampler, not an exact reverse-diffusion likelihood sampler.
    """
    if steps < 1 or temperature < 0 or not math.isfinite(temperature):
        raise ValueError("steps must be positive and temperature finite and nonnegative")
    ids = [MASK if b == ord("~") else b for b in encode(template)]
    if not ids or len(ids) > model.config.max_length:
        raise ValueError(f"Template must contain 1..{model.config.max_length} UTF-8 bytes")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.tensor([ids], device=device)
    total = int(tokens.eq(MASK).sum())
    trace = []
    model.eval()
    for step in range(min(steps, total)):
        unknown = tokens.eq(MASK)
        remaining = int(unknown.sum())
        noise = torch.tensor([remaining / tokens.shape[1]], device=device)
        logits = model(tokens, noise)[0]
        probs = (logits / max(temperature, 1e-6)).softmax(-1)
        candidates = logits.argmax(-1) if temperature == 0 else torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        confidence = probs.gather(-1, candidates[:, None]).squeeze(-1).masked_fill(~unknown[0], -1)
        # Integer arithmetic avoids rounding a remaining count of 1 down to 0.
        iterations = min(steps, total)
        target_remaining = total * (iterations - step - 1) // iterations
        positions = confidence.topk(remaining - target_remaining).indices
        tokens[0, positions] = candidates[positions]
        trace.append({"step": step + 1, "remaining": int(tokens.eq(MASK).sum()), "text": display(tokens[0].tolist())})
    return {"text": decode(tokens[0].tolist()), "tokens": tokens[0].tolist(), "trace": trace, "seed": seed}


def display(ids):
    # Decode runs together so a valid UTF-8 character is not split for display.
    return b"".join(b"~" if t == MASK else bytes([t]) if t < 256 else b"" for t in ids).decode("utf-8", errors="replace")
