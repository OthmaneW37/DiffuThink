from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from diffuthink.model import Block, RMSNorm

PAD, BOS, EOS, MASK, UNK = range(5)
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"]


@dataclass
class Config:
    vocab_size: int = 8192
    width: int = 320
    heads: int = 8
    layers: int = 6
    max_length: int = 192
    architecture: str = "diffuthink_v2"

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.max_length) < 1:
            raise ValueError("Dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("Vocabulary too small")


class Denoiser(nn.Module):
    """Original small Transformer, initialized randomly, with tied embeddings.

    Learned absolute positions supplement RoPE so all-masked sequences have
    position-specific information. No external model weights are used.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        self.time_mlp = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.out.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, noise, selected=None, causal=False, right_padded=False):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds configured context")
        half = self.config.width // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=tokens.device) / max(half - 1, 1))
        angle = noise[:, None] * 1000 * freq
        t = self.time_mlp(torch.cat((angle.sin(), angle.cos()), -1))
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x, t, None if causal and right_padded else tokens.ne(PAD), causal=causal)
        x = self.norm(x)
        # Training projects only supervised positions, reducing vocabulary memory.
        if selected is not None:
            x = x[selected]
        return F.linear(x, self.embedding.weight)

    def save_pretrained(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        temporary = directory / "model.safetensors.tmp"
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, str(temporary))
        temporary.replace(directory / "model.safetensors")

    def extend_context(self, length):
        """Keep learned positions exactly; initialize new positions near their mean."""
        if length < self.config.max_length:
            raise ValueError("Cannot shrink context during extension")
        if length == self.config.max_length:
            return
        old = self.position.weight.detach()
        position = nn.Embedding(length, self.config.width, device=old.device, dtype=old.dtype)
        with torch.no_grad():
            nn.init.normal_(position.weight, std=0.02)
            position.weight.add_(old.mean(0))
            position.weight[:len(old)].copy_(old)
        self.position = position
        self.config.max_length = length

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        from safetensors.torch import load_file
        directory = Path(directory)
        model = cls(Config(**json.loads((directory / "config.json").read_text(encoding="utf-8"))))
        model.load_state_dict(load_file(str(directory / "model.safetensors")))
        return model.to(device).eval()


def corrupt(clean, rng, mode="mixed", rate=None):
    """Mix random holes and continuation masks. BOS/PAD are never targets."""
    valid = clean.ne(PAD) & clean.ne(BOS)
    b, n = clean.shape
    if not valid.any(-1).all():
        raise ValueError("All examples need a target")
    noise = 0.05 + 0.95 * torch.rand(b, device=clean.device, generator=rng) if rate is None else torch.full((b,), rate, device=clean.device)
    selected = (torch.rand(clean.shape, device=clean.device, generator=rng) < noise[:, None]) & valid
    if mode == "mixed":
        suffix_rows = torch.rand(b, device=clean.device, generator=rng) < 0.5
        lengths = valid.sum(-1)
        starts = 1 + (torch.rand(b, device=clean.device, generator=rng) * lengths).long()
        suffix = (torch.arange(n, device=clean.device)[None] >= starts[:, None]) & valid
        selected = torch.where(suffix_rows[:, None], suffix, selected)
    elif mode != "random":
        raise ValueError("Unknown corruption mode")
    empty = ~selected.any(-1)
    fallback = torch.rand(clean.shape, device=clean.device, generator=rng).masked_fill(~valid, -1).argmax(-1)
    selected[empty, fallback[empty]] = True
    # Training and inference use the same meaning of t: actual masked fraction.
    noise = selected.sum(-1).float() / valid.sum(-1)
    return clean.masked_fill(selected, MASK), selected, noise
