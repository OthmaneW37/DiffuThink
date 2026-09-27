"""Small original synthetic corpus and leakage-aware line splitting."""
import hashlib
import random
from pathlib import Path
import torch
from .model import PAD, encode


def make_corpus(path):
    subjects = ["The cat", "The dog", "The bird", "The rabbit", "The child", "Alice", "Bob", "The teacher"]
    verbs = ["rests", "plays", "waits", "walks", "sits", "runs"]
    places = ["in the garden", "near the house", "by the river", "under the tree", "in the park", "near the school"]
    times = ["today", "at noon", "every day", "in the morning"]
    lines = [f"{s} {v} {p} {t}." for s in subjects for v in verbs for p in places for t in times]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def load_data(path, length, seed):
    raw = Path(path).read_bytes()
    lines = sorted(set(s.strip() for s in raw.decode("utf-8").splitlines() if s.strip()))
    if len(lines) < 20:
        raise ValueError("Need at least 20 unique nonempty lines")
    longest = max(len(encode(s)) for s in lines)
    if longest > length:
        raise ValueError(f"Longest line is {longest} bytes; increase --length (no silent truncation)")
    random.Random(seed).shuffle(lines)
    n = max(1, len(lines) // 10)
    splits = {"test": lines[:n], "validation": lines[n:2*n], "train": lines[2*n:]}
    tensors = {name: torch.tensor([encode(s) + [PAD] * (length - len(encode(s))) for s in part]) for name, part in splits.items()}
    manifest = {"sha256": hashlib.sha256(raw).hexdigest(), "split_seed": seed,
                "counts": {k: len(v) for k, v in splits.items()},
                "split_hashes": {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in splits.items()}}
    return tensors, manifest
