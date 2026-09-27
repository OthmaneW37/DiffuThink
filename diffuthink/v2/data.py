"""Pinned story download, document splits and a tokenizer trained on train only."""
import hashlib
import json
import shutil
from pathlib import Path
import requests
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .model import SPECIAL_TOKENS, PAD, BOS, EOS


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_stories(revision, filename, count, seen):
    url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/{filename}"
    stories, buffer = [], ""
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for chunk in response.iter_content(1024 * 128, decode_unicode=True):
            buffer += chunk
            while "<|endoftext|>" in buffer:
                story, buffer = buffer.split("<|endoftext|>", 1)
                story = story.strip()
                key = hashlib.sha256(" ".join(story.split()).encode()).hexdigest()
                if len(story) >= 100 and key not in seen:
                    seen.add(key)
                    stories.append(story)
                    if len(stories) % 5000 == 0:
                        print(f"{filename}: {len(stories)} stories", flush=True)
                    if len(stories) == count:
                        return stories
    raise ValueError(f"Only found {len(stories)} unique stories, requested {count}")


def prepare(args):
    if min(args.stories, args.heldout, args.tokenizer_stories) < 1 or args.length < 4 or not 261 <= args.vocab_size <= 65535:
        raise ValueError("Positive document counts, length >=4 and vocabulary 261..65535 required")
    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise ValueError("Prepared data already exists; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    metadata = requests.get("https://huggingface.co/api/datasets/roneneldan/TinyStories", timeout=30)
    metadata.raise_for_status()
    revision = args.revision or metadata.json()["sha"]
    seen = set()
    # Held-out source first: reject exact document duplicates from training.
    heldout = download_stories(revision, "TinyStories-valid.txt", args.heldout * 2, seen)
    train = download_stories(revision, "TinyStories-train.txt", args.stories, seen)
    splits = {"train": train, "validation": heldout[::2], "test": heldout[1::2]}
    existing = getattr(args, "tokenizer", None)
    if existing:
        tokenizer = Tokenizer.from_file(existing)
        if any(tokenizer.token_to_id(t) != i for i, t in enumerate(SPECIAL_TOKENS)):
            raise ValueError("Incompatible special-token IDs")
        shutil.copy2(existing, out / "tokenizer.json")
    else:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(train[:args.tokenizer_stories], trainers.BpeTrainer(vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2))
        tokenizer.save(str(out / "tokenizer.json"))
    manifest = {"dataset": "roneneldan/TinyStories", "revision": revision, "license": "cdla-sharing-1.0",
                "source": "https://huggingface.co/datasets/roneneldan/TinyStories", "synthetic": True,
                "selection": "First unique documents in pinned source order; alternating validation/test documents",
                "tokenizer_train_documents": min(args.tokenizer_stories, len(train)), "vocab_size": tokenizer.get_vocab_size(),
                "max_length": args.length, "tokenizer_sha256": digest(out / "tokenizer.json"), "splits": {}}
    if existing:
        manifest["tokenizer_source"] = str(existing)
    for name, stories in splits.items():
        # Nonoverlapping windows within documents, never across split boundaries.
        tokens = windows = complete = 0
        raw_path = out / f"{name}.rows.tmp"
        with raw_path.open("wb") as stream:
            for start in range(0, len(stories), 1000):
                rows = []
                for encoded in tokenizer.encode_batch(stories[start:start+1000]):
                    ids = encoded.ids
                    tokens += len(ids)
                    complete += int(len(ids) <= args.length - 2)
                    for offset in range(0, len(ids), args.length - 2):
                        segment = [BOS] + ids[offset:offset + args.length - 2]
                        if offset + args.length - 2 >= len(ids):
                            segment += [EOS]
                        rows.append(segment + [PAD] * (args.length - len(segment)))
                np.asarray(rows, dtype=np.uint16).tofile(stream)
                windows += len(rows)
        path = out / f"{name}.npy"
        raw = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=(windows, args.length))
        np.save(path, raw)
        del raw
        raw_path.unlink()
        manifest["splits"][name] = {"documents": len(stories), "windows": windows, "text_tokens": tokens,
            "complete_story_windows": complete,
            "documents_sha256": hashlib.sha256("\n<|endoftext|>\n".join(stories).encode()).hexdigest(), "array_sha256": digest(path)}
        print(name, manifest["splits"][name], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_arrays(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if digest(directory / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    arrays = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.npy"
        if digest(path) != manifest["splits"][name]["array_sha256"]:
            raise ValueError(f"Corrupted {name} array")
        arrays[name] = np.load(path, mmap_mode="r")
        array=arrays[name]
        error=None
        if array.ndim!=2 or array.shape[1]!=manifest["max_length"] or array.dtype!=np.uint16:
            error=f"Invalid {name} array shape or dtype"
        else:
            for start in range(0,len(array),4096):
                chunk=array[start:start+4096]
                if np.any(chunk[:,0]!=BOS) or np.any(chunk>=manifest["vocab_size"]):
                    error=f"Invalid {name} token IDs or BOS"
                    break
                if np.any((chunk[:,:-1]==PAD)&(chunk[:,1:]!=PAD)):
                    error=f"{name} must be right-padded for causal attention"
                    break
        if error:
            for opened in arrays.values():opened._mmap.close()
            raise ValueError(error)
    return arrays, manifest
