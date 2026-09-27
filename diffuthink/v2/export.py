"""Create a self-contained Hub folder. Never uploads or publishes anything."""
import json
from pathlib import Path
import shutil
from .model import Denoiser
from .data import digest


def export(args):
    source = Path(args.model)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty to prevent stale artifacts")
    model = Denoiser.from_pretrained(source)
    info = json.loads((source / "training_info.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "training_info.json"):
        shutil.copy2(source / name, output / name)
    license_file = Path(__file__).resolve().parents[2] / "LICENSE"
    if license_file.exists():
        shutil.copy2(license_file, output / "LICENSE")
    package = Path(__file__).resolve().parents[1]
    shutil.copytree(package, output / "diffuthink", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / "requirements.txt").write_text("torch>=2.2\ntokenizers>=0.22,<0.23\nsafetensors>=0.4\nrequests>=2.31\nnumpy>=1.26\n", encoding="utf-8")
    report = json.loads(Path(args.report).read_text())
    if report["best"] != info:
        raise ValueError("Report does not describe the exported checkpoint")
    shutil.copy2(args.report, output / "evaluation.json")
    parameters = sum(p.numel() for p in model.parameters())
    hybrid = info.get("continuation_mode") == "autoregressive"
    reconstruction = report["denoising_test"] if hybrid else report["test"]
    test_samples = 512 if hybrid else report["training"]["test_samples"]
    rows = "\n".join(f"| {r['mask_rate']:.0%} | {r['accuracy']:.2%} | {r['top5_accuracy']:.2%} | {r['cross_entropy']:.3f} | {r['unigram_accuracy']:.2%} |" for r in reconstruction)
    card = f'''---
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- diffusion
- masked-language-modeling
- from-scratch
- research
license: apache-2.0
datasets:
- roneneldan/TinyStories
model-index:
- name: DiffuThink-13M
  results: []
---
# DiffuThink-13M

A {parameters:,}-parameter subword text denoiser trained from random initialization.
No pretrained model weights or external tokenizer vocabulary were used.
Custom PyTorch architecture and inference API; **not** an AutoModel-compatible checkpoint.

## What it does

Completes English story fragments and reconstructs masked subwords. Uses bidirectional
RoPE attention, RMSNorm, SwiGLU, learned absolute positions, tied input/output embeddings
and noise conditioning. The training objective mixes random masks and masked suffixes.
Generation uses confidence-ordered iterative unmasking in blocks. The optional adaptive
budget is a confidence heuristic, not learned reasoning or a new diffusion theorem.

## Measured held-out reconstruction

| Mask rate | Top-1 token accuracy | Top-5 | Cross-entropy (nats) | Unigram top-1 |
| --- | --- | --- | --- | --- |
{rows}

Exact experiment details, counts, versions and limitations: `evaluation.json`.
These numbers are subword reconstruction metrics, not chat quality or autoregressive perplexity.
The test uses the first {test_samples} prepared held-out windows.

## Run locally

Download this repository, install `requirements.txt`, and run from its directory:

```python
from diffuthink.v2.inference import load, continue_text
model, tokenizer = load(".", device="cpu")
result = continue_text(model, tokenizer,
    "Once upon a time, a little girl found", max_new_tokens=64,
    steps=12, temperature=0.7, seed=42)
print(result["text"])
```

For a Hub download use `huggingface_hub.snapshot_download(repo_id=YOUR_REPO_ID)`,
then use that returned directory as the model path and Python import root.
The Python code is included for inspection; no remote-code execution is required by a loader.

## Data and provenance

TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories
Pinned revision: `{info['manifest']['revision']}`.
100,000 training stories; 1,000 validation and 1,000 test stories from the source validation
file. Normalized exact-document duplicates are removed across splits; near-duplicate
and semantic leakage detection is not claimed. Each document is windowed only after splitting.
The byte-level BPE tokenizer is trained only on training stories (first 50,000).
TinyStories is **synthetic**, generated using GPT-3.5/4, and its card declares CDLA-Sharing-1.0.
The data is not included in this model release. See `training_info.json` for lineage and hashes.

## Limitations and authorship

Small English story model: can repeat, invent, stop early, or produce ungrammatical text.
No general reasoning, instruction following, factual accuracy, multilingual proficiency,
or superiority to established language models is claimed. The basic unigram comparison
is weak; sampler and bigram experiments are documented separately in the project.
Development was substantially AI-assisted. The portfolio contribution is the implemented
pipeline, experimental method, analysis and subsequent personal work, not a claim to have
invented Transformers, BPE or discrete diffusion.

## Publication

This is a local release candidate, not an uploaded model. Select an appropriate project/model
license before a public release and preserve source-data attribution. No license tag is
asserted automatically for the trained weights.
'''
    if hybrid:
        card = card.replace("DiffuThink-13M", "DiffuThink-Hybrid-13M")
        card = card.replace("subword text denoiser trained from random initialization.",
            "hybrid language model whose entire weight lineage was trained from random initialization in this project.")
        card = card.replace("Generation uses confidence-ordered iterative unmasking in blocks.",
            "This hybrid checkpoint adds 90% causal and 10% denoising training batches. Continuation uses strictly causal next-token prediction. Infilling uses confidence-ordered unmasking; autoregressive continuation is NOT diffusion.")
        card = card.replace("from diffuthink.v2.inference import load, continue_text", "from diffuthink.v2.inference import load\nfrom diffuthink.v2.hybrid import generate")
        card = card.replace("result = continue_text(model, tokenizer,", "result = generate(model, tokenizer,")
        card = card.replace("steps=12, temperature=0.7, seed=42", "temperature=0.5, seed=42, precision=\"fp32\"")
        metric = report["causal_test"]
        card += f"\n## Causal evaluation\n\nHeld-out next-token NLL: {metric['nll']:.4f} nats; perplexity: {metric['perplexity']:.3f}. Evaluated over {metric['tokens']} targets. This is a genuine causal likelihood metric, unlike masked reconstruction CE. It does not measure story coherence.\n\nDefault decoding uses top-p 0.9, repetition penalty 1.12 and no repeated four-token sequences. These are disclosed logit constraints, not grammatical rewriting. The project report also preserves samples without repetition controls.\n"
    if info.get("context_extension"):
        card=card.replace("DiffuThink-Hybrid-13M", "DiffuThink-Story-512")
        card=card.replace("100,000 training stories", f"{info['manifest']['splits']['train']['documents']:,} training stories")
        card=card.replace("max_new_tokens=64", "max_new_tokens=192")
        card=card.replace('precision="fp32")', 'precision="fp32", finish_sentence_tokens=32)')
        card += f"\n## Expanded context phase\n\nContext: {model.config.max_length} tokens. The original positional embeddings were preserved; added positions were initialized near their mean and then trained. The BPE vocabulary and held-out document splits are unchanged. This is continued training of this project's own from-scratch weights.\n\nThe local demo allows up to 32 extra tokens after its requested budget to reach terminal punctuation. It reports whether EOS, punctuation, or the hard length limit ended generation. That display heuristic does not guarantee a finished or coherent story. Fair before/after benchmarks disable the extra-token allowance.\n"
    if getattr(args,"comparison",None):
        comparison=json.loads(Path(args.comparison).read_text(encoding="utf-8"))
        if comparison["weights"]["after"] != digest(source/"model.safetensors"):
            raise ValueError("Comparison does not describe these weights")
        shutil.copy2(args.comparison,output/"comparison.json")
        before=comparison["likelihood"]["before"];after=comparison["likelihood"]["after"]
        card+=f"\n## Controlled previous-checkpoint comparison\n\nBoth models were evaluated on the same {comparison['protocol']['common_test_windows']} legacy test windows (192 tokens), with identical targets and tokenizer. Perplexity: **{before['perplexity']:.3f} before, {after['perplexity']:.3f} after**. This differs from the 512-token-window evaluation above. All 24-prompt, two-seed samples and their decoding settings are in `comparison.json`. Repetition and EOS rates are descriptive indicators, not a semantic-coherence assessment. Corpus size, context and training compute changed together; their individual effects are not isolated.\n"
    (output / "README.md").write_text(card, encoding="utf-8")
    checksums = {str(path.relative_to(output)).replace("\\", "/"): digest(path) for path in output.rglob("*") if path.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"Local Hub release candidate: {output}")
