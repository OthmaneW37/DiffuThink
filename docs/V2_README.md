# DiffuThink · From-Scratch Text Denoising

A **13.3M-parameter research model trained from random weights**, with a byte-level BPE tokenizer trained on the project corpus. No pretrained model or tokenizer weights.

DiffuThink reconstructs masked subwords and continues short English stories through iterative denoising. This repository includes the data pipeline, custom Transformer, GPU training, exact CPU resume test, sampler ablations, contextual baseline, local demo and a Hugging Face export.

**Status:** experimental English story model, not a general chatbot. Development is substantially AI-assisted. Results and limitations are part of the deliverable.

## Architecture

| Component | Configuration |
| --- | --- |
| Parameters | 13,337,280 |
| Transformer | 6 layers, width 320, 8 heads |
| Building blocks | Bidirectional attention, RoPE, RMSNorm, SwiGLU |
| Position signal | RoPE + learned absolute embeddings |
| Vocabulary | 8,192 byte-level BPE tokens, trained on train only |
| Context | 192 subword tokens |
| Objective | Random-mask and suffix-mask reconstruction |
| Generation | Blockwise confidence-guided unmasking |
| Training data | 100,000 TinyStories documents / 21.1M text tokens |

TinyStories consists of **synthetic** short stories, not human-authored prose. The source, license declaration, pinned revision and partition hashes are recorded in the data manifest.

## Quick start

Python 3.10+ and a working PyTorch installation:

```powershell
python -m pip install -e '.[research]'
python -m diffuthink.v2 prepare
python -m diffuthink.v2 train
python -m diffuthink.v2 train --output runs/stories-v2-refined --initialize-from runs/stories-v2/best --steps 6000 --lr 0.0002 --warmup 150 --seed 43
python -m diffuthink.v2 sample --prompt "Once upon a time, a little girl found"
python -m diffuthink.v2 serve
```

Open [the local demo](http://127.0.0.1:7861). The training commands default to GPU BF16. For a CPU-only installation use `--device cpu --precision fp32` when training and `--device cpu` when serving. Training will be much slower on CPU. GPU driver installation is outside this repository.

The trained local checkpoint is `runs/stories-v2-refined/best`. Weights, optimizer checkpoints and prepared data are deliberately excluded from Git. A clean clone must train or download a separately published model bundle. `requirements-environment.txt` records the execution environment, including a platform-specific AMD PyTorch build; it is not a portable installation recipe.

## Reproduce and evaluate

```powershell
python -m unittest discover -s tests -v
python -m diffuthink.v2 benchmark --split validation --output reports/v2/sampling_validation.json
python -m diffuthink.v2 benchmark --split test --output reports/v2/sampling_test.json
python -m diffuthink.v2 export
```

The benchmark compares parallel one-pass reconstruction, left-to-right unmasking, confidence ordering, adaptive confidence ordering and a bidirectional bigram baseline on identical masks. All generated fixed-prompt samples, including failures, remain in the reports.

Masked-token accuracy is **not** conversational quality. These subword results are not comparable to the byte accuracy of V1. More denoising steps are not assumed to be better. Adaptive sampling is a transparent confidence heuristic, not learned reasoning.

## Resume safely

For an interrupted second phase, preserve its output folder and resume the saved state with the same schedule:

```powershell
python -m diffuthink.v2 train --output runs/stories-v2-refined --resume runs/stories-v2-refined/last.pt --steps 6000 --lr 0.0002 --warmup 150 --seed 43
```

`--resume` restores optimizer and random-generator states. `--initialize-from` starts a new optimization phase from our own weights and records their lineage. They are intentionally different operations.

## What to read

- [French research guide: architecture, mathematics and protocol](docs/V2_RESEARCH.md)
- [Publication guide and LinkedIn draft](docs/PUBLICATION.md)
- [Measured V2 results](reports/v2/RESULTATS.md)
- [Original byte-level prototype](docs/V1_README.md)
- [Introductory explanation](docs/EXPLICATION.md)

Core code: `diffuthink/v2/model.py`, `data.py`, `train.py`, `inference.py`, `benchmark.py`, `baselines.py`, `demo.py`, and `export.py`. The educational experiments and V1 are preserved separately.

## Honest portfolio positioning

The contribution is a custom implementation, reproducible training pipeline, measured comparisons and analysis of failures. Transformers, BPE and diffusion concepts are established techniques. Before presenting this work, reproduce an experiment, explain the tensor shapes and loss, and document a modification you personally understand. No state-of-the-art, general reasoning or production-readiness claim is made.
