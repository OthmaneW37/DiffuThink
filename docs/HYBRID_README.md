# DiffuThink Hybrid · From-Scratch Language Modelling

A **13.3M-parameter model trained in this project**, with a custom 8,192-token BPE vocabulary. No external pretrained weights or tokenizer.

The current release uses **autoregressive continuation** for stories and **bidirectional denoising** for infilling. It grew out of a diffusion experiment whose reconstruction scores did not translate into reliable free generation. The change of objective is explicit: causal continuation is not presented as diffusion.

## Current release

- Custom Transformer: 6 layers, width 320, 8 heads; RoPE, learned absolute positions, RMSNorm, SwiGLU and tied embeddings.
- 100,000 synthetic English TinyStories documents, split before windowing; context length 192 tokens.
- Two initial denoising phases followed by a 16,000-update hybrid phase: 90% causal batches, 10% denoising batches.
- Validation-selected checkpoints, held-out causal likelihood, raw before/after samples, repetition-control ablation and resumable training.
- Local demo, Safetensors export and documented AI-assisted development.

## Use the trained local model

```powershell
python -m pip install -e '.[research]'
python -m diffuthink.v2 sample --prompt "One day a man was under a tree and " --device cuda
python -m diffuthink.v2 serve
```

Open [the local demo](http://127.0.0.1:7861). The current checkpoint is `runs/stories-hybrid/best`. Set `--device cpu` on machines without a supported GPU. A clean clone must train or obtain the separately published bundle: weights and prepared data are excluded from Git.

Continuation defaults to temperature 0.5, nucleus sampling 0.9, repetition penalty 1.12 and no repeated four-token sequence. Temperature 0 gives deterministic decoding. These are disclosed sampling controls, not post-generation grammatical correction. The report also shows outputs without repetition controls.

For the experimental diffusion path:

```powershell
python -m diffuthink.v2 sample --generation-mode diffusion --prompt "Once upon a time"
python -m diffuthink.v2 sample --prompt "The little girl" --suffix " in the garden." --missing-tokens 1
```

## Reproduce training

```powershell
python -m diffuthink.v2 prepare
python -m diffuthink.v2 train
python -m diffuthink.v2 train --output runs/stories-v2-refined --initialize-from runs/stories-v2/best --steps 6000 --lr 0.0002 --warmup 150 --seed 43
python -m diffuthink.v2.hybrid
```

Default training uses GPU BF16 on the verified AMD PyTorch installation. `python -m diffuthink.v2.hybrid --resume` recovers the latest completed checkpoint with unchanged settings. `requirements-environment.txt` records the local environment, not a portable AMD driver installation recipe.

## Evaluate and export

```powershell
python -m unittest discover -s tests -v
python scripts/test_hybrid_resume.py
python scripts/compare_continuations.py
python -m pip install -e '.[plots]'
python scripts/hybrid_report.py
python -m diffuthink.v2 export
```

The export in `artifacts/DiffuThink-Hybrid-13M` includes weights, tokenizer, source, dependencies, model card, evaluation and checksums. It uses a custom PyTorch API rather than claiming AutoModel compatibility. Export does not publish anything.

## Read the evidence

- [Current results and every before/after example](reports/hybrid/RESULTATS.md)
- [Why the model became hybrid; training and inference details](docs/HYBRID.md)
- [Earlier V2 protocol](docs/V2_RESEARCH.md)
- [Earlier V2 results](reports/v2/RESULTATS.md)
- [Publication guide](docs/PUBLICATION.md)

The held-out causal perplexity measures next-token probability, not narrative coherence. The prior masked accuracy is a different metric. Examples retain errors and incomplete outputs at the token budget. This remains a small English story model: no claim of perfect grammar, general reasoning, multilingual quality, state of the art or production readiness. TinyStories is synthetic; source attribution and hashes are retained.

Development is substantially AI-assisted. The portfolio contribution is the implemented pipeline, experimental decisions, measurements and personal understanding—not inventing standard Transformer techniques. Reproduce and explain the work before presenting it as your expertise.
