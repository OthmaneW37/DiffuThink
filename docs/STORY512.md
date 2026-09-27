# DiffuThink Story 512 — corpus and context expansion

This experiment addresses contradictions, repetitive local continuations and missing endings in the 192-token hybrid checkpoint. Longer context and more diverse training documents are hypotheses, not guarantees of narrative coherence.

## What changes

- 500,000 training stories instead of 100,000, at the same pinned TinyStories revision.
- The exact existing byte-level BPE file is reused. Token IDs cannot be reassigned when reusing embeddings.
- Context grows from 192 to 512. The first 192 learned positional vectors are preserved exactly. The 320 added vectors are initialized around the old mean with Gaussian noise of standard deviation 0.02, then trained.
- The architecture retains six layers, width 320, eight heads, RoPE, RMSNorm, SwiGLU and tied vocabulary embeddings. The additional position vectors add 102,400 parameters, for a total of 13,439,680.
- 485,910 of 500,000 training stories (97.182%) fit in a complete window. Longer documents are split without overlap. Only the actual final segment receives EOS; no artificial end-of-story label is inserted at a window boundary.
- The 1,000 validation documents and 1,000 test documents are unchanged, as verified by document hashes and frozen tokenizer hash. Their window counts change with the longer context. Exact deduplication is retained; near-deduplication is not claimed.
- The new phase has 16,000 optimizer updates, batch size 32, peak learning rate 0.00015, warmup 400, cosine decay, seed 45, and 90% causal / 10% denoising batches. This is a continuation of our own from-scratch weights, not a new pretrained base.

## Reproduce

Starting from the previous `runs/stories-hybrid/best` release:

```powershell
python -m diffuthink.v2 prepare --output data/processed/stories-512 --stories 500000 --length 512 --tokenizer data/processed/stories-v2/tokenizer.json
python -m diffuthink.v2.hybrid --source runs/stories-hybrid/best --data data/processed/stories-512 --output runs/stories-512 --expand-data --steps 16000 --batch-size 32 --lr 0.00015 --warmup 400 --eval-every 1000 --eval-samples 256 --seed 45
```

For recovery, repeat the training command with `--resume`. It reloads model, optimizer, RNG state, schedule position and best validation loss; settings and manifest must match. CPU recovery was tested against uninterrupted training, including context expansion.

On this AMD GPU the first 2,000 updates used the ordinary attention path. We resumed at that checkpoint with tested accelerated attention and the process-local environment setting below. The change is recorded in `execution_notes.json`. GPU backend rounding can change subsequent sampling/training slightly; bit-for-bit cross-backend reproduction is not claimed. This flag is optional and specific to the local ROCm installation, not a requirement on other machines.

```powershell
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1'
```

The optimized causal path assumes all padding is on the right, as produced by this preparation pipeline. A supervised non-PAD position cannot attend to a future PAD position through causal attention. The generic model API keeps the explicit validity mask by default. Tests compare both supervised logits and gradients. Infilling always retains the bidirectional validity mask.

## Evaluate before claiming improvement

```powershell
python -m unittest discover -s tests -v
python scripts/test_hybrid_resume.py --expanded
python scripts/compare_context.py
python scripts/context_report.py
```

The likelihood comparison evaluates both checkpoints on all 1,380 **identical legacy test windows of length 192**, using the same target tokens. The separate new-window test result must not be directly compared with the previous 512-window result.

Generation comparison uses 24 fixed prompts and two seeds, temperature 0.5, top-p 0.9, repetition penalty 1.12, no repeated 4-grams, and 160 new tokens. The first prompt is the already known user failure; the remaining prompts were fixed before inspecting new weights. All outputs are retained. EOS rate, terminal punctuation and repeated trigrams are descriptive metrics, not semantic-coherence scores. No blind human rating is claimed.

This experiment changes training duration, data volume and context together. A gain cannot be attributed uniquely to any one factor. Causal-only versus hybrid training remains a separate future ablation, not a completed experiment.

## Demo and endings

```powershell
python -m diffuthink.v2 serve --model runs/stories-512/best --port 7861
python -m diffuthink.v2 sample --model runs/stories-512/best --device cuda --new-tokens 192 --finish-sentence-tokens 32 --prompt "One day a man was under a tree and "
```

The demo requests 192 new tokens by default and permits up to 32 extra tokens to reach punctuation. It never rewrites, truncates back to a nicer sentence or selects the best of several generations. It reports `eos`, `sentence_boundary` or `length`. A sentence boundary is not proof that a story is complete. For fair checkpoint comparisons this extra allowance is disabled.

## Export locally

```powershell
python -m diffuthink.v2 export --model runs/stories-512/best --report runs/stories-512/report.json --comparison reports/context512/comparison.json --output artifacts/DiffuThink-Story-512
```

The export includes source, weights, tokenizer, training lineage, evaluation and checksums. It does not upload anything. The project remains AI-assisted and specialized in simple English stories.
