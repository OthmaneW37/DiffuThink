# StoryPatch v2: whole-word span training

This experiment tests whether training on the actual editing pattern improves
StoryPatch. The published 13.44M model is the baseline. We train an adaptation of
that model and a 58,471,936-parameter challenger initialized from random weights.
No external teacher, tokenizer vocabulary or pretrained weights are used.

## Predeclared protocol

- Reuse the pinned 500,000-document TinyStories training corpus and its original
  validation/test document separation. The tokenizer and 512-token context stay fixed.
- Filter replacement characters, control characters, very short windows and
  windows with over 20% repeated word trigrams. This is not semantic quality verification.
- Reserve held-out windows first and reject normalized exact duplicates across
  splits. General near-duplicate detection is not implemented in this experiment.
- Precompute up to eight different whole-word spans per accepted window. Short
  windows can repeat a span to fill the eight slots; slot counts are not counts
  of distinct documents or guaranteed unique edits.
- Select 1–12 complete English words, at most 16 subword tokens, with context on
  both sides. Special tokens and original EOS boundaries remain untouched.
- Targets are authentic source spans, not human-verified corrections. An equally
  valid alternative wording can score zero on reference reconstruction.
- Alternate causal updates and span updates. Half of the span updates reveal a
  subset of the missing tokens to match intermediate iterative inference states.
- Uniform sampling of anchors in a length-sorted array, with wrapped random
  offsets, groups similar sequence lengths without changing marginal sampling.
- Report presented non-PAD tokens separately from supervised targets. These are
  not unique-token counts or total training FLOPs.

The initial configurations were fixed before test evaluation:

| Candidate | Initialization | Updates | Peak LR | Batch | Seed |
|---|---|---:|---:|---:|---:|
| 13.44M adaptation | Our published Story-512 checkpoint | 3,000 | 0.00005 | 24 | 48 |
| 58.47M mixed pilot | Random weights; width 512, 12 layers, 8 heads | 12,000 planned; stopped after checkpoint 2,500 | 0.0003 | 24 | 48 |

### Validation-driven correction

The mixed-from-scratch pilot stayed near 6% span accuracy across four validation
checkpoints. A CPU diagnostic confirmed repeated prediction of the same token
across masked positions, while causal likelihood improved. We stopped the process
after the step-2,500 checkpoint and retained its logs and weights. Some updates
after that checkpoint were discarded. See `reports/storypatch-v2/pilot_notes.json`.

The revised continuation uses only this project's own pilot weights:

| Phase | Initialization | Updates | Peak LR | Objective / selection |
|---|---|---:|---:|---|
| 58M causal consolidation | Pilot checkpoint 2,500 | 3,500 | 0.00015 | Causal only; select by causal validation NLL |
| 58M span specialization | Best causal consolidation | 6,000 | 0.00005 | Mixed objective; original mixed validation score |

This schedule change was informed by validation and the CPU diagnostic, not by
candidate test results. It is an adaptive experiment, not a fully preregistered
size comparison. The failed initial approach is part of the report.

Both retain the existing RoPE + learned absolute positions. Removing the learned
positions would be a separate architecture ablation; no benefit is assumed.
The runs change size and initialization history, so they do not isolate a causal
effect of model size. The 13M adaptation isolates a practical training update,
but does not isolate filtering from objective changes.

Checkpoint selection uses only the first 256 fixed validation windows:
`span NLL + 0.25 * causal NLL`. A deployment candidate must beat the baseline on
that score and on span NLL, while causal NLL may regress by no more than 10%.
The baseline remains available. Test results are descriptive, not used to select
checkpoints or tune generation settings.

## Evaluation

- 400 held-out windows, with a fixed whole-word mask: masked-token NLL, token
  accuracy and parallel exact span reconstruction.
- First 64 of those windows: iterative greedy reconstruction, with the reference
  span length supplied. This oracle-length task is explicitly separate from editing.
- 36 authored diagnostic editing cases across six categories: emotion, object,
  state, location, identity and agreement. These cases are never training data.
- Actual editor, unchanged six attempts and original causal-NLL ranking: accepted
  reference top-1/top-3, empty results, exact context preservation, latency and
  every candidate. Accepted-answer sets are incomplete; this is not a general
  semantic-coherence score or a blind human evaluation.
- Model weights, tokenizer, datasets and challenge file are SHA-256 fingerprinted.

## Reproduce

```powershell
python -m diffuthink.storypatch.data
python -m diffuthink.storypatch.benchmark
python -m unittest discover -s tests -v
python scripts/run_storypatch_experiment.py
python scripts/storypatch_report.py
python scripts/package_storypatch_v2.py
```

The orchestrator is a finite local experiment. It stops on failure, records its
current phase in `runs/storypatch-experiment-status.json`, and writes each phase's
log under `reports/storypatch-v2/`. It does not publish or overwrite released weights.
On this machine it enables the already-tested process-local ROCm attention flag.
Other hardware does not require that flag.

For interrupted training, repeat the exact training command with `--resume`.
The whole sequence can also be restarted with `python scripts/run_storypatch_experiment.py --resume`;
finished training phases reload their final state and evaluation is recomputed.
`--stop-after N` saves an early checkpoint without altering the planned LR schedule.
The CPU interruption test verifies exact equality of resumed weights and RNG state.

```powershell
python scripts/run_storypatch_experiment.py --resume
python -m diffuthink.storypatch.evaluate --model runs/storypatch-58m-edit/best --output reports/storypatch-v2/scratch58m.json
```

## Attribution and boundaries

The corpus originates from [TinyStories](https://arxiv.org/abs/2305.07759), pinned
at revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, also recorded in the data
manifest. Source text and derived reconstruction
examples retain the source dataset's CDLA-Sharing-1.0 attribution and terms;
project code remains Apache-2.0. No new scientifically novel infilling method is claimed.

[Fill-in-the-middle training](https://arxiv.org/abs/2207.14255) is related work.
This implementation uses bidirectional masked reconstruction, not that paper's
autoregressive prefix/suffix/middle reordering. A learned reranker, instruction
following and semantic contradiction penalties require additional data and
separate experiments; they are not claimed by this release.
