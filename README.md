# StoryPatch · powered by DiffuThink

**Select a passage. Explore alternatives. Keep the rest of the story unchanged.**

StoryPatch is an inspectable editing workshop for simple English stories, powered by a 13.44M-parameter model whose weight lineage was trained from scratch in this project. It generates local alternatives using bidirectional denoising, ranks them with causal likelihood, and lets the user apply or undo one change. No external pretrained model or generation API is used.

The contribution is the constrained, reversible workflow and visible model behavior. Text infilling, Transformers and likelihood ranking are established techniques; no scientific novelty claim is made.

## Learned ranking and keeping the original (local experiment)

A separate 3.53M-parameter Transformer learns span compatibility from random
initialization. It ranks the same denoising proposals and compares them with the
original passage. Validation chooses when to recommend a replacement or keep
the original. The user always decides whether to apply the suggestion. The
original NLL ranking stays the UI default: the new scorer improves the procedural
paired test but regresses on legacy diagnostics (top1 52.8% vs 72.2%). Select
"Correction ciblée · expérimental" to compare it explicitly.

Training uses 12,000 procedural contexts and additional proposals mined from
training contexts only. A paired test contains 600 editing cases across 300
contexts. Labels are synthetic, with incomplete accepted-answer lists; no
human-validated quality gain is claimed. A CSV for blind review is provided.

See [measured results](reports/storypatch-ranker-v2/RESULTATS.md) and
[the French implementation and reproduction guide](docs/STORYPATCH_RANKER.md).
The earlier unsuccessful ranker experiment is also retained. This local
experiment does not replace the published Hub checkpoint.

```powershell
python -m diffuthink.v2 serve --model runs/storypatch-13m/best --ranker runs/storypatch-ranker-v2/best --device cpu --port 7862
```

**Published model:** [OthmaneW/DiffuThink-Story-512](https://huggingface.co/OthmaneW/DiffuThink-Story-512)

## StoryPatch v2 experiment

Whole-word span training is implemented and measured. A 13.44M adaptation and a
58.47M model initialized from scratch were trained locally. Validation selects
the adapted 13M: iterative exact reconstruction improves from 14.1% to 18.8% on
64 held-out passages, while causal perplexity and top-three diagnostic coverage
slightly regress. The larger model does not beat the smaller one at this budget.

See [all results and failures](reports/storypatch-v2/RESULTATS.md),
[reproduction protocol](docs/STORYPATCH_V2.md), and
[the French explanation](docs/STORYPATCH_V2_EXPLIQUE.md).
This local experiment does not replace the previously published Hub checkpoint.

```powershell
python -m diffuthink.v2 serve --model runs/storypatch-13m/best --device cpu --port 7862
```

The standalone local release folder `artifacts/StoryPatch-v2` includes its own
inference source, checkpoint, model card, evidence and checksums.

## Try locally

```powershell
python -m pip install -e '.[research]'
python -m diffuthink.v2 serve
```

Open [StoryPatch](http://127.0.0.1:7861/). The [research lab](http://127.0.0.1:7861/lab) retains continuation and infilling. Select 1–12 words, compare proposals, apply one and undo it. Every character outside the selected passage is preserved by string composition, not by an instruction to the model.

The checkpoint is `runs/stories-512/best`. Use `--device cpu` without a supported GPU. Weights and data are excluded from Git; a clean clone must train or obtain the separate bundle.

## Model and evidence

- 13,439,680 parameters; six layers, width 320, eight heads, RoPE, learned positions, RMSNorm, SwiGLU and tied embeddings.
- Own 8,192-token BPE; 512-token context; 500,000 synthetic TinyStories training documents.
- Latest phase: 16,000 additional updates and 108,397,864 non-PAD tokens presented.
- On identical 1,380 legacy test windows, perplexity improves from **7.206 to 6.147**. Repetition and EOS rates do not uniformly improve. All samples and regressions are retained.
- 42 unit tests, exact CPU recovery tests, independent CPU/GPU bundle reload and browser application/undo/keep checks.

Read [all results](reports/context512/RESULTATS.md), [training protocol](docs/STORY512.md), [StoryPatch design](docs/STORYPATCH.md), and the [French interview guide](docs/ENTRETIEN.md).

## Reproduce and package

Reproduce the earlier lineage using [the hybrid instructions](docs/HYBRID_README.md), then follow [the context expansion protocol](docs/STORY512.md).

```powershell
python -m unittest discover -s tests -v
python scripts/test_hybrid_resume.py --expanded
python scripts/compare_context.py
python scripts/context_report.py
python -m diffuthink.v2 export --comparison reports/context512/comparison.json
python scripts/package_space.py
docker build -t storypatch artifacts/StoryPatch-Space
docker run --rm -p 127.0.0.1:7860:7860 storypatch
```

`artifacts/DiffuThink-Story-512` contains weights, tokenizer, source, model card, comparisons and checksums. `artifacts/StoryPatch-Space` adds a CPU Docker app and Space metadata. The released code and weights use Apache-2.0; preserve the separate TinyStories dataset attribution.

## Limits and authorship

Suggestions can be incorrect or less appropriate than the original. Lower local NLL is a model preference, not proof of correctness. Six attempts, variable mask lengths and ranking are disclosed. The app saves no submitted texts to application files and calls no external model. A hosted Space processes input on its server.

TinyStories is synthetic. This is a small English story model, not a general reasoning or factual assistant. Development was substantially AI-assisted. Reproduce and understand the work before presenting it as personal expertise.

## License

Project code and released weights: [Apache-2.0](LICENSE). TinyStories is not redistributed and retains its CDLA-Sharing-1.0 terms and attribution.
