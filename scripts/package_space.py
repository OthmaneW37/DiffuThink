"""Prepare a local Docker Space from a verified model bundle; never upload."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",default="artifacts/DiffuThink-Story-512")
    p.add_argument("--output",default="artifacts/StoryPatch-Space")
    args=p.parse_args();source=Path(args.bundle);out=Path(args.output)
    checks=json.loads((source/"SHA256SUMS.json").read_text())
    for name,expected in checks.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Invalid bundle file: {name}")
    if out.exists():raise ValueError("Use a new Space output folder")
    parameters=json.loads((source/"training_info.json").read_text())["parameters"]
    shutil.copytree(source,out)
    (out/"README.md").rename(out/"MODEL_CARD.md")
    (out/"app.py").write_text('''import os
from argparse import Namespace
from diffuthink.v2.demo import serve

if __name__ == "__main__":
    serve(Namespace(model=".",device="cpu",port=int(os.environ.get("PORT","7860")),
                    host=os.environ.get("SPACE_HOST","0.0.0.0"),threads=2))
''',encoding="utf-8")
    (out/"Dockerfile").write_text('''FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
''',encoding="utf-8")
    (out/".dockerignore").write_text("__pycache__\n*.pyc\n.git\n",encoding="utf-8")
    (out/"README.md").write_text('''---
title: StoryPatch by DiffuThink
emoji: 🧩
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---
# StoryPatch

Select a short passage in a simple English story, inspect alternatives, and keep everything outside the selection unchanged. Powered by a 13.44M-parameter model trained within this project, without external pretrained weights or generation APIs.

## Try it

1. Start with an example or paste a short English text.
2. Select 1–12 whole words and explore the passage.
3. Compare suggestions and inspect the denoising trace.
4. Apply one change, undo it, or export your text.

The app generates six deterministic attempts, deduplicates them and ranks the remaining alternatives by local causal negative log-likelihood. It preserves surrounding text exactly through string composition. Ranking is a model preference, **not** a guarantee of grammar or coherence. Original wording can score better than every proposal. There is no claim that text infilling itself is a new research invention.

The generation research lab remains available at `/lab`.

## Model and evidence

See [MODEL_CARD.md](MODEL_CARD.md), `evaluation.json`, `comparison.json` and `training_info.json`. TinyStories is synthetic; corpus attribution and weight lineage are documented. Development was substantially AI-assisted.

Texts are processed by this app without being saved to application files or sent to another model. When hosted, inference runs on the Space server. No account or API key is needed by the app.

## Run

```sh
docker build -t storypatch .
docker run --rm -p 127.0.0.1:7860:7860 storypatch
```

CPU inference, two PyTorch threads. This is an exploratory demo with sequential requests, not a multi-user production service.

## Publication status

This folder is a local release candidate. No Space has been created or uploaded automatically. Code and weights use Apache-2.0; preserve TinyStories attribution and its separate dataset terms. Docker metadata follows the [official Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).
''',encoding="utf-8")
    readme=(out/"README.md").read_text(encoding="utf-8")
    readme=readme.replace("13.44M-parameter",f"{parameters/1e6:.2f}M-parameter")
    if not (out/"comparison.json").exists():
        readme=readme.replace("`comparison.json`", "`selection.json`" if (out/"selection.json").exists() else "the model card")
    (out/"README.md").write_text(readme,encoding="utf-8")
    checks={str(f.relative_to(out)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
    (out/"SHA256SUMS.json").write_text(json.dumps(checks,indent=2))
    print(out)


if __name__=="__main__":main()
