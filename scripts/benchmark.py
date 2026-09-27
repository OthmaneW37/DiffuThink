"""Compare denoising budgets on identical held-out templates."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffuthink.data import load_data
from diffuthink.engine import load_checkpoint, write_json
from diffuthink.model import corrupt, display, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/demo/best.pt")
    parser.add_argument("--data", default="data/synthetic.txt")
    parser.add_argument("--output", default="reports/sampling.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint)
    data, manifest = load_data(args.data, model.config.max_length, checkpoint["seed"])
    if manifest != checkpoint["manifest"]:
        raise ValueError("Dataset differs from training manifest")
    clean = data["test"][:24]
    noisy, selected = corrupt(clean, torch.full((len(clean),), 0.5), torch.Generator().manual_seed(314))
    generate(model, "The ~~~ runs today.", temperature=0)
    results = []
    for steps in (1, 4, 12):
        correct = total = exact = 0
        examples = []
        start = time.perf_counter()
        for target, source, mask in zip(clean, noisy, selected):
            template = display(source.tolist())
            result = generate(model, template, steps=steps, temperature=0)
            truth = display(target.tolist())
            prediction = result["tokens"]
            for i in mask.nonzero().flatten().tolist():
                correct += int(prediction[i] == int(target[i]))
                total += 1
            exact += int(prediction == target[:len(prediction)].tolist())
            examples.append({"template": template, "target": truth, "prediction": result["text"]})
        results.append({"steps": steps, "masked_accuracy": correct / total, "exact_match": exact / len(clean),
                        "mean_latency_ms": (time.perf_counter() - start) * 1000 / len(clean), "examples": examples})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"examples": len(clean), "mask_seed": 314, "temperature": 0, "device": "cpu", "results": results})
    for row in results:
        print({k: v for k, v in row.items() if k != "examples"})


if __name__ == "__main__":
    main()
