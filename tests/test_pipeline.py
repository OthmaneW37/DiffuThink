import tempfile
import unittest
from pathlib import Path
import torch

from diffuthink.data import load_data, make_corpus
from diffuthink.engine import evaluate, load_checkpoint
from diffuthink.model import DiffuThink, ModelConfig, PAD, MASK, encode, decode, corrupt, masked_loss, generate, rope
from dataclasses import asdict


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model = DiffuThink(ModelConfig(width=32, heads=4, layers=1, max_length=32))

    def test_utf8_roundtrip(self):
        self.assertEqual(decode(encode("Été à 東京 🌍")), "Été à 東京 🌍")

    def test_padding_and_forced_target(self):
        clean = torch.tensor([[65, PAD, PAD], [66, 67, PAD]])
        noisy, selected = corrupt(clean, torch.zeros(2))
        self.assertTrue(selected.any(-1).all())
        self.assertFalse(selected[clean.eq(PAD)].any())
        self.assertTrue(noisy[selected].eq(MASK).all())

    def test_rope_preserves_norm(self):
        x = torch.randn(2, 4, 8, 8)
        torch.testing.assert_close(rope(x).square().sum(-1), x.square().sum(-1))

    def test_padding_does_not_change_predictions(self):
        ids = torch.tensor([encode("Hello")])
        time = torch.tensor([0.5])
        padded = torch.cat((ids, torch.full((1, 4), PAD)), 1)
        torch.testing.assert_close(self.model(ids, time), self.model(padded, time)[:, :5], atol=1e-5, rtol=1e-5)

    def test_loss_ignores_unselected_positions(self):
        logits = torch.randn(1, 3, 256)
        targets = torch.tensor([[1, 2, PAD]])
        selected = torch.tensor([[True, False, False]])
        first = masked_loss(logits, targets, selected)
        logits[:, 1:] = 100
        torch.testing.assert_close(first, masked_loss(logits, targets, selected))

    def test_sampler_context_completion_and_seed(self):
        a = generate(self.model, "Hi ~~~~~!", steps=3, seed=99)
        b = generate(self.model, "Hi ~~~~~!", steps=3, seed=99)
        self.assertEqual(a, b)
        self.assertTrue(a["text"].startswith("Hi "))
        self.assertTrue(a["text"].endswith("!"))
        self.assertEqual([x["remaining"] for x in a["trace"]], [3, 1, 0])
        self.assertEqual([x["remaining"] for x in generate(self.model, "Hi ~~~~~!", steps=12)["trace"]], [4, 3, 2, 1, 0])
        self.assertEqual(generate(self.model, "Hi!")["text"], "Hi!")
        with self.assertRaises(ValueError):
            generate(self.model, "~" * 33)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": asdict(self.model.config), "model": self.model.state_dict()}, path)
            restored, _ = load_checkpoint(path)
            self.assertEqual(generate(self.model, "a~~~", seed=5), generate(restored, "a~~~", seed=5))

    def test_split_and_evaluation_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            make_corpus(path)
            data, manifest = load_data(path, 64, 42)
            sets = {k: set(tuple(row.tolist()) for row in v) for k, v in data.items()}
            self.assertFalse(sets["train"] & sets["test"])
            self.assertFalse(sets["train"] & sets["validation"])
            self.assertFalse(sets["test"] & sets["validation"])
            self.assertEqual(manifest, load_data(path, 64, 42)[1])
            model = DiffuThink(ModelConfig(width=32, layers=1))
            frequencies = torch.ones(256) / 256
            a = evaluate(model, data["test"][:4], frequencies)
            self.assertEqual(a, evaluate(model, data["test"][:4], frequencies))

    def test_can_overfit_small_batch(self):
        clean = torch.tensor([encode("the cat")])
        noisy = torch.full_like(clean, MASK)
        selected = torch.ones_like(clean, dtype=torch.bool)
        noise = torch.ones(1)
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = masked_loss(self.model(noisy, noise), clean, selected).item()
        for _ in range(50):
            opt.zero_grad()
            loss = masked_loss(self.model(noisy, noise), clean, selected)
            loss.backward()
            opt.step()
        # Fully masked identical inputs cannot identify absolute positions via
        # relative RoPE alone; they can still learn the small byte distribution.
        self.assertLess(loss.item(), initial * 0.6)


if __name__ == "__main__":
    unittest.main()
