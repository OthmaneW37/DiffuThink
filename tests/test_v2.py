from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from diffuthink.v2.data import digest
from diffuthink.v2.model import Config, Denoiser, PAD, BOS, EOS, MASK, SPECIAL_TOKENS, corrupt
from diffuthink.v2.inference import denoise
from diffuthink.v2.train import train, atomic_save


class V2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = Denoiser(Config(vocab_size=300, width=32, heads=4, layers=1, max_length=16))

    def test_fully_masked_positions_have_distinct_predictions(self):
        tokens = torch.tensor([[BOS, MASK, MASK, MASK]])
        logits = self.model(tokens, torch.ones(1))
        self.assertFalse(torch.allclose(logits[:, 1], logits[:, 2]))

    def test_corruption_protects_boundary_and_padding(self):
        clean = torch.tensor([[BOS, 9, 10, EOS, PAD], [BOS, 8, EOS, PAD, PAD]])
        noisy, selected, t = corrupt(clean, torch.Generator().manual_seed(1))
        self.assertFalse(selected[:, 0].any())
        self.assertFalse(selected[clean.eq(PAD)].any())
        self.assertTrue(selected.any(-1).all())
        torch.testing.assert_close(t, selected.sum(-1) / (clean.ne(PAD) & clean.ne(BOS)).sum(-1))
        self.assertTrue(noisy[selected].eq(MASK).all())

    def test_selected_projection_matches_full(self):
        tokens = torch.tensor([[BOS, MASK, 6, EOS]])
        selected = tokens.eq(MASK)
        torch.testing.assert_close(self.model(tokens, torch.ones(1), selected), self.model(tokens, torch.ones(1))[selected])

    def test_padding_invariance(self):
        tokens = torch.tensor([[BOS, MASK, 6, EOS]])
        padded = torch.cat((tokens, torch.full((1, 3), PAD)), -1)
        torch.testing.assert_close(self.model(tokens, torch.ones(1)), self.model(padded, torch.ones(1))[:, :4], atol=1e-5, rtol=1e-5)

    def test_safetensors_roundtrip(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            other = Denoiser.from_pretrained(path)
            for a, b in zip(self.model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b)

    def test_sampler_context_reproducibility_and_completion(self):
        class Display:
            def decode(self, ids, **kwargs):
                return str(ids)
        ids = [BOS, 9, MASK, MASK, MASK, 10, EOS]
        a = denoise(self.model, Display(), ids, steps=2, seed=7)
        b = denoise(self.model, Display(), ids, steps=2, seed=7)
        self.assertEqual(a["tokens"], b["tokens"])
        self.assertNotIn(MASK, a["tokens"])
        for i in (0, 1, 5, 6):
            self.assertEqual(a["tokens"][i], ids[i])
        self.assertEqual(a["trace"][-1]["remaining"], 0)
        self.assertLessEqual(a["forward_passes"], 2)

    def test_exact_cpu_resume(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data"
            data.mkdir()
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            tokenizer.train_from_iterator(["Once upon a time, a cat ran home."], trainers.BpeTrainer(vocab_size=280,
                special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tokenizer.save(str(data / "tokenizer.json"))
            manifest = {"vocab_size": tokenizer.get_vocab_size(), "max_length": 8,
                        "tokenizer_sha256": digest(data / "tokenizer.json"), "splits": {}}
            for name in ("train", "validation", "test"):
                np.save(data / f"{name}.npy", np.array([[BOS, 10, 11, 12, 13, EOS, PAD, PAD], [BOS, 11, 12, 10, EOS, PAD, PAD, PAD]], dtype=np.uint16))
                manifest["splits"][name] = {"array_sha256": digest(data / f"{name}.npy")}
            (data / "manifest.json").write_text(json.dumps(manifest))
            args = Namespace(command="train",data=str(data),output=str(root / "full"),resume=None,device="cpu",precision="fp32",
                steps=4,batch_size=2,accumulation=1,eval_every=2,eval_samples=2,test_samples=2,threads=2,seed=42,
                width=32,heads=4,layers=1,lr=0.001,warmup=1)
            def capture(state, path):
                atomic_save(state, path)
                if state["step"] == 2:
                    shutil.copy2(path, root / "middle.pt")
            with patch("diffuthink.v2.train.atomic_save", capture):
                train(args)
            args.resume = str(root / "middle.pt")
            args.output = str(root / "resumed")
            # Best weights must accompany a recovery state; the CLI preserves
            # them in the same run. Copy here to emulate that layout.
            shutil.copytree(root / "full" / "best", root / "resumed" / "best")
            train(args)
            full = torch.load(root / "full" / "last.pt", weights_only=True)
            resumed = torch.load(root / "resumed" / "last.pt", weights_only=True)
            self.assertEqual(full["seen_tokens"], resumed["seen_tokens"])
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
