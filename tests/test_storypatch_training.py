import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from diffuthink.v2.data import digest
from diffuthink.v2.model import BOS, EOS, PAD, MASK, SPECIAL_TOKENS
from diffuthink.storypatch.data import word_spans, quality
from diffuthink.storypatch.train import span_batch, parser, train
from diffuthink.storypatch.benchmark import cases


class StoryPatchTrainingTests(unittest.TestCase):
    def test_span_mask_only_touches_requested_tokens(self):
        clean = torch.tensor([[BOS,5,6,7,8,EOS,PAD], [BOS,9,10,11,EOS,PAD,PAD]])
        ranges = torch.tensor([[2,5], [1,3]])
        for partial in (False, True):
            noisy, selected, noise = span_batch(clean, ranges, torch.Generator().manual_seed(7), partial)
            self.assertTrue(torch.equal(noisy[~selected], clean[~selected]))
            self.assertTrue(noisy[selected].eq(MASK).all())
            self.assertTrue(selected.any(-1).all())
            self.assertFalse((selected & (clean.eq(PAD) | clean.eq(BOS) | clean.eq(EOS))).any())
            self.assertTrue(torch.equal(noise, selected.sum(-1)/(clean.ne(PAD) & clean.ne(BOS)).sum(-1)))

    def test_word_boundaries_include_subwords_and_skip_punctuation(self):
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        text = "Mia's extraordinary blue umbrella stayed outside, beside the garden."
        tokenizer.train_from_iterator([text], trainers.BpeTrainer(vocab_size=280, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        ids, spans = word_spans(text, tokenizer, np.random.default_rng(48), 8)
        self.assertTrue(spans)
        offsets = tokenizer.encode(text).offsets
        for a, b in spans:
            fragment = tokenizer.decode(ids[a-1:b-1]).strip()
            self.assertTrue(fragment[0].isalpha())
            self.assertTrue(fragment[-1].isalpha())
            x, y = offsets[a-1][0], offsets[b-2][1]
            self.assertFalse(x > 0 and text[x-1].isalpha() and text[x].isalpha())
            self.assertFalse(y < len(text) and text[y-1].isalpha() and text[y].isalpha())
            self.assertLessEqual(b-a, 16)

    def test_quality_rejects_corruption_and_heavy_repetition(self):
        self.assertFalse(quality('the dog ran '*30))
        self.assertFalse(quality('The little girl found a broken \ufffd toy near the house.'))
        self.assertTrue(quality('The little girl found a red ball and carried it home.'))

    def test_challenges_have_real_selection_and_no_answer_as_original(self):
        for case in cases():
            self.assertGreater(case['end'], case['start'])
            self.assertNotIn(case['text'][case['start']:case['end']], case['accepted'])
            self.assertNotIn('[', case['text'])

    def test_resume_matches_uninterrupted_cpu_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data'
            data.mkdir()
            (data/'tokenizer.json').write_text('{}')
            manifest = {'source_manifest': {'vocab_size': 32, 'max_length': 12},
                        'tokenizer_sha256': digest(data/'tokenizer.json'), 'splits': {}}
            for split in ('train','validation','test'):
                rows = np.array([[BOS, 5+i, 12, 13, 14, EOS]+[PAD]*6 for i in range(8)], dtype=np.uint16)
                spans = np.tile(np.array([[[2,4]]], dtype=np.uint16), (8,8,1))
                np.save(data/f'{split}.npy', rows)
                np.save(data/f'{split}_spans.npy', spans)
                manifest['splits'][split] = {'files': {f'{split}{suffix}.npy': digest(data/f'{split}{suffix}.npy') for suffix in ('','_spans')}}
            (data/'manifest.json').write_text(json.dumps(manifest))
            common = ['--data',str(data),'--device','cpu','--precision','fp32','--steps','6','--warmup','1',
                      '--batch-size','2','--eval-every','2','--eval-samples','2','--width','16','--layers','1']
            with contextlib.redirect_stdout(io.StringIO()):
                train(parser().parse_args(common+['--output',str(root/'full')]))
                train(parser().parse_args(common+['--output',str(root/'resumed'),'--stop-after','2']))
                train(parser().parse_args(common+['--output',str(root/'resumed'),'--resume']))
            full = torch.load(root/'full'/'last.pt', weights_only=True)
            resumed = torch.load(root/'resumed'/'last.pt', weights_only=True)
            for key, value in full['model'].items():
                self.assertTrue(torch.equal(value, resumed['model'][key]), key)
            for key in ('sampling_rng','noise_rng'):
                self.assertTrue(torch.equal(full[key], resumed[key]))
            self.assertEqual(full['seen_tokens'], resumed['seen_tokens'])
            self.assertEqual(full['supervised_tokens'], resumed['supervised_tokens'])


if __name__ == '__main__':
    unittest.main()
