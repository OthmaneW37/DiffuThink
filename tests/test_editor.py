import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from diffuthink.v2.model import Config,Denoiser,SPECIAL_TOKENS,BOS,EOS,MASK
from diffuthink.v2.editor import rewrite_span
from diffuthink.v2.inference import denoise


class EditorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        self.tokenizer=Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder=decoders.ByteLevel()
        self.tokenizer.train_from_iterator(["Élodie was happy yesterday. She cried. She was sad. "*10],
            trainers.BpeTrainer(vocab_size=400,special_tokens=SPECIAL_TOKENS,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=1))
        self.model=Denoiser(Config(vocab_size=self.tokenizer.get_vocab_size(),width=32,heads=4,layers=1,max_length=128)).eval()

    def test_surrounding_text_and_whitespace_preserved_exactly(self):
        token=self.tokenizer.encode(" sad").ids
        self.assertEqual(len(token),1)
        def proposals(model,tokenizer,ids,**kwargs):
            return {"tokens":[token[0] if i==MASK else i for i in ids],"trace":[]}
        text="Élodie was  happy  yesterday.\nShe cried."
        start=text.index("  happy");end=start+len("  happy  ")
        with patch("diffuthink.v2.editor.denoise",side_effect=proposals):
            result=rewrite_span(self.model,self.tokenizer,text,start,end)
        self.assertGreater(len(result["candidates"]),0)
        for item in result["candidates"]:
            self.assertEqual(item["text"],result["prefix"]+item["replacement"]+result["suffix"])
            self.assertTrue(item["text"].startswith(text[:start]+"  "))
            self.assertTrue(item["text"].endswith("  "+text[end:]))
        self.assertEqual(len(set(c["replacement"] for c in result["candidates"])),len(result["candidates"]))
        self.assertEqual(result["original"],text)

    def test_reject_partial_words_and_invalid_selection(self):
        with self.assertRaisesRegex(ValueError,"complete words"):
            rewrite_span(self.model,self.tokenizer,"She was happy.",9,11)
        with self.assertRaises(ValueError):
            rewrite_span(self.model,self.tokenizer,"hello",0,99)
        with self.assertRaises(ValueError):
            rewrite_span(self.model,self.tokenizer,"hello",0,0)

    def test_eos_cannot_replace_an_editor_mask(self):
        def logits(tokens,*args,**kwargs):
            scores=torch.zeros(1,tokens.shape[1],self.model.config.vocab_size)
            scores[:,:,EOS]=10;scores[:,:,8]=5
            return scores
        with patch.object(self.model,"forward",side_effect=logits):
            result=denoise(self.model,self.tokenizer,[BOS,MASK,EOS],temperature=0,allow_eos=False)
        self.assertEqual(result["tokens"],[BOS,8,EOS])


if __name__=="__main__":unittest.main()
