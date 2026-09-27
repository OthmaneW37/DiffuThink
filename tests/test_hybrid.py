import unittest
from unittest.mock import patch
import torch
from diffuthink.v2.model import Config,Denoiser,BOS,EOS,PAD,MASK
from diffuthink.v2.hybrid import generate,causal_evaluate,banned_completions,sentence_ended


class HybridTests(unittest.TestCase):
    def test_fast_causal_padding_logits_and_gradients(self):
        x=torch.tensor([[BOS,8,9,EOS,PAD,PAD],[BOS,7,8,9,10,EOS]])
        selected=x.ne(PAD)
        expected=self.model(x,torch.zeros(2),selected,causal=True)
        expected.square().mean().backward()
        gradients={n:p.grad.clone() for n,p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        actual=self.model(x,torch.zeros(2),selected,causal=True,right_padded=True)
        actual.square().mean().backward()
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)
        for n,p in self.model.named_parameters():
            if n in gradients:torch.testing.assert_close(gradients[n],p.grad,atol=1e-5,rtol=1e-5)
    def test_extension_preserves_old_predictions(self):
        tokens=torch.tensor([[BOS,8,9,EOS]])
        expected=self.model(tokens,torch.zeros(1),causal=True)
        self.model.extend_context(64)
        actual=self.model(tokens,torch.zeros(1),causal=True)
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
        self.assertEqual(self.model.position.num_embeddings,64)
        self.assertTrue(torch.isfinite(self.model(torch.ones(1,64,dtype=torch.long),torch.zeros(1),causal=True)).all())

    def test_sentence_extension_is_bounded(self):
        class Tokenizer:
            def encode(self,text):return type("Encoded",(),{"ids":[8]})()
            def decode(self,ids,**kwargs):return "unfinished"
        with torch.no_grad():
            for p in self.model.parameters():p.zero_()
        result=generate(self.model,Tokenizer(),"prompt",max_new_tokens=2,finish_sentence_tokens=3,temperature=0,precision="fp32",no_repeat_ngram=0)
        self.assertEqual(result["generated_tokens"],5)
        self.assertEqual(result["stop_reason"],"length")
        self.assertTrue(sentence_ended('He said, "Yes!"'))
        self.assertFalse(sentence_ended('He reached the'))

    def test_sentence_extension_stops_at_boundary_without_rewriting(self):
        class Tokenizer:
            def encode(self,text):return type("Encoded",(),{"ids":[8]})()
            def decode(self,ids,**kwargs):return "word "*(len(ids)-2)+("." if len(ids)>=6 else "")
        def logits(*args,**kwargs):
            values=torch.zeros(1,40);values[0,8]=10
            return values
        with patch.object(self.model,"forward",side_effect=logits):
            result=generate(self.model,Tokenizer(),"prompt",max_new_tokens=2,finish_sentence_tokens=4,
                            temperature=0,precision="fp32",no_repeat_ngram=0)
        self.assertEqual(result["generated_tokens"],4)
        self.assertEqual(result["stop_reason"],"sentence_boundary")
        self.assertEqual(result["text"],"word word word word .")

    def test_eos_is_reported_and_not_emitted_as_text(self):
        class Tokenizer:
            def encode(self,text):return type("Encoded",(),{"ids":[8]})()
            def decode(self,ids,**kwargs):return str(ids)
        def logits(*args,**kwargs):
            values=torch.zeros(1,40);values[0,8]=10;values[0,EOS]=20
            return values
        with patch.object(self.model,"forward",side_effect=logits):
            result=generate(self.model,Tokenizer(),"prompt",max_new_tokens=20,temperature=0,
                            precision="fp32",no_repeat_ngram=0)
        self.assertEqual(result["generated_tokens"],8)
        self.assertEqual(result["stop_reason"],"eos")
        self.assertNotIn(EOS,result["tokens"])
    def test_ngram_blocking_only_for_existing_prefix(self):
        self.assertEqual(banned_completions([8,9,10,11,8,9,10],4),{11})
        self.assertEqual(banned_completions([8,9,10,11,8,9,12],4),set())
        self.assertEqual(banned_completions([8,8,8],0),set())
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        self.model=Denoiser(Config(vocab_size=40,width=32,heads=4,layers=2,max_length=32)).eval()

    def test_no_information_from_future(self):
        a=torch.tensor([[BOS,8,9,10,11,EOS]])
        b=torch.tensor([[BOS,8,9,27,28,EOS]])
        time=torch.zeros(1)
        first=self.model(a,time,causal=True)
        second=self.model(b,time,causal=True)
        torch.testing.assert_close(first[:,:3],second[:,:3],atol=0,rtol=0)
        self.assertFalse(torch.allclose(self.model(a,time)[:,:3],self.model(b,time)[:,:3]))

    def test_full_causal_matches_prefix(self):
        a=torch.tensor([[BOS,8,9,10,11,EOS]])
        full=self.model(a,torch.zeros(1),causal=True)
        prefix=self.model(a[:,:3],torch.zeros(1),causal=True)
        torch.testing.assert_close(full[:,:3],prefix,atol=1e-5,rtol=1e-5)

    def test_tokenizer_boundary_and_determinism(self):
        class Tokenizer:
            def encode(self,text):
                self.encoded=text
                return type("Encoded",(),{"ids":[8,9]})()
            def decode(self,ids,**kwargs): return str(ids)
        t=Tokenizer()
        a=generate(self.model,t,"a prompt   ",max_new_tokens=8,temperature=0,precision="fp32")
        b=generate(self.model,t,"a prompt",max_new_tokens=8,temperature=0,precision="fp32",seed=51)
        self.assertEqual(t.encoded,"a prompt")
        self.assertEqual(a["tokens"],b["tokens"])
        self.assertEqual(a["tokens"][:3],[BOS,8,9])
        self.assertNotIn(MASK,a["tokens"])
        self.assertNotIn(PAD,a["tokens"])
        self.assertEqual(a["mode"],"autoregressive")


if __name__=="__main__":unittest.main()
