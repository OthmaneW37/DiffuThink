import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from diffuthink.v2.model import Config, SPECIAL_TOKENS
from diffuthink.storypatch.ranker import SpanRanker, Reranker, encode_candidate, collate
from diffuthink.storypatch.corrections import build, read, editing_cases
from diffuthink.storypatch.evaluate_ranker import calibrate, decision
from diffuthink.storypatch.corrections import alternatives


class RankerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(91)
        self.tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder=decoders.ByteLevel()
        self.tokenizer.train_from_iterator(['Élodie was happy. She was sad. The toy was blue. '*10],
            trainers.BpeTrainer(vocab_size=350,special_tokens=SPECIAL_TOKENS,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        self.model=SpanRanker(Config(vocab_size=self.tokenizer.get_vocab_size(),width=32,heads=4,layers=1,max_length=128)).eval()
        self.info={'tokenizer_sha256':hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest(),
            'calibration':{'margin_threshold':1.,'score_floor':0.}}

    def test_scores_are_padding_invariant_and_span_tracks_unicode(self):
        text='Élodie was happy.'; a=text.index('happy')
        row=encode_candidate(self.tokenizer,text,a,a+5,128)
        marked=[i for i,flag in enumerate(row[1]) if flag]
        self.assertEqual(self.tokenizer.decode([row[0][i] for i in marked]).strip(),'happy')
        other=encode_candidate(self.tokenizer,text*3,a,a+5,128)
        with torch.inference_mode():
            solo=self.model(*collate([row],'cpu'))[0]
            padded=self.model(*collate([row,other],'cpu'))[0]
        torch.testing.assert_close(solo,padded,atol=1e-6,rtol=1e-5)

    def test_roundtrip_and_tokenizer_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory,self.info)
            loaded=Reranker.load(directory,self.tokenizer)
            text='She was sad.';a=text.index('sad')
            reference=Reranker(self.model,self.tokenizer,self.info).scores([(text,a,a+3)])
            self.assertEqual(loaded.scores([(text,a,a+3)]),reference)
            info={**self.info,'tokenizer_sha256':'wrong'}
            (Path(directory)/'ranker_info.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'fingerprint'):Reranker.load(directory,self.tokenizer)

    def test_keep_original_at_threshold_or_without_candidates(self):
        ranker=Reranker(self.model,self.tokenizer,self.info)
        result={'original':'She was happy.','selection':{'start':8,'end':13,'text':'happy'},
                'candidates':[{'replacement':'sad','text':'She was sad.'}]}
        ranker.scores=lambda texts:[0.,1.]
        ranked=ranker.rank(result)
        self.assertEqual(ranked['recommendation']['action'],'keep')
        ranker.scores=lambda texts:[0.,1.01]
        self.assertEqual(ranker.rank(result)['recommendation']['action'],'replace')
        result['candidates']=[];ranker.scores=lambda texts:[0.]
        self.assertEqual(ranker.rank(result)['recommendation']['action'],'keep')

    def test_context_overflow_refuses_without_cropping(self):
        text='She was happy.'*100
        with self.assertRaisesRegex(ValueError,'context'):
            encode_candidate(self.tokenizer,text,8,13,16)
        result={'original':text,'selection':{'start':8,'end':13,'text':'happy'},'candidates':[]}
        answer=Reranker(self.model,self.tokenizer,self.info).rank(result)
        self.assertEqual(answer['recommendation']['reason'],'ranker_unavailable')

    def test_dataset_contexts_and_template_ids_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'data';manifest=build(root,train=60,validation=30,test_contexts=30)
            splits={s:read(root/f'{s}.jsonl') for s in manifest['splits']}
            seen=set();templates=set()
            for split,rows in splits.items():
                keys={r['context_sha256'] for r in rows}
                self.assertFalse(keys & seen);seen|=keys
                template_set=set(manifest['splits'][split]['templates'])
                self.assertFalse(templates & template_set);templates|=template_set
                for r in rows:
                    self.assertFalse(set(r['accepted']) & set(r['negative']))
                cases=list(editing_cases(rows))
                self.assertEqual(sum(c['clean'] for c in cases),len(rows))
                for c in cases:self.assertEqual(c['text'][:c['start']],c['prefix'])
            with self.assertRaises(ValueError):build(root)

    def test_calibration_can_abstain_on_bad_proposals(self):
        rows=[]
        for clean in (False,True):
            rows.append({'case':{'clean':clean,'accepted':['blue']},'original_fragment':'blue' if clean else 'red',
                         'original_score':0.,'candidates':[{'replacement':'yellow','ranker_score':2.}]})
        calibration=calibrate(rows)
        for r in rows:self.assertEqual(decision(r,calibration['margin_threshold'],calibration['score_floor']),r['original_fragment'])
        self.assertEqual(calibration['validation']['clean_modified'],0.)

    def test_intensity_alternatives_do_not_double_existing_very(self):
        self.assertIn('very sad',alternatives('She was ',' yesterday.','emotion',['sad']))
        self.assertNotIn('very sad',alternatives('She was very ',' yesterday.','emotion',['sad']))
        self.assertIn('underneath',alternatives('It was ',' the bed.','location',['under']))

    def test_ranker_training_cpu_resume_matches_uninterrupted_weights(self):
        from argparse import Namespace
        from unittest.mock import patch
        from diffuthink.storypatch.train_ranker import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build(root/'data',train=12,validation=6,test_contexts=6)
            self.tokenizer.save(str(root/'tokenizer.json'))
            args=Namespace(data=str(root/'data'),tokenizer=str(root/'tokenizer.json'),output=str(root/'run'),
                device='cpu',seed=17,width=16,lr=.001,steps=2,batch_pairs=2,eval_every=1,resume=None)
            save=torch.save
            def capture(state,path):
                save(state,path)
                if state['step']==1:save(state,root/'step1.pt')
            with patch('diffuthink.storypatch.train_ranker.torch.save',side_effect=capture):run(args)
            reference=torch.load(root/'run/resume.pt',weights_only=False)['model']
            args.resume=str(root/'step1.pt');run(args)
            resumed=torch.load(root/'run/resume.pt',weights_only=False)['model']
            for key in reference:torch.testing.assert_close(reference[key],resumed[key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
