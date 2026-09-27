import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from diffuthink.v2.data import prepare, load_arrays, digest
from diffuthink.v2.model import SPECIAL_TOKENS, BOS, EOS, PAD


class StoryWindowTests(unittest.TestCase):
    def test_frozen_tokenizer_and_true_document_end(self):
        stories=["A small bird flew home.","The little girl went to the park. "*20]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            tok=Tokenizer(models.BPE(unk_token="[UNK]"))
            tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder=decoders.ByteLevel()
            tok.train_from_iterator(stories,trainers.BpeTrainer(vocab_size=300,special_tokens=SPECIAL_TOKENS,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            frozen=root/"tokenizer.json";tok.save(str(frozen))
            args=Namespace(output=str(root/"data"),revision="pinned",stories=2,heldout=1,
                tokenizer_stories=2,vocab_size=300,length=32,tokenizer=str(frozen))
            response=Mock();response.json.return_value={"sha":"pinned"}
            with patch("diffuthink.v2.data.requests.get",return_value=response),patch(
                "diffuthink.v2.data.download_stories",side_effect=[[stories[0],stories[1]],stories]):
                prepare(args)
            arrays,manifest=load_arrays(args.output)
            self.assertEqual(manifest["tokenizer_sha256"],digest(frozen))
            expected=[]
            for story in stories:expected.extend(tok.encode(story).ids+[EOS])
            actual=[int(x) for row in arrays["train"] for x in row if x not in (BOS,PAD)]
            self.assertEqual(actual,expected)
            self.assertEqual(actual.count(EOS),2)
            self.assertEqual(manifest["splits"]["train"]["complete_story_windows"],1)
            self.assertEqual(manifest["max_length"],32)
            malformed=np.array(arrays["train"])
            for array in arrays.values():
                array._mmap.close()
            malformed[0,1]=PAD
            np.save(root/"data"/"train.npy",malformed)
            manifest["splits"]["train"]["array_sha256"]=digest(root/"data"/"train.npy")
            (root/"data"/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"right-padded"):
                load_arrays(args.output)


if __name__=="__main__":unittest.main()
