"""Check exact CPU recovery of the hybrid trainer on a tiny local fixture."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from diffuthink.v2.model import Config,Denoiser,SPECIAL_TOKENS,BOS,EOS,PAD
from diffuthink.v2.data import digest
from diffuthink.v2.hybrid import train
from diffuthink.v2.train import atomic_save


def main():
    expanded="--expanded" in sys.argv
    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as root:
        root=Path(root);data=root/"data";data.mkdir();source=root/"source"
        tokenizer=Tokenizer(models.BPE(unk_token="[UNK]"));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel()
        tokenizer.decoder=decoders.ByteLevel()
        tokenizer.train_from_iterator(["A little bird flew home."],trainers.BpeTrainer(vocab_size=280,special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tokenizer.save(str(data/"tokenizer.json"))
        manifest={"dataset":"fixture","revision":"fixed","vocab_size":tokenizer.get_vocab_size(),"max_length":16 if expanded else 8,"tokenizer_sha256":digest(data/"tokenizer.json"),"splits":{}}
        for split in ("train","validation","test"):
            rows=np.array([[BOS,10,11,12,EOS,PAD,PAD,PAD],[BOS,13,12,10,11,EOS,PAD,PAD]],dtype=np.uint16)
            if expanded:rows=np.pad(rows,((0,0),(0,8)))
            np.save(data/f"{split}.npy",rows)
            manifest["splits"][split]={"array_sha256":digest(data/f"{split}.npy"),"documents_sha256":"fixed"}
        (data/"manifest.json").write_text(json.dumps(manifest))
        model=Denoiser(Config(vocab_size=tokenizer.get_vocab_size(),width=32,heads=4,layers=1,max_length=8));model.save_pretrained(source)
        previous=json.loads(json.dumps(manifest));previous["max_length"]=8
        (source/"training_info.json").write_text(json.dumps({"manifest":previous,"pretrained_weights":False}))
        args=Namespace(source=str(source),data=str(data),output=str(root/"run"),device="cpu",steps=12,batch_size=2,lr=.001,
            warmup=2,eval_every=6,eval_samples=2,seed=3,resume=False,expand_data=expanded)
        def save(state,path):
            atomic_save(state,path)
            if state["step"]==6:
                shutil.copy2(path,root/"middle.pt")
                shutil.copytree(root/"run"/"best",root/"middle-best")
        with contextlib.redirect_stdout(io.StringIO()):
            wall_start=time.perf_counter()
            with patch("diffuthink.v2.hybrid.atomic_save",save):train(args)
            duration=json.loads((root/"run"/"report.json").read_text())["seconds"]
            assert 0<=duration<=time.perf_counter()-wall_start
            full=torch.load(root/"run"/"last.pt",weights_only=True)
            shutil.copy2(root/"middle.pt",root/"run"/"last.pt")
            shutil.copytree(root/"middle-best",root/"run"/"best",dirs_exist_ok=True)
            args.resume=True
            train(args)
            resumed=torch.load(root/"run"/"last.pt",weights_only=True)
        assert full["seen_tokens"]==resumed["seen_tokens"]
        for key in full["model"]:torch.testing.assert_close(full["model"][key],resumed["model"][key],atol=0,rtol=0)
        print("Hybrid causal + denoising recovery: exact CPU weight match")


if __name__=="__main__":main()
