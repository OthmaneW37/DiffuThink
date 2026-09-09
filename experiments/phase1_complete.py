# experiments/phase1_complete.py

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tokenizer_v0 import (
    ByteTokenizer,
    VOCAB_SIZE,
)

from dataset_v0 import ByteTextDataset


def main():
    tokenizer = ByteTokenizer()

    dataset = ByteTextDataset(
        file_path="data/tiny.txt",
        tokenizer=tokenizer,
        sequence_length=64,
        stride=64,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    batch = next(iter(dataloader))

    print(
        "Avant GPU :",
        batch.device,
    )

    batch = batch.to(device)

    print(
        "Après GPU :",
        batch.device,
    )

    print(
        "Shape tokens :",
        batch.shape,
    )

    d_model = 64

    token_embedding = nn.Embedding(
        VOCAB_SIZE,
        d_model,
    ).to(device)

    hidden_states = token_embedding(batch)

    print(
        "Shape hidden states :",
        hidden_states.shape,
    )

    print(
        "\nTexte du premier exemple :"
    )

    print(
        tokenizer.decode(
            batch[0].cpu().tolist()
        )
    )


if __name__ == "__main__":
    main()