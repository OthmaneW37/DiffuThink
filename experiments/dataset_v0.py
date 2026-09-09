# experiments/dataset_v0.py

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer_v0 import ByteTokenizer


class ByteTextDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: ByteTokenizer,
        sequence_length: int = 64,
        stride: int = 64,
    ):
        self.sequence_length = sequence_length
        self.stride = stride

        text = Path(file_path).read_text(
            encoding="utf-8"
        )

        token_ids = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        if len(self.tokens) < sequence_length:
            raise ValueError(
                "Le dataset contient moins de tokens "
                "que sequence_length."
            )

    def __len__(self):
        usable_tokens = (
            len(self.tokens)
            - self.sequence_length
        )

        return (
            usable_tokens // self.stride
        ) + 1

    def __getitem__(self, index):
        start = index * self.stride

        end = start + self.sequence_length

        clean_tokens = self.tokens[start:end]

        return clean_tokens.clone()


def main():
    tokenizer = ByteTokenizer()

    dataset = ByteTextDataset(
        file_path="data/tiny.txt",
        tokenizer=tokenizer,
        sequence_length=64,
        stride=64,
    )

    print("Nombre total de tokens :")
    print(len(dataset.tokens))

    print("\nNombre de séquences :")
    print(len(dataset))

    sample = dataset[0]

    print("\nShape d'un exemple :")
    print(sample.shape)

    print("\nPremiers tokens :")
    print(sample[:20])

    print("\nTexte correspondant :")
    print(
        tokenizer.decode(
            sample.tolist()
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    batch = next(iter(dataloader))

    print("\nShape du batch :")
    print(batch.shape)

    print("\nPremier exemple du batch :")
    print(
        tokenizer.decode(
            batch[0].tolist()
        )
    )


if __name__ == "__main__":
    main()