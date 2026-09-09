# experiments/embeddings_v0.py

import torch
import torch.nn as nn

from tokenizer_v0 import (
    ByteTokenizer,
    PAD_ID,
    VOCAB_SIZE,
)


torch.manual_seed(42)


def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
) -> list[list[int]]:
    """
    Complète toutes les séquences afin qu'elles aient
    la même longueur.
    """

    max_length = max(len(sequence) for sequence in sequences)

    padded_sequences = []

    for sequence in sequences:
        number_of_padding_tokens = max_length - len(sequence)

        padded_sequence = sequence + (
            [pad_id] * number_of_padding_tokens
        )

        padded_sequences.append(padded_sequence)

    return padded_sequences


def main():
    tokenizer = ByteTokenizer()

    texts = [
        "Hello!",
        "Hello Othmane!",
    ]

    encoded_texts = []

    for text in texts:
        tokens = tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        encoded_texts.append(tokens)

    padded_texts = pad_sequences(
        encoded_texts,
        PAD_ID,
    )

    print("Tokens après padding :")

    for sequence in padded_texts:
        print(sequence)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice :", device)

    x = torch.tensor(
        padded_texts,
        dtype=torch.long,
        device=device,
    )

    print("\nShape de x :")
    print(x.shape)

    D_MODEL = 64

    embedding = nn.Embedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=D_MODEL,
    ).to(device)

    embedded_x = embedding(x)

    print("\nShape après embedding :")
    print(embedded_x.shape)

    print("\nShape de la table d'embeddings :")
    print(embedding.weight.shape)

    print("\nNombre de paramètres :")
    print(embedding.weight.numel())

    print("\nPremier token du premier texte :")
    print(x[0, 0])

    print("\nEmbedding correspondant :")
    print(embedded_x[0, 0])

    print("\nLes deux textes commencent par BOS.")

    same_embedding = torch.allclose(
        embedded_x[0, 0],
        embedded_x[1, 0],
    )

    print(
        "Même token → même embedding :",
        same_embedding,
    )


if __name__ == "__main__":
    main()