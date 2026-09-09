import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tokenizer_v0 import (
    ByteTokenizer,
    VOCAB_SIZE,
)

from dataset_v0 import (
    ByteTextDataset,
)

from diffusion_v0 import (
    corrupt_batch,
    sample_timesteps,
    visualize_tokens,
)

from multihead_attention_v0 import (
    MultiHeadSelfAttention,
)


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

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

    # =============================
    # 1. CLEAN TOKENS
    # =============================

    clean_tokens = next(
        iter(dataloader)
    ).to(device)

    # =============================
    # 2. TIMESTEPS
    # =============================

    B, T = clean_tokens.shape

    timesteps = sample_timesteps(
        batch_size=B,
        device=device,
    )

    # =============================
    # 3. DIFFUSION
    # =============================

    (
        noisy_tokens,
        corruption_mask,
    ) = corrupt_batch(
        clean_tokens,
        timesteps,
    )

    # =============================
    # 4. EMBEDDING
    # =============================

    D_MODEL = 64

    embedding = nn.Embedding(
        VOCAB_SIZE,
        D_MODEL,
    ).to(device)

    x = embedding(
        noisy_tokens
    )

    # =============================
    # 5. SELF ATTENTION
    # =============================

    attention = MultiHeadSelfAttention(
        d_model=D_MODEL,
        n_heads=4,
    ).to(device)

    contextualized, weights = (
        attention(x)
    )

    # =============================
    # PRINT
    # =============================

    print(
        "\nclean_tokens :"
    )
    print(clean_tokens.shape)

    print(
        "\nnoisy_tokens :"
    )
    print(noisy_tokens.shape)

    print(
        "\ntimesteps :"
    )
    print(timesteps.shape)

    print(
        "\nembeddings :"
    )
    print(x.shape)

    print(
        "\nafter attention :"
    )
    print(contextualized.shape)

    print(
        "\nattention weights :"
    )
    print(weights.shape)

    print(
        "\nPremier texte bruité :"
    )

    print(
        visualize_tokens(
            tokenizer,
            noisy_tokens[0]
            .cpu()
            .tolist(),
        )
    )


if __name__ == "__main__":
    main()