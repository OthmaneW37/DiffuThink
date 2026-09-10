import torch

from torch.utils.data import DataLoader

from tokenizer_v0 import (
    ByteTokenizer,
)

from dataset_v0 import (
    ByteTextDataset,
)

from diffusion_v0 import (
    corrupt_batch,
    sample_timesteps,
    visualize_tokens,
)

from diffuthink_backbone_v0 import (
    DiffuThinkBackbone,
)


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    # =====================================
    # DATA
    # =====================================

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

    clean_tokens = next(
        iter(dataloader)
    ).to(device)

    B, T = clean_tokens.shape

    # =====================================
    # DIFFUSION
    # =====================================

    timesteps = sample_timesteps(
        batch_size=B,
        device=device,
    )

    (
        noisy_tokens,
        corruption_mask,
    ) = corrupt_batch(
        clean_tokens,
        timesteps,
    )

    # =====================================
    # MODEL
    # =====================================

    model = DiffuThinkBackbone(
        vocab_size=260,
        d_model=64,
        n_heads=4,
        hidden_dim=256,
        n_layers=4,
    ).to(device)

    (
        hidden_states,
        attention_maps,
    ) = model(
        noisy_tokens,
        timesteps,
    )

    # =====================================
    # RESULTS
    # =====================================

    print("\nclean_tokens :")
    print(clean_tokens.shape)

    print("\nnoisy_tokens :")
    print(noisy_tokens.shape)

    print("\ntimesteps :")
    print(timesteps)

    print("\nhidden_states :")
    print(hidden_states.shape)

    print(
        "\nNombre de Transformer Blocks :"
    )
    print(len(model.blocks))

    print(
        "\nNombre de matrices d'attention :"
    )
    print(len(attention_maps))

    print(
        "\nShape attention layer 0 :"
    )
    print(
        attention_maps[0].shape
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nNombre de paramètres :")
    print(parameter_count)

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