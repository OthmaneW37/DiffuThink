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


def main():

    torch.manual_seed(42)

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

    print("Device :")
    print(device)

    # -----------------------------------
    # 1. Récupérer un batch propre
    # -----------------------------------

    clean_tokens = next(
        iter(dataloader)
    )

    clean_tokens = (
        clean_tokens.to(device)
    )

    batch_size = (
        clean_tokens.shape[0]
    )

    # -----------------------------------
    # 2. Tirer un timestep par séquence
    # -----------------------------------

    timesteps = sample_timesteps(
        batch_size=batch_size,
        device=device,
    )

    # -----------------------------------
    # 3. Créer les versions bruitées
    # -----------------------------------

    noisy_tokens, corruption_mask = (
        corrupt_batch(
            clean_tokens,
            timesteps,
        )
    )

    # -----------------------------------
    # 4. Afficher les shapes
    # -----------------------------------

    print(
        "\nclean_tokens shape :"
    )

    print(
        clean_tokens.shape
    )

    print(
        "\nnoisy_tokens shape :"
    )

    print(
        noisy_tokens.shape
    )

    print(
        "\ntimesteps shape :"
    )

    print(
        timesteps.shape
    )

    print(
        "\ncorruption_mask shape :"
    )

    print(
        corruption_mask.shape
    )

    # -----------------------------------
    # 5. Visualiser chaque séquence
    # -----------------------------------

    for i in range(batch_size):

        clean_text = visualize_tokens(
            tokenizer,
            clean_tokens[i]
            .cpu()
            .tolist(),
        )

        noisy_text = visualize_tokens(
            tokenizer,
            noisy_tokens[i]
            .cpu()
            .tolist(),
        )

        masked_ratio = (
            corruption_mask[i]
            .float()
            .mean()
            .item()
        )

        print("\n" + "=" * 80)

        print(
            f"EXEMPLE {i}"
        )

        print(
            f"t = "
            f"{timesteps[i].item():.3f}"
        )

        print(
            f"Ratio MASK = "
            f"{masked_ratio:.2%}"
        )

        print("\nCLEAN :")
        print(clean_text)

        print("\nNOISY :")
        print(noisy_text)


if __name__ == "__main__":
    main()