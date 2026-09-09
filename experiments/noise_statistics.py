import torch

from tokenizer_v0 import (
    ByteTokenizer,
)

from diffusion_v0 import (
    corrupt_batch,
)


def main():

    tokenizer = ByteTokenizer()

    text = (
        "This is a sufficiently long "
        "sentence for our experiment. "
    ) * 20

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    clean = torch.tensor(
        [tokens],
        dtype=torch.long,
    )

    trials = 1000

    noise_levels = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    ]

    for t in noise_levels:

        ratios = []

        for _ in range(trials):

            timestep = torch.tensor(
                [t]
            )

            _, mask = corrupt_batch(
                clean,
                timestep,
            )

            ratio = (
                mask.float()
                .mean()
                .item()
            )

            ratios.append(ratio)

        average_ratio = (
            sum(ratios)
            / len(ratios)
        )

        print(
            f"t={t:.2f} "
            f"→ masked moyen="
            f"{average_ratio:.3f}"
        )


if __name__ == "__main__":
    main()