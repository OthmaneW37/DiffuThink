import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                "d_model doit être pair."
            )

        self.d_model = d_model

        self.linear1 = nn.Linear(
            d_model,
            d_model * 4,
        )

        self.linear2 = nn.Linear(
            d_model * 4,
            d_model,
        )

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        timesteps :
            [B]

        output :
            [B,C]
        """

        half_dim = (
            self.d_model // 2
        )

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / max(
                half_dim - 1,
                1,
            )
        )

        # On étale t pour avoir
        # des variations sinusoïdales suffisantes.
        scaled_t = (
            timesteps.float()
            * 1000.0
        )

        angles = (
            scaled_t[:, None]
            * frequencies[None, :]
        )

        embedding = torch.cat(
            (
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        )

        # [B,C]

        embedding = self.linear1(
            embedding
        )

        embedding = F.silu(
            embedding
        )

        embedding = self.linear2(
            embedding
        )

        return embedding


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    timesteps = torch.tensor(
        [
            0.1,
            0.3,
            0.6,
            0.9,
        ],
        device=device,
    )

    time_embedding = TimeEmbedding(
        d_model=64,
    ).to(device)

    output = time_embedding(
        timesteps
    )

    print("Timesteps :")
    print(timesteps)

    print("\nTimesteps shape :")
    print(timesteps.shape)

    print("\nTime embedding shape :")
    print(output.shape)

    print("\nEmbedding pour t=0.1 :")
    print(output[0])

    print("\nEmbedding pour t=0.9 :")
    print(output[3])


if __name__ == "__main__":
    main()