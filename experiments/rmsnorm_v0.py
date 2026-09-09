import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        # Un coefficient apprenable
        # pour chaque dimension.
        #
        # Shape :
        # [C]
        self.weight = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        """
        x :
            [B, T, C]

        output :
            [B, T, C]
        """

        # Carré de chaque valeur
        squared = x.pow(2)

        # Moyenne sur C seulement.
        #
        # [B,T,C]
        #     ↓
        # [B,T,1]
        mean_squared = squared.mean(
            dim=-1,
            keepdim=True,
        )

        # Root Mean Square
        rms = torch.sqrt(
            mean_squared + self.eps
        )

        # Normalisation
        normalized = x / rms

        # self.weight : [C]
        #
        # PyTorch broadcast automatiquement
        # vers [B,T,C].
        output = (
            normalized
            * self.weight
        )

        return output


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    B = 2
    T = 4
    C = 8

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    ) * 10

    norm = RMSNorm(
        d_model=C,
    ).to(device)

    y = norm(x)

    print("Input shape :")
    print(x.shape)

    print("\nOutput shape :")
    print(y.shape)

    print("\nPremier vecteur AVANT :")
    print(x[0, 0])

    print("\nPremier vecteur APRÈS :")
    print(y[0, 0])

    print("\nRMS avant :")

    rms_before = torch.sqrt(
        x[0, 0].pow(2).mean()
    )

    print(rms_before)

    print("\nRMS après :")

    rms_after = torch.sqrt(
        y[0, 0].pow(2).mean()
    )

    print(rms_after)


if __name__ == "__main__":
    main()