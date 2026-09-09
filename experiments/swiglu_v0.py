import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Branche qui contrôle la "porte"
        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        # Branche qui contient les valeurs
        self.up_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )

        # Retour vers d_model
        self.down_proj = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        """
        x :
            [B,T,C]

        output :
            [B,T,C]
        """

        gate = self.gate_proj(x)

        up = self.up_proj(x)

        # SiLU(gate) * up
        hidden = (
            F.silu(gate)
            * up
        )

        output = self.down_proj(
            hidden
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
    C = 64

    HIDDEN_DIM = 256

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    )

    mlp = SwiGLU(
        d_model=C,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    y = mlp(x)

    print("Input :")
    print(x.shape)

    print("\nGate :")
    print(
        mlp.gate_proj(x).shape
    )

    print("\nUp :")
    print(
        mlp.up_proj(x).shape
    )

    print("\nOutput :")
    print(y.shape)


if __name__ == "__main__":
    main()