import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model doit être divisible par n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = (
            d_model // n_heads
        )

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x:
            [B,T,C]
        """

        B, T, C = x.shape

        # -------------------------
        # Q K V
        # -------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Actuellement :
        #
        # [B,T,C]

        # -------------------------
        # Séparer les heads
        # -------------------------

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        )

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        )

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        )

        # Maintenant :
        #
        # [B,T,H,D]

        # On préfère :
        #
        # [B,H,T,D]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -------------------------
        # Attention
        # -------------------------

        scores = (
            q
            @ k.transpose(-2, -1)
        )

        scores = (
            scores
            / math.sqrt(self.head_dim)
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        output = (
            attention_weights
            @ v
        )

        # output :
        #
        # [B,H,T,D]

        # -------------------------
        # Recombiner les heads
        # -------------------------

        output = output.transpose(
            1,
            2,
        )

        # [B,T,H,D]

        output = output.contiguous()

        output = output.view(
            B,
            T,
            C,
        )

        # [B,T,C]

        # -------------------------
        # Projection finale
        # -------------------------

        output = self.out_proj(
            output
        )

        return (
            output,
            attention_weights,
        )


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    B = 4
    T = 64
    C = 64

    N_HEADS = 4

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    )

    attention = MultiHeadSelfAttention(
        d_model=C,
        n_heads=N_HEADS,
    ).to(device)

    output, weights = attention(x)

    print("Device :")
    print(device)

    print("\nInput :")
    print(x.shape)

    print("\nOutput :")
    print(output.shape)

    print("\nAttention weights :")
    print(weights.shape)

    print("\nNombre de heads :")
    print(N_HEADS)

    print("\nHead dimension :")
    print(
        attention.head_dim
    )

    print(
        "\nSomme attention "
        "head 0, token 0 :"
    )

    print(
        weights[
            0,
            0,
            0,
        ].sum()
    )


if __name__ == "__main__":
    main()