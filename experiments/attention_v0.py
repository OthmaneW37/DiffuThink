import math

import torch
import torch.nn as nn


class SingleHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        head_dim: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.head_dim = head_dim

        self.q_proj = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x:
            [B, T, C]

        Retour :
            output:
                [B, T, D]

            attention_weights:
                [B, T, T]
        """

        # -----------------------------
        # 1. Construire Q, K et V
        # -----------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shapes :
        #
        # q : [B, T, D]
        # k : [B, T, D]
        # v : [B, T, D]

        # -----------------------------
        # 2. Comparer Q et K
        # -----------------------------

        scores = (
            q
            @ k.transpose(-2, -1)
        )

        # Shape :
        #
        # [B, T, T]

        # -----------------------------
        # 3. Scaling
        # -----------------------------

        scores = (
            scores
            / math.sqrt(self.head_dim)
        )

        # -----------------------------
        # 4. Transformer en probabilités
        # -----------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # -----------------------------
        # 5. Combiner les Values
        # -----------------------------

        output = (
            attention_weights
            @ v
        )

        # Shape :
        #
        # [B, T, D]

        return (
            output,
            attention_weights,
            q,
            k,
            v,
        )


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device :", device)

    B = 2
    T = 4
    C = 8

    HEAD_DIM = 8

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    )

    print("\nInput X shape :")
    print(x.shape)

    attention = SingleHeadSelfAttention(
        d_model=C,
        head_dim=HEAD_DIM,
    ).to(device)

    (
        output,
        weights,
        q,
        k,
        v,
    ) = attention(x)

    print("\nQ shape :")
    print(q.shape)

    print("\nK shape :")
    print(k.shape)

    print("\nV shape :")
    print(v.shape)

    print("\nAttention weights du premier token :")
    print(weights[0, 0])

    print("\nSomme des poids :")
    print(weights[0, 0].sum())

    print("\nMatrice d'attention du premier exemple :")
    print(weights[0])

    print("\nChaque ligne doit sommer à 1 :")
    print(weights[0].sum(dim=-1))


if __name__ == "__main__":
    main()
    