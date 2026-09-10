import math

import torch
import torch.nn as nn

from rope_v0 import apply_rope


class MultiHeadSelfAttentionRoPE(nn.Module):

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

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim doit être pair pour RoPE."
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

        B, T, C = x.shape

        # =================================
        # Q K V
        # =================================

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B,T,C]

        # =================================
        # Séparer les heads
        # =================================

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Maintenant :
        #
        # [B,H,T,D]

        # =================================
        # RoPE
        # =================================

        q = apply_rope(q)
        k = apply_rope(k)

        # V n'est PAS modifié par RoPE.

        # =================================
        # Attention
        # =================================

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

        # [B,H,T,D]

        # =================================
        # Recombiner les heads
        # =================================

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

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

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    )

    attention = MultiHeadSelfAttentionRoPE(
        d_model=C,
        n_heads=4,
    ).to(device)

    output, weights = attention(x)

    print("Input :")
    print(x.shape)

    print("\nOutput :")
    print(output.shape)

    print("\nAttention weights :")
    print(weights.shape)


if __name__ == "__main__":
    main()