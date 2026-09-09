import torch
import torch.nn as nn

from rmsnorm_v0 import RMSNorm
from swiglu_v0 import SwiGLU

from multihead_attention_v0 import (
    MultiHeadSelfAttention,
)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Normalisation avant attention
        self.attention_norm = RMSNorm(
            d_model=d_model,
        )

        # Attention
        self.attention = (
            MultiHeadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
            )
        )

        # Normalisation avant MLP
        self.mlp_norm = RMSNorm(
            d_model=d_model,
        )

        # Feed-forward
        self.mlp = SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        x :
            [B,T,C]

        output :
            [B,T,C]
        """

        # =================================
        # PARTIE ATTENTION
        # =================================

        # 1. Normaliser
        normalized_x = (
            self.attention_norm(x)
        )

        # 2. Attention
        (
            attention_output,
            attention_weights,
        ) = self.attention(
            normalized_x
        )

        # 3. Residual connection
        x = (
            x
            + attention_output
        )

        # =================================
        # PARTIE MLP
        # =================================

        # 4. Normaliser
        normalized_x = (
            self.mlp_norm(x)
        )

        # 5. SwiGLU
        mlp_output = self.mlp(
            normalized_x
        )

        # 6. Residual connection
        x = (
            x
            + mlp_output
        )

        return (
            x,
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

    block = TransformerBlock(
        d_model=C,
        n_heads=4,
        hidden_dim=256,
    ).to(device)

    x = torch.randn(
        B,
        T,
        C,
        device=device,
    )

    output, attention_weights = (
        block(x)
    )

    print("Input :")
    print(x.shape)

    print("\nOutput :")
    print(output.shape)

    print("\nAttention weights :")
    print(
        attention_weights.shape
    )

    print("\nNombre de paramètres :")

    parameter_count = sum(
        parameter.numel()
        for parameter in block.parameters()
    )

    print(parameter_count)


if __name__ == "__main__":
    main()