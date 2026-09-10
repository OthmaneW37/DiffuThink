import torch
import torch.nn as nn

from rmsnorm_v0 import RMSNorm
from swiglu_v0 import SwiGLU

from multihead_attention_rope_v0 import (
    MultiHeadSelfAttentionRoPE,
)


class TransformerBlockV1(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.time_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.attention_norm = RMSNorm(
            d_model,
        )

        self.attention = (
            MultiHeadSelfAttentionRoPE(
                d_model=d_model,
                n_heads=n_heads,
            )
        )

        self.mlp_norm = RMSNorm(
            d_model,
        )

        self.mlp = SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ):

        """
        x :
            [B,T,C]

        time_embedding :
            [B,C]
        """

        # =================================
        # Injecter le timestep
        # =================================

        time_information = (
            self.time_projection(
                time_embedding
            )
        )

        # [B,C]
        #
        # devient :
        #
        # [B,1,C]

        time_information = (
            time_information.unsqueeze(1)
        )

        # Broadcasting :
        #
        # [B,T,C]
        # +
        # [B,1,C]
        #
        # =
        # [B,T,C]

        x = (
            x
            + time_information
        )

        # =================================
        # Attention
        # =================================

        normalized_x = (
            self.attention_norm(x)
        )

        (
            attention_output,
            attention_weights,
        ) = self.attention(
            normalized_x
        )

        x = (
            x
            + attention_output
        )

        # =================================
        # MLP
        # =================================

        normalized_x = (
            self.mlp_norm(x)
        )

        mlp_output = self.mlp(
            normalized_x
        )

        x = (
            x
            + mlp_output
        )

        return (
            x,
            attention_weights,
        )