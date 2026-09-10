import torch
import torch.nn as nn

from tokenizer_v0 import VOCAB_SIZE
from rmsnorm_v0 import RMSNorm

from time_embedding_v0 import (
    TimeEmbedding,
)

from transformer_block_v1 import (
    TransformerBlockV1,
)


class DiffuThinkBackbone(nn.Module):

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_heads: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()

        self.d_model = d_model

        # =================================
        # Token Embedding
        # =================================

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # =================================
        # Time Embedding
        # =================================

        self.time_embedding = TimeEmbedding(
            d_model=d_model,
        )

        # =================================
        # Transformer Blocks
        # =================================

        self.blocks = nn.ModuleList(
            [
                TransformerBlockV1(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )

        # =================================
        # Final Norm
        # =================================

        self.final_norm = RMSNorm(
            d_model=d_model,
        )

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ):

        # noisy_tokens :
        # [B,T]

        # timesteps :
        # [B]

        # =================================
        # Tokens → vectors
        # =================================

        x = self.token_embedding(
            noisy_tokens
        )

        # [B,T,C]

        # =================================
        # t → vector
        # =================================

        t_emb = self.time_embedding(
            timesteps
        )

        # [B,C]

        all_attention_weights = []

        # =================================
        # Transformer stack
        # =================================

        for block in self.blocks:

            (
                x,
                attention_weights,
            ) = block(
                x,
                t_emb,
            )

            all_attention_weights.append(
                attention_weights
            )

        # =================================
        # Final normalization
        # =================================

        x = self.final_norm(x)

        return (
            x,
            all_attention_weights,
        )