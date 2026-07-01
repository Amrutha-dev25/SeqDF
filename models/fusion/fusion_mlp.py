"""
Fusion MLP: concatenates the RGB/SRM/DCT embeddings and projects down to a shared
fused representation that the autoregressive decoder then consumes.
"""
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int = 1792, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, rgb_emb: torch.Tensor, srm_emb: torch.Tensor, dct_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs: each (B, embedding_dim_i)
        Output: (B, output_dim) fused embedding
        """
        fused = torch.cat([rgb_emb, srm_emb, dct_emb], dim=-1)
        return self.mlp(fused)
