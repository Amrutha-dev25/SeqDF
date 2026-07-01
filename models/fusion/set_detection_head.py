"""
Multi-label set detection head (Stage A).

Predicts which 3 of the 5 methods were used ANYWHERE in the video, with no notion
of order or position. This is a 5-way multi-label classification problem (one
sigmoid output per method), not a 60-way ordered classification problem.

Why this is the better-posed problem (see chat discussion): a method's evidence
no longer needs to survive specifically at its temporal position (first/middle/
last) - it just needs to survive somewhere, in any of the 3 streams (RGB, SRM,
DCT), at any frame. This sidesteps the steep degradation curve that made m1
(first-step) prediction so hard in the original ordered-sequence formulation.
"""
import torch
import torch.nn as nn


class SetDetectionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_methods: int = 5, dropout: float = 0.3):
        super().__init__()
        self.num_methods = num_methods
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_methods),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, input_dim) fused embedding from the 3-stream backbone
        Output: (B, num_methods) raw logits - apply sigmoid externally for
                probabilities, or use BCEWithLogitsLoss directly on these logits
                during training (numerically more stable than sigmoid+BCE).
        """
        return self.mlp(fused_embedding)


def predict_set_top_k(logits: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Converts raw logits to a hard multi-hot set prediction by taking the top-k
    highest-scoring methods. We use top-k (not a 0.5 sigmoid threshold) because
    we know by construction that every video has EXACTLY 3 manipulations - so
    forcing exactly 3 positive predictions is a safe, label-distribution-aware
    inductive bias rather than an arbitrary threshold choice.

    Input: (B, num_methods) logits
    Output: (B, num_methods) multi-hot 0/1 tensor with exactly k ones per row
    """
    B, num_methods = logits.shape
    topk_indices = logits.topk(k, dim=-1).indices  # (B, k)
    multi_hot = torch.zeros_like(logits)
    multi_hot.scatter_(1, topk_indices, 1.0)
    return multi_hot
