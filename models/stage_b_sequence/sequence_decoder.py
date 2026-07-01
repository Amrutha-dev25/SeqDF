"""
Stage B: sequence-within-known-set decoder.

Given a KNOWN (or Stage-A-predicted) unordered 3-method set, this small model
predicts which of the 6 possible orderings (3! = 6) was actually used. This is
a much smaller, more tractable problem than the original 60-way ordered
sequence classification, because conditioning on the set already eliminates
4 of the 5 methods from consideration at each position.

Deliberately small and trained SEPARATELY from Stage A:
  - Stage A's backbones + fusion are frozen and reused as a fixed feature
    extractor (no need to re-train the expensive video backbones)
  - Input = [frozen fused embedding from Stage A] + [explicit one-hot encoding
    of the known/predicted 3-method set] concatenated together
  - Output = 6-way classification over the permutations of that specific set

Run this only after Stage A (set detection) is working well - see
training/stageB_train_sequence.py and the README for the full reasoning on why
this two-stage approach is more tractable than joint sequence prediction.

NOTE: this module is provided as a complete, runnable stub. Expected accuracy
here is harder to predict in advance than Stage A's, since it depends heavily on
how much positional information survives in the fused embedding even after
collapsing to set-level supervision in Stage A. Realistically expect somewhere
in the 30-50% range on the 6-way task (vs ~16.7% random chance), but treat this
as a rough prior to be checked empirically rather than a guarantee - report
actual validation numbers once you've run it, since this hasn't been
characterized against literature the way the Stage A ranges have.
"""
import torch
import torch.nn as nn

from data.filename_parser import METHODS, METHOD_TO_IDX


class StageBSequenceDecoder(nn.Module):
    def __init__(self, fused_embedding_dim: int = 256, hidden_dim: int = 128,
                 num_layers: int = 1, num_heads: int = 2, num_orderings: int = 6,
                 num_methods: int = 5):
        super().__init__()
        self.num_orderings = num_orderings
        self.num_methods = num_methods

        # input = frozen fused embedding + one-hot set encoding (5-dim, exactly
        # 3 ones for the known/predicted set) concatenated together
        input_dim = fused_embedding_dim + num_methods

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_orderings)

    def forward(self, fused_embedding: torch.Tensor, set_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_embedding: (B, fused_embedding_dim) - FROZEN, from Stage A
            set_one_hot: (B, num_methods) multi-hot vector, exactly 3 ones,
                         representing the known/predicted method set
        Returns:
            (B, num_orderings) logits over the 6 possible orderings of THIS set.
            Caller is responsible for mapping the predicted ordering index back
            to actual (m1,m2,m3) method names using the set - see
            ordering_index_to_methods() below, since "ordering 3" means something
            different for set {DF,F2F,FS} than for set {NT,FSh,FS}.
        """
        x = torch.cat([fused_embedding, set_one_hot], dim=-1)  # (B, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim) - single token sequence
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])  # (B, num_orderings)


def set_to_one_hot(set_sorted_str: str) -> torch.Tensor:
    """Converts a 'DF-F2F-FS' style canonical set string into a 5-dim multi-hot vector."""
    methods_in_set = set_sorted_str.split("-")
    vec = torch.zeros(len(METHODS))
    for m in methods_in_set:
        vec[METHOD_TO_IDX[m]] = 1.0
    return vec


def ordering_index_to_methods(set_sorted_str: str, ordering_idx: int) -> tuple:
    """
    Maps a predicted ordering index (0-5) back to actual (m1, m2, m3) method
    names, given the known/predicted set. Must use the SAME deterministic
    permutation ordering as filename_parser.parse_filename's
    `ordering_within_set` computation (itertools.permutations of the sorted
    set tuple), or predictions will be silently mislabeled.
    """
    from itertools import permutations
    methods_in_set = tuple(set_sorted_str.split("-"))  # already sorted - matches parser convention
    all_orderings = list(permutations(methods_in_set))
    if ordering_idx >= len(all_orderings):
        raise ValueError(f"ordering_idx {ordering_idx} out of range for set {set_sorted_str} "
                          f"(only {len(all_orderings)} orderings exist)")
    return all_orderings[ordering_idx]
