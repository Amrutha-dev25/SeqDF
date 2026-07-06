"""
Two set-prediction heads are provided here:

1. SetDetectionHead (original): 5 independent sigmoid outputs, trained with
   BCEWithLogitsLoss. Kept for rollback/comparison - this is what produced the
   val_exact_set_match plateau discussed in chat (pointwise loss vs listwise
   top-k eval metric mismatch).

2. SetClassificationHead (new): a single 10-way softmax over all C(5,3)=10
   possible 3-of-5 sets, trained with CrossEntropyLoss. This directly optimizes
   the same thing exact_set_match measures (get the whole combination right),
   instead of decomposing into 5 independent marginal decisions. See chat
   discussion for the full reasoning.

Method order (must match data/filename_parser.py's METHOD_TO_IDX, itself
derived from configs/data_config.yaml's `methods:` list) is:
    0=DF, 1=F2F, 2=FSh, 3=FS, 4=NT
This order is baked into ALL_SETS below - changing configs/data_config.yaml's
`methods:` order would silently break the mapping, so don't change one without
the other.
"""
import itertools

import torch
import torch.nn as nn

NUM_METHODS = 5
NUM_CLASSES = 10  # C(5, 3) - every video has exactly 3 of 5 methods, by construction

# Canonical enumeration of all 10 possible 3-of-5 sets, as index tuples into the
# 5-method order above. itertools.combinations(range(5), 3) with range(5) already
# in [DF, F2F, FSh, FS, NT] order guarantees this matches METHOD_TO_IDX exactly.
ALL_SETS: list[tuple] = list(itertools.combinations(range(NUM_METHODS), 3))
assert len(ALL_SETS) == NUM_CLASSES, f"Expected {NUM_CLASSES} sets, got {len(ALL_SETS)}"

_ALL_SETS_TENSOR_CACHE = {}


def _all_sets_tensor(device, dtype=torch.float32) -> torch.Tensor:
    """(NUM_CLASSES, NUM_METHODS) multi-hot reference tensor, cached per device."""
    key = (device, dtype)
    if key not in _ALL_SETS_TENSOR_CACHE:
        t = torch.zeros(NUM_CLASSES, NUM_METHODS, dtype=dtype, device=device)
        for i, combo in enumerate(ALL_SETS):
            for m in combo:
                t[i, m] = 1.0
        _ALL_SETS_TENSOR_CACHE[key] = t
    return _ALL_SETS_TENSOR_CACHE[key]


def multi_hot_to_class_idx(multi_hot: torch.Tensor) -> torch.Tensor:
    """
    (B, 5) multi-hot ground truth -> (B,) integer class index in [0, 9].
    Used to convert your existing set_vector labels (already multi-hot in the
    dataset/CSV) into the target format CrossEntropyLoss needs, with no changes
    required to data/video_dataset.py or the label CSVs themselves.
    """
    ref = _all_sets_tensor(multi_hot.device, multi_hot.dtype)  # (10, 5)
    matches = (multi_hot.unsqueeze(1) == ref.unsqueeze(0)).all(dim=-1)  # (B, 10)
    counts = matches.sum(dim=-1)
    if not torch.all(counts == 1):
        bad = (counts != 1).nonzero(as_tuple=True)[0]
        raise ValueError(
            f"multi_hot_to_class_idx: {len(bad)} row(s) are not a valid 3-of-5 "
            f"multi-hot vector (matched {counts[bad].tolist()} classes instead of "
            f"exactly 1). Check upstream labels - this should never happen given "
            f"the dataset's construction, so a mismatch here indicates a real bug, "
            f"not just noisy data."
        )
    return matches.float().argmax(dim=-1)


def class_idx_to_multi_hot(class_idx: torch.Tensor) -> torch.Tensor:
    """(B,) integer class index -> (B, 5) multi-hot. Inverse of the above."""
    ref = _all_sets_tensor(class_idx.device)  # (10, 5)
    return ref[class_idx]


class SetDetectionHead(nn.Module):
    """Original multi-label head - 5 independent sigmoid outputs. Kept for
    rollback/comparison against the classification head below."""

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


class SetClassificationHead(nn.Module):
    """
    New: direct 10-way classification over all possible 3-of-5 sets, trained
    with CrossEntropyLoss. Deliberately mirrors SetDetectionHead's first three
    layers (Linear -> LayerNorm -> GELU -> Dropout) exactly, so those layers'
    weights can be warm-started (transferred) from an existing
    SetDetectionHead checkpoint via load_state_dict(..., strict=False) - only
    the final Linear layer's shape differs (num_methods=5 -> num_classes=10),
    so only that one layer needs fresh initialization.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, input_dim) fused embedding
        Output: (B, num_classes) raw logits over the 10 possible 3-of-5 sets -
                use CrossEntropyLoss directly on these (it applies log_softmax
                internally), or softmax externally for probabilities.
        """
        return self.mlp(fused_embedding)


def predict_set_top_k(logits: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    For SetDetectionHead (multilabel) outputs only. Converts raw logits to a
    hard multi-hot set prediction by taking the top-k highest-scoring methods.
    Input: (B, num_methods) logits
    Output: (B, num_methods) multi-hot 0/1 tensor with exactly k ones per row
    """
    B, num_methods = logits.shape
    topk_indices = logits.topk(k, dim=-1).indices  # (B, k)
    multi_hot = torch.zeros_like(logits)
    multi_hot.scatter_(1, topk_indices, 1.0)
    return multi_hot


def predict_set_from_classification(class_logits: torch.Tensor) -> torch.Tensor:
    """
    For SetClassificationHead outputs only. argmax -> class index -> (B, 5)
    multi-hot, so downstream per-method metrics (accuracy/F1) stay directly
    comparable in definition/scale to your existing multilabel-head logs -
    only the loss/head changed, not what the metrics mean.
    """
    class_idx = class_logits.argmax(dim=-1)
    return class_idx_to_multi_hot(class_idx)