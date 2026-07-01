"""
Loss functions for both stages.

Stage A: multi-label binary cross-entropy (one independent binary decision per
method: "did this method appear anywhere in the video"). BCEWithLogitsLoss is
used directly on raw logits (not sigmoid + BCELoss) for numerical stability.

Stage B: standard cross-entropy over the 6 possible orderings of a known set.
"""
import torch
import torch.nn as nn


def set_detection_loss(set_logits: torch.Tensor, set_targets: torch.Tensor,
                        label_smoothing: float = 0.0, pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        set_logits: (B, 5) raw logits from SetDetectionHead
        set_targets: (B, 5) multi-hot 0/1 ground truth, exactly 3 ones per row
        label_smoothing: softens hard 0/1 targets toward 0.5 by this amount,
                          reduces overconfidence given the small (93-source)
                          dataset
        pos_weight: optional (5,) tensor to rebalance per-method positive/negative
                    weighting (passed through to BCEWithLogitsLoss) - see
                    training/stageA_train_set_detection.py for how this gets
                    computed from the training split's actual class balance
    """
    if label_smoothing > 0:
        # smooth targets: 1 -> 1-eps/2, 0 -> eps/2
        set_targets = set_targets * (1 - label_smoothing) + 0.5 * label_smoothing

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(set_logits, set_targets)


def sequence_ordering_loss(ordering_logits: torch.Tensor, ordering_targets: torch.Tensor,
                            label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Args:
        ordering_logits: (B, 6) logits over the 6 orderings of a known set
        ordering_targets: (B,) integer class index 0-5
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return loss_fn(ordering_logits, ordering_targets)
