"""
Loss functions for both stages.

Stage A: two options, controlled by configs/model_config.yaml set_head.type:
  - "multilabel_mlp": independent binary cross-entropy per method (original).
    BCEWithLogitsLoss is used directly on raw logits (not sigmoid + BCE) for
    numerical stability.
  - "classification_10way": direct 10-way cross-entropy over all C(5,3)=10
    possible 3-of-5 sets (see chat discussion - this directly optimizes the
    same objective val_exact_set_match measures, unlike the multilabel option).

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


def set_classification_loss(class_logits: torch.Tensor, class_targets: torch.Tensor,
                             label_smoothing: float = 0.0, class_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        class_logits: (B, 10) raw logits from SetClassificationHead
        class_targets: (B,) integer class index 0-9 - convert your existing
                       multi-hot set_vector labels with
                       models.fusion.set_detection_head.multi_hot_to_class_idx()
        label_smoothing: standard CrossEntropyLoss label smoothing
        class_weight: optional (10,) tensor to rebalance class frequency
                      (passed through to CrossEntropyLoss's weight= arg) - see
                      training/stageA_train_set_detection.py for how this gets
                      computed from the training split's actual class balance
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weight)
    return loss_fn(class_logits, class_targets)


def sequence_ordering_loss(ordering_logits: torch.Tensor, ordering_targets: torch.Tensor,
                            label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Args:
        ordering_logits: (B, 6) logits over the 6 orderings of a known set
        ordering_targets: (B,) integer class index 0-5
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return loss_fn(ordering_logits, ordering_targets)