"""
Metrics for both stages.

Stage A (set detection) metrics:
  - per_method_accuracy: averaged binary accuracy across all 5 methods
  - per_method_f1: averaged F1 across all 5 methods (more informative than
    accuracy alone, since "is method X present" labels are roughly 60/40
    positive/negative per method - not wildly imbalanced, but F1 still gives
    a fuller picture than accuracy alone)
  - exact_set_match_accuracy: all 3 predicted methods exactly match the true
    set (order-agnostic) - this is the headline Stage A metric

Stage B (sequence ordering) metrics:
  - ordering_top1_accuracy / ordering_top2_accuracy: standard classification
    accuracy over the 6 possible orderings of a known set
"""
import torch
import numpy as np

from models.fusion.set_detection_head import predict_set_top_k, predict_set_from_classification


def per_method_accuracy(pred_multi_hot: torch.Tensor, true_multi_hot: torch.Tensor) -> float:
    """Binary accuracy per method, averaged across all 5 methods and the batch."""
    correct = (pred_multi_hot == true_multi_hot).float()
    return correct.mean().item()


def per_method_f1(pred_multi_hot: torch.Tensor, true_multi_hot: torch.Tensor) -> float:
    """F1 score per method, averaged across all 5 methods."""
    tp = ((pred_multi_hot == 1) & (true_multi_hot == 1)).float().sum(dim=0)  # (5,)
    fp = ((pred_multi_hot == 1) & (true_multi_hot == 0)).float().sum(dim=0)
    fn = ((pred_multi_hot == 0) & (true_multi_hot == 1)).float().sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def exact_set_match_accuracy(pred_multi_hot: torch.Tensor, true_multi_hot: torch.Tensor) -> float:
    """All 5 binary decisions must match simultaneously (equivalently: the
    predicted 3-method set exactly equals the true 3-method set)."""
    row_match = (pred_multi_hot == true_multi_hot).all(dim=-1).float()
    return row_match.mean().item()


def compute_set_detection_metrics(set_logits: torch.Tensor, set_targets: torch.Tensor, top_k: int = 3) -> dict:
    """
    Args:
        set_logits: (B, 5) raw logits
        set_targets: (B, 5) multi-hot ground truth
        top_k: number of methods to predict as "present" (3, since every video
               has exactly 3 manipulations by construction)
    """
    pred_multi_hot = predict_set_top_k(set_logits, k=top_k)

    return {
        "per_method_acc": per_method_accuracy(pred_multi_hot, set_targets),
        "per_method_f1": per_method_f1(pred_multi_hot, set_targets),
        "exact_set_match_acc": exact_set_match_accuracy(pred_multi_hot, set_targets),
    }


def ordering_top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def ordering_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 2) -> float:
    topk_preds = logits.topk(k, dim=-1).indices
    targets_expanded = targets.unsqueeze(-1).expand_as(topk_preds)
    correct = (topk_preds == targets_expanded).any(dim=-1).float()
    return correct.mean().item()


def compute_set_classification_metrics(class_logits: torch.Tensor, set_targets_multihot: torch.Tensor) -> dict:
    """
    Classification-head (10-way softmax) variant of compute_set_detection_metrics.
    Converts the predicted class back to a multi-hot vector so per_method_acc/
    per_method_f1 stay directly comparable in definition and scale to your
    existing multilabel-head logs - only the loss/head changed underneath,
    the metric definitions themselves did not. exact_set_match_acc under this
    head is equivalent to (argmax(class_logits) == true_class_idx).mean(),
    just computed via the same multi-hot comparison for consistency.

    Args:
        class_logits: (B, 10) raw logits
        set_targets_multihot: (B, 5) multi-hot ground truth (same format as
                               already used everywhere else in this project -
                               no changes needed to data/video_dataset.py)
    """
    pred_multi_hot = predict_set_from_classification(class_logits)

    return {
        "per_method_acc": per_method_accuracy(pred_multi_hot, set_targets_multihot),
        "per_method_f1": per_method_f1(pred_multi_hot, set_targets_multihot),
        "exact_set_match_acc": exact_set_match_accuracy(pred_multi_hot, set_targets_multihot),
    }


def compute_stage_b_metrics(ordering_logits: torch.Tensor, ordering_targets: torch.Tensor) -> dict:
    return {
        "ordering_top1_acc": ordering_top1_accuracy(ordering_logits, ordering_targets),
        "ordering_top2_acc": ordering_top_k_accuracy(ordering_logits, ordering_targets, k=2),
    }


def average_metrics(metrics_list: list) -> dict:
    """Averages a list of per-batch metric dicts into one summary dict."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}