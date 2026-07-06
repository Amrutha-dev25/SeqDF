"""
Stage A: trains the full SeqDFModel (3-stream backbone + fusion + set prediction
head) END-TO-END, predicting WHICH 3 of 5 methods were used anywhere in the
video (order-agnostic).

Two head/loss modes, controlled by configs/model_config.yaml set_head.type:
  - "multilabel_mlp": 5 independent sigmoid outputs, BCEWithLogitsLoss.
  - "classification_10way": direct 10-way softmax over all C(5,3)=10 possible
    sets, CrossEntropyLoss. Added to fix a loss/metric mismatch discovered via
    val_exact_set_match plateauing while val_per_method_f1 kept improving -
    see chat discussion. This directly optimizes what exact_set_match measures.

Resume support (SAME architecture only):
    python training/stageA_train_set_detection.py --resume
Resumes from the highest-epoch stageA_set_detection_epoch*.pt checkpoint,
restoring model, optimizer, scheduler, epoch counter, and best-metric-so-far
exactly. Requires the checkpoint's head architecture to match the CURRENT
set_head.type in model_config.yaml - if you just switched types, use
--warm-start-backbone instead (see below).

Warm start across an architecture change (e.g. switching set_head.type):
    python training/stageA_train_set_detection.py --warm-start-backbone C:/path/to/checkpoint.pt
Loads whatever DOES match by name+shape (rgb/srm/dct backbones, fusion MLP,
and - since SetClassificationHead deliberately mirrors SetDetectionHead's
first three layers - even the head's first Linear+LayerNorm) from the given
checkpoint via load_state_dict(..., strict=False), and randomly initializes
only what doesn't match (the final classification layer). Always starts a
FRESH optimizer and epoch counter at 0 - momentum and LR-schedule position
cannot be transferred across a loss-function change, this is expected and
not a bug. Prints exactly which keys were skipped so you can visually confirm
only the expected final-layer mismatch occurred, not something unintended.
"""
import argparse
import os
import sys

import torch
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqdf_model import SeqDFModel
from models.fusion.set_detection_head import multi_hot_to_class_idx, NUM_CLASSES
from data.video_dataset import build_dataloaders
from training.losses import set_detection_loss, set_classification_loss
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint, find_best_metric_so_far
from utils.logger import TrainingLogger
from utils.metrics import compute_set_detection_metrics, compute_set_classification_metrics, average_metrics

STAGE_TAG = "stageA_set_detection"


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def compute_pos_weight(train_loader, num_methods: int, device) -> torch.Tensor:
    """
    multilabel_mlp only. Computes BCEWithLogitsLoss pos_weight from the actual
    training split's per-method positive rate, so rare method/position
    combinations don't get systematically under-predicted.
    """
    print("Computing pos_weight for BCE loss from training split class balance...")
    pos_counts = torch.zeros(num_methods)
    total = 0
    for batch in train_loader:
        set_targets = batch["labels"]["set_vector"]  # (B, 5)
        pos_counts += set_targets.sum(dim=0)
        total += set_targets.shape[0]

    pos_rate = pos_counts / total
    neg_rate = 1 - pos_rate
    pos_weight = (neg_rate / (pos_rate + 1e-8)).clamp(0.2, 5.0)
    print(f"  Per-method positive rate: {pos_rate.tolist()}")
    print(f"  Computed pos_weight: {pos_weight.tolist()}")
    return pos_weight.to(device)


def compute_class_weight(train_loader, num_classes: int, device) -> torch.Tensor:
    """
    classification_10way only. Computes CrossEntropyLoss class weight from the
    actual training split's per-class (per-set) frequency, analogous to
    compute_pos_weight above but for the 10-way classification target.
    """
    print("Computing class_weight for CrossEntropy loss from training split class balance...")
    class_counts = torch.zeros(num_classes)
    total = 0
    for batch in train_loader:
        set_targets = batch["labels"]["set_vector"]  # (B, 5) multi-hot
        class_idx = multi_hot_to_class_idx(set_targets)
        for c in class_idx.tolist():
            class_counts[c] += 1
        total += set_targets.shape[0]

    class_rate = class_counts / total
    print(f"  Per-class rate (10 sets): {class_rate.tolist()}")
    # standard inverse-frequency weighting, normalized so weights average to 1
    inv_freq = 1.0 / (class_rate + 1e-8)
    class_weight = inv_freq / inv_freq.mean()
    class_weight = class_weight.clamp(0.2, 5.0)
    print(f"  Computed class_weight: {class_weight.tolist()}")
    return class_weight.to(device)


def run_stage_a(cfgs: dict, resume: bool = False, known_best: float = None, warm_start_backbone: str = None):
    set_seed(cfgs["data_config"]["split"]["random_seed"])
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    amp_enabled = cfgs["train_config"]["amp_enabled"] and device.type == "cuda"

    loaders = build_dataloaders(cfgs["paths"], cfgs["data_config"])
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = SeqDFModel(cfgs["model_config"]).to(device)
    set_head_type = model.set_head_type
    print(f"[Model] set_head_type = {set_head_type}")

    stage_cfg = cfgs["train_config"]["stage_a_set_detection"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["epochs"])
    scaler = GradScaler(enabled=amp_enabled)

    checkpoint_dir = cfgs["paths"]["checkpoint_dir"]
    logger = TrainingLogger(cfgs["paths"]["log_dir"], run_name=f"stage_a_set_detection_{set_head_type}")
    label_smoothing = cfgs["model_config"]["label_smoothing"]["set_detection"]
    num_methods = cfgs["data_config"]["num_methods"]
    top_k = cfgs["model_config"]["set_head"]["top_k"]

    start_epoch = 0
    best_exact_match = 0.0
    patience_counter = 0

    if warm_start_backbone:
        if not os.path.exists(warm_start_backbone):
            print(f"[ERROR] --warm-start-backbone path does not exist: {warm_start_backbone}")
            sys.exit(1)
        print(f"Warm-starting from {warm_start_backbone} (architecture-change mode - "
              f"only matching-shape layers transfer, optimizer/epoch/scheduler start fresh)")
        raw = torch.load(warm_start_backbone, map_location=device, weights_only=False)
        # Handle both a full checkpoint dict (from save_checkpoint) and a raw
        # state_dict (e.g. stageA_best.pt) transparently.
        src_state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        result = model.load_state_dict(src_state, strict=False)
        print(f"  Transferred layers loaded OK.")
        print(f"  Skipped (not found in checkpoint - freshly initialized): {result.missing_keys}")
        print(f"  Ignored (in checkpoint but not used by current architecture): {result.unexpected_keys}")
        start_epoch = 0
        best_exact_match = known_best if known_best is not None else 0.0
        if known_best is not None:
            print(f"  Using manually supplied --known-best={known_best:.4f} as the floor "
                  f"for 'new best' comparisons under the new head/loss.")
    elif resume:
        try:
            latest_path = find_latest_checkpoint(checkpoint_dir, tag=STAGE_TAG)
            ckpt = load_checkpoint(model, latest_path, optimizer=optimizer, scheduler=scheduler,
                                    map_location=device)
            start_epoch = ckpt["epoch"] + 1
            best_exact_match = find_best_metric_so_far(
                checkpoint_dir, tag=STAGE_TAG, metric_key="exact_set_match_acc", map_location=device
            )
            print(f"Resuming Stage A from epoch {start_epoch} "
                  f"(best_exact_set_match_acc so far = {best_exact_match:.4f})")
        except FileNotFoundError:
            best_path = os.path.join(checkpoint_dir, "stageA_best.pt")
            if os.path.exists(best_path):
                print(f"--resume was passed: no per-epoch checkpoint found for tag "
                      f"'{STAGE_TAG}', but {best_path} exists. Loading those weights "
                      f"as a WARM START (optimizer/scheduler state and true epoch "
                      f"number CANNOT be recovered from this file - only weights).")
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
                start_epoch = 0
                if known_best is not None:
                    best_exact_match = known_best
                    print(f"  Using manually supplied --known-best={known_best:.4f} as the "
                          f"floor for 'new best' comparisons, so early re-warming epochs "
                          f"won't overwrite these weights with a worse checkpoint.")
                else:
                    print("  WARNING: no --known-best supplied. best_exact_match starts at "
                          "0.0, so the FIRST validation pass this run will be saved as "
                          "'new best' and OVERWRITE stageA_best.pt, even if it scores worse "
                          "than what's currently saved.")
            else:
                print(f"--resume was passed but no checkpoint found for tag '{STAGE_TAG}' "
                      f"and no {best_path} exists either - starting from scratch instead.")

    pos_weight = None
    class_weight = None
    if set_head_type == "multilabel_mlp" and stage_cfg.get("pos_weight_balancing", False):
        pos_weight = compute_pos_weight(train_loader, num_methods, device)
    elif set_head_type == "classification_10way" and stage_cfg.get("pos_weight_balancing", False):
        class_weight = compute_class_weight(train_loader, NUM_CLASSES, device)

    early_stop_cfg = cfgs["train_config"]["early_stopping"]

    for epoch in range(start_epoch, stage_cfg["epochs"]):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"StageA epoch {epoch}"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            srm = batch["srm"].to(device, non_blocking=True)
            dct = batch["dct"].to(device, non_blocking=True)
            set_targets = batch["labels"]["set_vector"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=amp_enabled):
                outputs = model(rgb, srm, dct)
                if set_head_type == "classification_10way":
                    class_targets = multi_hot_to_class_idx(set_targets)
                    loss = set_classification_loss(
                        outputs["set_logits"], class_targets,
                        label_smoothing=label_smoothing, class_weight=class_weight,
                    )
                else:
                    loss = set_detection_loss(
                        outputs["set_logits"], set_targets,
                        label_smoothing=label_smoothing, pos_weight=pos_weight,
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        scheduler.step()

        # --- validation ---
        model.eval()
        val_metrics_list = []
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                srm = batch["srm"].to(device, non_blocking=True)
                dct = batch["dct"].to(device, non_blocking=True)
                set_targets = batch["labels"]["set_vector"].to(device, non_blocking=True)

                with autocast(enabled=amp_enabled):
                    outputs = model(rgb, srm, dct)

                if set_head_type == "classification_10way":
                    val_metrics_list.append(compute_set_classification_metrics(outputs["set_logits"], set_targets))
                else:
                    val_metrics_list.append(compute_set_detection_metrics(outputs["set_logits"], set_targets, top_k=top_k))

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_metrics = average_metrics(val_metrics_list)

        logger.log_text(
            f"[StageA][Epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_per_method_acc={avg_val_metrics['per_method_acc']:.4f} "
            f"val_per_method_f1={avg_val_metrics['per_method_f1']:.4f} "
            f"val_exact_set_match={avg_val_metrics['exact_set_match_acc']:.4f}"
        )
        logger.log_scalar("stageA/train_loss", avg_train_loss, epoch)
        logger.log_metrics(avg_val_metrics, epoch, prefix="stageA/val_")

        save_checkpoint(model, optimizer, epoch, avg_val_metrics,
                         checkpoint_dir, tag=STAGE_TAG, scheduler=scheduler)

        if avg_val_metrics["exact_set_match_acc"] > best_exact_match:
            best_exact_match = avg_val_metrics["exact_set_match_acc"]
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "stageA_best.pt")
            tmp_best_path = best_path + ".tmp"
            try:
                torch.save(model.state_dict(), tmp_best_path)
                os.replace(tmp_best_path, best_path)
                logger.log_text(f"  New best val_exact_set_match={best_exact_match:.4f} -> saved to {best_path}")
            except OSError as e:
                if os.path.exists(tmp_best_path):
                    try:
                        os.remove(tmp_best_path)
                    except OSError:
                        pass
                logger.log_text(f"  [WARNING] New best val_exact_set_match={best_exact_match:.4f} but "
                                 f"FAILED to save to {best_path}: {e}. Likely low disk space. "
                                 f"Training continues, but stageA_best.pt was NOT updated this epoch.")
        else:
            patience_counter += 1
            if early_stop_cfg.get("enabled", True) and patience_counter >= early_stop_cfg["patience"]:
                logger.log_text(f"Early stopping at epoch {epoch}")
                break

    logger.close()
    print(f"\nStage A complete. Best val_exact_set_match_acc={best_exact_match:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the highest-epoch stageA_set_detection_epoch*.pt "
                             "checkpoint - requires SAME set_head.type as when it was saved.")
    parser.add_argument("--known-best", type=float, default=None,
                         help="Manually supply a best val_exact_set_match_acc floor - used by "
                              "both the --resume warm-start-from-stageA_best.pt fallback and "
                              "--warm-start-backbone, to avoid overwriting good weights with a "
                              "worse checkpoint during early re-warming epochs.")
    parser.add_argument("--warm-start-backbone", type=str, default=None,
                         help="Path to a checkpoint (.pt) to transfer matching-shape layers from "
                              "(backbones + fusion, and head layers that still match shape) when "
                              "switching set_head.type. Optimizer/epoch/scheduler always start "
                              "fresh. Takes priority over --resume if both are passed.")
    args = parser.parse_args()

    cfgs = load_configs()
    run_stage_a(cfgs, resume=args.resume, known_best=args.known_best,
                warm_start_backbone=args.warm_start_backbone)