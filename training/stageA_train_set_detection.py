"""
Stage A: trains the full SeqDFModel (3-stream backbone + fusion + multi-label
set detection head) END-TO-END from the start, predicting WHICH 3 of 5 methods
were used anywhere in the video (order-agnostic).

This replaces the old per-position staged training (stage1_rgb -> stage2_srm ->
stage3_dct -> stage4_joint) from the sequence-prediction version of this
pipeline. The staging rationale there was specific to ordered prediction
(easiest target first, freeze, build up); set detection doesn't have the same
positional difficulty gradient (every stream can usefully learn "is method X
present" from epoch 1), so joint training from the start is both simpler and
appropriate here.
"""
import os
import sys

import torch
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqdf_model import SeqDFModel
from data.video_dataset import build_dataloaders
from training.losses import set_detection_loss
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from utils.logger import TrainingLogger
from utils.metrics import compute_set_detection_metrics, average_metrics


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def compute_pos_weight(train_loader, num_methods: int, device) -> torch.Tensor:
    """
    Computes BCEWithLogitsLoss pos_weight from the actual training split's
    per-method positive rate, so rare method/position combinations don't get
    systematically under-predicted. With exactly 3-of-5 positive per video this
    is already fairly balanced (60% positive rate per method on average if
    methods are evenly distributed), but real generation runs are rarely
    perfectly uniform, so computing this from data is safer than assuming.
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
    # standard pos_weight formula: neg_count / pos_count, clamped to avoid
    # extreme weights if a method is very rare in this particular split
    pos_weight = (neg_rate / (pos_rate + 1e-8)).clamp(0.2, 5.0)
    print(f"  Per-method positive rate: {pos_rate.tolist()}")
    print(f"  Computed pos_weight: {pos_weight.tolist()}")
    return pos_weight.to(device)


def run_stage_a(cfgs: dict):
    set_seed(cfgs["data_config"]["split"]["random_seed"])
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    amp_enabled = cfgs["train_config"]["amp_enabled"] and device.type == "cuda"

    loaders = build_dataloaders(cfgs["paths"], cfgs["data_config"])
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = SeqDFModel(cfgs["model_config"]).to(device)

    stage_cfg = cfgs["train_config"]["stage_a_set_detection"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["epochs"])
    scaler = GradScaler(enabled=amp_enabled)

    logger = TrainingLogger(cfgs["paths"]["log_dir"], run_name="stage_a_set_detection")
    label_smoothing = cfgs["model_config"]["label_smoothing"]["set_detection"]
    num_methods = cfgs["data_config"]["num_methods"]
    top_k = cfgs["model_config"]["set_head"]["top_k"]

    pos_weight = None
    if stage_cfg.get("pos_weight_balancing", False):
        pos_weight = compute_pos_weight(train_loader, num_methods, device)

    best_exact_match = 0.0
    patience_counter = 0
    early_stop_cfg = cfgs["train_config"]["early_stopping"]

    for epoch in range(stage_cfg["epochs"]):
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
                         cfgs["paths"]["checkpoint_dir"], tag="stageA_set_detection")

        if avg_val_metrics["exact_set_match_acc"] > best_exact_match:
            best_exact_match = avg_val_metrics["exact_set_match_acc"]
            patience_counter = 0
            best_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageA_best.pt")
            torch.save(model.state_dict(), best_path)
            logger.log_text(f"  New best val_exact_set_match={best_exact_match:.4f} -> saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_cfg["patience"]:
                logger.log_text(f"Early stopping at epoch {epoch}")
                break

    logger.close()
    print(f"\nStage A complete. Best val_exact_set_match_acc={best_exact_match:.4f}")
    print("Reference range from chat discussion: expect ~35-55% exact set match "
          "(vs ~0.84% random chance for choosing 3-of-5), and ~75-88% per-method accuracy.")
    if best_exact_match > 0.65:
        print("NOTE: exact_set_match is notably above the expected range. Before "
              "celebrating, double check train/val splits are correctly isolated by "
              "source video id (see data/dataset_builder.py) and that no filename/"
              "metadata is leaking into the model inputs.")


if __name__ == "__main__":
    cfgs = load_configs()
    run_stage_a(cfgs)
