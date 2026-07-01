"""
Stage B: trains the small sequence-within-known-set decoder on top of a FROZEN,
already-trained Stage A model. Run this only after stageA_train_set_detection.py
has produced a good checkpoint at outputs/checkpoints/stageA_best.pt.

This stage is cheap relative to Stage A: the big video backbones (RGB/SRM/DCT)
are frozen and only run in eval/no_grad mode to produce fused embeddings; the
only thing being trained is the small StageBSequenceDecoder.

By default, this trains and evaluates using the GROUND TRUTH set (not Stage A's
predicted set) as the conditioning input - i.e. it measures "if we knew the
correct set, how well could we order it." This isolates Stage B's own quality
from Stage A's set-detection errors. For realistic deployment numbers, chain
Stage A's actual predicted set into Stage B's input at inference time instead -
see scripts/05_predict_single_two_stage.py for that full pipeline.
"""
import os
import sys

import torch
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqdf_model import SeqDFModel
from models.stage_b_sequence.sequence_decoder import StageBSequenceDecoder, set_to_one_hot
from data.video_dataset import build_dataloaders
from training.losses import sequence_ordering_loss
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from utils.logger import TrainingLogger
from utils.metrics import compute_stage_b_metrics, average_metrics


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def build_set_one_hot_batch(set_sorted_list: list, num_methods: int, device) -> torch.Tensor:
    """Builds a (B, num_methods) multi-hot tensor from a list of 'DF-F2F-FS' style strings."""
    vecs = [set_to_one_hot(s) for s in set_sorted_list]
    return torch.stack(vecs, dim=0).to(device)


def run_stage_b(cfgs: dict):
    set_seed(cfgs["data_config"]["split"]["random_seed"])
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    amp_enabled = cfgs["train_config"]["amp_enabled"] and device.type == "cuda"

    loaders = build_dataloaders(cfgs["paths"], cfgs["data_config"])
    train_loader, val_loader = loaders["train"], loaders["val"]

    # --- load frozen Stage A model ---
    stage_a_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageA_best.pt")
    if not os.path.exists(stage_a_path):
        raise FileNotFoundError(
            f"No trained Stage A model found at {stage_a_path}. "
            f"Run training/stageA_train_set_detection.py first."
        )
    stage_a_model = SeqDFModel(cfgs["model_config"]).to(device)
    stage_a_model.load_state_dict(torch.load(stage_a_path, map_location=device, weights_only=False))
    stage_a_model.freeze_all()
    print(f"Loaded frozen Stage A model from {stage_a_path}")

    # --- build trainable Stage B decoder ---
    sb_cfg = cfgs["model_config"]["stage_b_decoder"]
    fused_dim = cfgs["model_config"]["fusion"]["hidden_dims"][-1]
    num_methods = cfgs["data_config"]["num_methods"]

    decoder = StageBSequenceDecoder(
        fused_embedding_dim=fused_dim,
        hidden_dim=sb_cfg["hidden_dim"],
        num_layers=sb_cfg["num_layers"],
        num_heads=sb_cfg["num_heads"],
        num_orderings=sb_cfg["num_orderings"],
        num_methods=num_methods,
    ).to(device)

    stage_cfg = cfgs["train_config"]["stage_b_sequence"]
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["epochs"])
    scaler = GradScaler(enabled=amp_enabled)

    logger = TrainingLogger(cfgs["paths"]["log_dir"], run_name="stage_b_sequence")

    best_top1 = 0.0
    patience_counter = 0
    early_stop_cfg = cfgs["train_config"]["early_stopping"]

    for epoch in range(stage_cfg["epochs"]):
        decoder.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"StageB epoch {epoch}"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            srm = batch["srm"].to(device, non_blocking=True)
            dct = batch["dct"].to(device, non_blocking=True)
            ordering_targets = batch["labels"]["ordering_within_set"].to(device, non_blocking=True)
            set_one_hot = build_set_one_hot_batch(batch["set_sorted"], num_methods, device)

            with torch.no_grad():
                with autocast(enabled=amp_enabled):
                    stage_a_out = stage_a_model(rgb, srm, dct, return_embedding=True)
                    fused_embedding = stage_a_out["fused_embedding"]

            optimizer.zero_grad()
            with autocast(enabled=amp_enabled):
                ordering_logits = decoder(fused_embedding, set_one_hot)
                loss = sequence_ordering_loss(ordering_logits, ordering_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        scheduler.step()

        # --- validation (using GROUND TRUTH set as conditioning - see module docstring) ---
        decoder.eval()
        val_metrics_list = []
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                srm = batch["srm"].to(device, non_blocking=True)
                dct = batch["dct"].to(device, non_blocking=True)
                ordering_targets = batch["labels"]["ordering_within_set"].to(device, non_blocking=True)
                set_one_hot = build_set_one_hot_batch(batch["set_sorted"], num_methods, device)

                with autocast(enabled=amp_enabled):
                    stage_a_out = stage_a_model(rgb, srm, dct, return_embedding=True)
                    ordering_logits = decoder(stage_a_out["fused_embedding"], set_one_hot)

                val_metrics_list.append(compute_stage_b_metrics(ordering_logits, ordering_targets))

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_metrics = average_metrics(val_metrics_list)

        logger.log_text(
            f"[StageB][Epoch {epoch}] train_loss={avg_train_loss:.4f} | "
            f"val_top1={avg_val_metrics['ordering_top1_acc']:.4f} "
            f"val_top2={avg_val_metrics['ordering_top2_acc']:.4f}"
        )
        logger.log_scalar("stageB/train_loss", avg_train_loss, epoch)
        logger.log_metrics(avg_val_metrics, epoch, prefix="stageB/val_")

        save_checkpoint(decoder, optimizer, epoch, avg_val_metrics,
                         cfgs["paths"]["checkpoint_dir"], tag="stageB_sequence")

        if avg_val_metrics["ordering_top1_acc"] > best_top1:
            best_top1 = avg_val_metrics["ordering_top1_acc"]
            patience_counter = 0
            best_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageB_best.pt")
            torch.save(decoder.state_dict(), best_path)
            logger.log_text(f"  New best val_top1={best_top1:.4f} -> saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_cfg["patience"]:
                logger.log_text(f"Early stopping at epoch {epoch}")
                break

    logger.close()
    print(f"\nStage B complete. Best val ordering_top1_acc={best_top1:.4f} (6-way, "
          f"vs ~16.7% random chance). This number assumes the TRUE set is known - "
          f"see module docstring for why, and run scripts/05_predict_single_two_stage.py "
          f"for an end-to-end demo chaining Stage A's predicted set into Stage B.")


if __name__ == "__main__":
    cfgs = load_configs()
    run_stage_b(cfgs)
