"""
Standalone diagnostic - does NOT modify your real training script.

Times each phase of a Stage A training step separately across N iterations:
  1. dataloader fetch (cache read + collate)
  2. host->device transfer
  3. model forward
  4. loss + backward
  5. optimizer step

Also runs torch.profiler for one iteration so you get a proper op-level
breakdown (which backbone / which kernel is actually dominating).

Automatically detects set_head.type from model_config.yaml (multilabel_mlp or
classification_10way) and uses the matching loss function - safe to run
regardless of which head you currently have configured.

Usage (from the seqdf/ project root, same place you run stageA_train_set_detection.py):
    python profile_stageA_step.py --iters 15
"""
import argparse
import os
import sys
import time

import torch
from torch.cuda.amp import autocast, GradScaler
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.seqdf_model import SeqDFModel
from models.fusion.set_detection_head import multi_hot_to_class_idx
from data.video_dataset import build_dataloaders
from training.losses import set_detection_loss, set_classification_loss


def load_configs():
    base = os.path.dirname(os.path.abspath(__file__))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def compute_loss(set_head_type, set_logits, set_targets, label_smoothing):
    if set_head_type == "classification_10way":
        class_targets = multi_hot_to_class_idx(set_targets)
        return set_classification_loss(set_logits, class_targets, label_smoothing=label_smoothing)
    else:
        return set_detection_loss(set_logits, set_targets, label_smoothing=label_smoothing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()

    cfgs = load_configs()
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    amp_enabled = cfgs["train_config"]["amp_enabled"] and device.type == "cuda"

    print("Building dataloaders...")
    loaders = build_dataloaders(cfgs["paths"], cfgs["data_config"])
    train_loader = loaders["train"]

    print("Building model...")
    model = SeqDFModel(cfgs["model_config"]).to(device)
    model.train()
    set_head_type = model.set_head_type
    print(f"[Model] set_head_type = {set_head_type}")

    stage_cfg = cfgs["train_config"]["stage_a_set_detection"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    scaler = GradScaler(enabled=amp_enabled)
    label_smoothing = cfgs["model_config"]["label_smoothing"]["set_detection"]

    times = {"data_fetch": [], "h2d": [], "forward": [], "loss_backward": [], "opt_step": [], "total": []}

    data_iter = iter(train_loader)

    print(f"\nRunning {args.iters} timed iterations (AMP={'on' if amp_enabled else 'off'})...\n")

    for i in range(args.iters):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        t0 = time.perf_counter()
        batch = next(data_iter)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rgb = batch["rgb"].to(device, non_blocking=True)
        srm = batch["srm"].to(device, non_blocking=True)
        dct = batch["dct"].to(device, non_blocking=True)
        set_targets = batch["labels"]["set_vector"].to(device, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            outputs = model(rgb, srm, dct)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        with autocast(enabled=amp_enabled):
            loss = compute_loss(set_head_type, outputs["set_logits"], set_targets, label_smoothing)
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        times["data_fetch"].append(t1 - t0)
        times["h2d"].append(t2 - t1)
        times["forward"].append(t3 - t2)
        times["loss_backward"].append(t4 - t3)
        times["opt_step"].append(t5 - t4)
        times["total"].append(t5 - t_step_start)

        print(f"  iter {i:2d}  total={t5 - t_step_start:6.3f}s  "
              f"data={t1 - t0:6.3f}s  h2d={t2 - t1:6.3f}s  "
              f"fwd={t3 - t2:6.3f}s  bwd={t4 - t3:6.3f}s  opt={t5 - t4:6.3f}s  "
              f"loss={loss.item():.4f}")

    print("\n=== AVERAGES (excluding first iter - warmup) ===")
    for k, v in times.items():
        avg = sum(v[1:]) / max(len(v[1:]), 1)
        pct = (avg / (sum(times['total'][1:]) / len(times['total'][1:]))) * 100 if k != "total" else 100.0
        print(f"  {k:15s}: {avg:6.3f}s  ({pct:5.1f}% of step)")

    print("\n=== Running torch.profiler for 1 extra iteration (op-level breakdown) ===")
    batch = next(data_iter)
    rgb = batch["rgb"].to(device, non_blocking=True)
    srm = batch["srm"].to(device, non_blocking=True)
    dct = batch["dct"].to(device, non_blocking=True)
    set_targets = batch["labels"]["set_vector"].to(device, non_blocking=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            outputs = model(rgb, srm, dct)
            loss = compute_loss(set_head_type, outputs["set_logits"], set_targets, label_smoothing)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()      