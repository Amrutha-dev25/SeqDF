"""
Evaluates the trained Stage A model on the held-out test set: per-method
accuracy/F1, exact set match accuracy, and a per-video prediction breakdown.

For end-to-end (Stage A + Stage B chained) evaluation including ordering
accuracy, see scripts/05_predict_single_two_stage.py for the single-video
version.
"""
import os
import sys

import torch
from torch.cuda.amp import autocast
import yaml
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqdf_model import SeqDFModel
from models.fusion.set_detection_head import predict_set_top_k
from data.video_dataset import build_dataloaders
from data.filename_parser import IDX_TO_METHOD
from utils.metrics import compute_set_detection_metrics, average_metrics


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def multi_hot_to_method_list(multi_hot_row, num_methods: int) -> str:
    return "-".join(sorted(IDX_TO_METHOD[i] for i in range(num_methods) if multi_hot_row[i] == 1))


def run_evaluation(cfgs: dict):
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    amp_enabled = cfgs["train_config"]["amp_enabled"] and device.type == "cuda"
    num_methods = cfgs["data_config"]["num_methods"]
    top_k = cfgs["model_config"]["set_head"]["top_k"]

    loaders = build_dataloaders(cfgs["paths"], cfgs["data_config"])
    test_loader = loaders["test"]

    model = SeqDFModel(cfgs["model_config"]).to(device)
    best_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageA_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No best Stage A checkpoint found at {best_path}. Run Stage A training first.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    model.eval()

    all_metrics = []
    per_video_results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Stage A on test set"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            srm = batch["srm"].to(device, non_blocking=True)
            dct = batch["dct"].to(device, non_blocking=True)
            set_targets = batch["labels"]["set_vector"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                outputs = model(rgb, srm, dct)

            batch_metrics = compute_set_detection_metrics(outputs["set_logits"], set_targets, top_k=top_k)
            all_metrics.append(batch_metrics)

            pred_multi_hot = predict_set_top_k(outputs["set_logits"], k=top_k).cpu().numpy()
            true_multi_hot = set_targets.cpu().numpy()

            for i, fname in enumerate(batch["filename"]):
                per_video_results.append({
                    "filename": fname,
                    "true_set": multi_hot_to_method_list(true_multi_hot[i], num_methods),
                    "pred_set": multi_hot_to_method_list(pred_multi_hot[i], num_methods),
                    "exact_match": bool((pred_multi_hot[i] == true_multi_hot[i]).all()),
                })

    summary = average_metrics(all_metrics)

    print("\n========== STAGE A TEST SET RESULTS ==========")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print("================================================")
    print("\nReference ranges from chat discussion:")
    print("  per_method_acc:      expected 0.75-0.88")
    print("  exact_set_match_acc: expected 0.35-0.55 (vs ~0.0084 random chance for 3-of-5)")

    os.makedirs(cfgs["paths"]["predictions_dir"], exist_ok=True)
    results_df = pd.DataFrame(per_video_results)
    out_path = os.path.join(cfgs["paths"]["predictions_dir"], "stageA_test_predictions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nPer-video predictions written to {out_path}")

    summary_path = os.path.join(cfgs["paths"]["predictions_dir"], "stageA_test_summary_metrics.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary metrics written to {summary_path}")


if __name__ == "__main__":
    cfgs = load_configs()
    run_evaluation(cfgs)
