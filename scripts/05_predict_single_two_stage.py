"""
End-to-end inference on a single .mp4: runs Stage A to predict which 3 methods
were used, then chains that predicted set into Stage B to predict their order.

This is the realistic deployment path - Stage A's predicted set (not ground
truth) feeds Stage B, so end-to-end accuracy will be somewhat lower than
Stage B's own validation numbers (which assume the true set is known - see
training/stageB_train_sequence.py docstring).

Usage:
    python scripts/05_predict_single_two_stage.py --video "C:\\path\\to\\video.mp4"
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seqdf_model import SeqDFModel
from models.fusion.set_detection_head import predict_set_top_k
from models.stage_b_sequence.sequence_decoder import StageBSequenceDecoder, ordering_index_to_methods
from data.frame_sampler import sample_frame_indices, extract_frames_at_indices
from data.preprocessing.srm_filters import extract_srm_residual_batch
from data.preprocessing.dct_features import extract_dct_energy_map_batch
from data.filename_parser import IDX_TO_METHOD


def load_configs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfgs = {}
    for name in ["paths", "data_config", "model_config", "train_config"]:
        with open(os.path.join(base, "configs", f"{name}.yaml")) as f:
            cfgs[name] = yaml.safe_load(f)
    return cfgs


def extract_tensors(video_path: str, cfgs: dict, device):
    fs_cfg = cfgs["data_config"]["frame_sampling"]
    resize_to = tuple(fs_cfg["resize_to"])

    print(f"Sampling {fs_cfg['num_frames']} frames from {video_path} ...")
    indices = sample_frame_indices(video_path, num_frames=fs_cfg["num_frames"], method=fs_cfg["method"])
    rgb_frames = extract_frames_at_indices(video_path, indices, resize_to=resize_to)

    print("Extracting SRM and DCT features ...")
    srm_frames = extract_srm_residual_batch(rgb_frames)
    dct_frames = extract_dct_energy_map_batch(rgb_frames, output_size=resize_to)

    rgb_tensor = torch.from_numpy(rgb_frames).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    srm_tensor = torch.from_numpy(srm_frames).permute(0, 3, 1, 2).unsqueeze(0)
    dct_tensor = torch.from_numpy(dct_frames).permute(0, 3, 1, 2).unsqueeze(0)

    return rgb_tensor.to(device), srm_tensor.to(device), dct_tensor.to(device)


def predict_single_video(video_path: str, cfgs: dict, run_stage_b: bool = True):
    device = torch.device(cfgs["train_config"]["device"] if torch.cuda.is_available() else "cpu")
    num_methods = cfgs["data_config"]["num_methods"]
    top_k = cfgs["model_config"]["set_head"]["top_k"]

    rgb, srm, dct = extract_tensors(video_path, cfgs, device)

    # --- Stage A: predict the set ---
    print("Loading Stage A model ...")
    stage_a_model = SeqDFModel(cfgs["model_config"]).to(device)
    stage_a_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageA_best.pt")
    if not os.path.exists(stage_a_path):
        raise FileNotFoundError(f"No trained Stage A model at {stage_a_path}. Run Stage A training first.")
    stage_a_model.load_state_dict(torch.load(stage_a_path, map_location=device, weights_only=False))
    stage_a_model.eval()

    with torch.no_grad():
        stage_a_out = stage_a_model(rgb, srm, dct, return_embedding=True)

    set_probs = torch.sigmoid(stage_a_out["set_logits"])[0]
    pred_multi_hot = predict_set_top_k(stage_a_out["set_logits"], k=top_k)[0]
    predicted_methods = sorted(IDX_TO_METHOD[i] for i in range(num_methods) if pred_multi_hot[i] == 1)
    set_sorted_str = "-".join(predicted_methods)

    print("\n========== STAGE A: SET DETECTION ==========")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Predicted methods used (order-agnostic): {predicted_methods}")
    print("Per-method confidence:")
    for i in range(num_methods):
        marker = " <-- selected" if pred_multi_hot[i] == 1 else ""
        print(f"  {IDX_TO_METHOD[i]:5s}: {set_probs[i].item():.2%}{marker}")
    print("==============================================")

    if not run_stage_b:
        return {"predicted_set": predicted_methods}

    # --- Stage B: predict ordering within the predicted set ---
    stage_b_path = os.path.join(cfgs["paths"]["checkpoint_dir"], "stageB_best.pt")
    if not os.path.exists(stage_b_path):
        print(f"\nNo trained Stage B model found at {stage_b_path} - skipping ordering "
              f"prediction. Run training/stageB_train_sequence.py to enable this step.")
        return {"predicted_set": predicted_methods}

    print("\nLoading Stage B model ...")
    sb_cfg = cfgs["model_config"]["stage_b_decoder"]
    fused_dim = cfgs["model_config"]["fusion"]["hidden_dims"][-1]
    stage_b_decoder = StageBSequenceDecoder(
        fused_embedding_dim=fused_dim,
        hidden_dim=sb_cfg["hidden_dim"],
        num_layers=sb_cfg["num_layers"],
        num_heads=sb_cfg["num_heads"],
        num_orderings=sb_cfg["num_orderings"],
        num_methods=num_methods,
    ).to(device)
    stage_b_decoder.load_state_dict(torch.load(stage_b_path, map_location=device, weights_only=False))
    stage_b_decoder.eval()

    set_one_hot = pred_multi_hot.unsqueeze(0).float()
    with torch.no_grad():
        ordering_logits = stage_b_decoder(stage_a_out["fused_embedding"], set_one_hot)

    ordering_probs = F.softmax(ordering_logits, dim=-1)[0]
    pred_ordering_idx = int(ordering_probs.argmax().item())
    pred_m1, pred_m2, pred_m3 = ordering_index_to_methods(set_sorted_str, pred_ordering_idx)
    confidence = float(ordering_probs[pred_ordering_idx].item())

    top2_idx = ordering_probs.topk(2).indices.tolist()
    top2_orderings = [ordering_index_to_methods(set_sorted_str, i) for i in top2_idx]

    print("\n========== STAGE B: SEQUENCE ORDERING ==========")
    print(f"Predicted order: m1={pred_m1}, m2={pred_m2}, m3={pred_m3}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Top-2 orderings: {top2_orderings}")
    print("==================================================")
    print("\nNote: this assumes Stage A's predicted set is correct. If Stage A's set "
          "prediction was wrong, Stage B's ordering prediction is conditioned on the "
          "wrong set and should not be trusted - check Stage A's per-method "
          "confidences above before relying on the ordering result.")

    return {
        "predicted_set": predicted_methods,
        "predicted_order": {"m1": pred_m1, "m2": pred_m2, "m3": pred_m3},
        "ordering_confidence": confidence,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the .mp4 file to analyze")
    parser.add_argument("--skip_stage_b", action="store_true", help="Only run Stage A set detection")
    args = parser.parse_args()

    cfgs = load_configs()
    predict_single_video(args.video, cfgs, run_stage_b=not args.skip_stage_b)
