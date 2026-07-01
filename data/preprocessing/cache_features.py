"""
Precomputes and caches per-video RGB/SRM/DCT tensors to disk as .pt files.

This matters a lot for training speed: without caching, every epoch re-decodes
every video from scratch (frame sampling + SRM conv + per-block DCT loop), which
is the dominant cost by far compared to actual model forward/backward passes.
With ~5580 videos x potentially 60+ epochs across 4 training stages, caching pays
for itself almost immediately.

Cache layout: {feature_cache_dir}/{video_id}_{m1}_{m2}_{m3}.pt containing a dict:
    {
      "rgb":  (16, 224, 224, 3) uint8 tensor,
      "srm":  (16, 224, 224, 3) float32 tensor,
      "dct":  (16, 224, 224, 3) float32 tensor,
      "label": {...same fields as filename_parser output...}
    }
"""
import os
import sys

import torch
import pandas as pd
from tqdm import tqdm
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.frame_sampler import sample_frame_indices, extract_frames_at_indices
from data.preprocessing.srm_filters import extract_srm_residual_batch
from data.preprocessing.dct_features import extract_dct_energy_map_batch


def load_config():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(base, "configs", "paths.yaml")) as f:
        paths = yaml.safe_load(f)
    with open(os.path.join(base, "configs", "data_config.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    return paths, data_cfg


def precompute_one_video(row: dict, data_cfg: dict) -> dict:
    fs_cfg = data_cfg["frame_sampling"]
    resize_to = tuple(fs_cfg["resize_to"])

    indices = sample_frame_indices(
        row["filepath"],
        num_frames=fs_cfg["num_frames"],
        method=fs_cfg["method"],
    )
    rgb_frames = extract_frames_at_indices(row["filepath"], indices, resize_to=resize_to)  # (T,H,W,3) uint8

    srm_frames = extract_srm_residual_batch(rgb_frames)  # (T,H,W,3) float32
    dct_frames = extract_dct_energy_map_batch(rgb_frames, output_size=resize_to)  # (T,H,W,3) float32

    return {
        "rgb": torch.from_numpy(rgb_frames),
        "srm": torch.from_numpy(srm_frames),
        "dct": torch.from_numpy(dct_frames),
        "label": {
            "video_id": row["video_id"],
            "m1_idx": row["m1_idx"], "m2_idx": row["m2_idx"], "m3_idx": row["m3_idx"],
            "sequence_class": row["sequence_class"],
            "set_vector": row["set_vector"],
            "set_sorted": row["set_sorted"],
            "ordering_within_set": row["ordering_within_set"],
        },
    }


def main():
    paths, data_cfg = load_config()

    if not paths.get("use_feature_cache", True):
        print("use_feature_cache is False in configs/paths.yaml - nothing to do.")
        return

    cache_dir = paths["feature_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    manifest_path = paths["manifest_csv"]
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_build_manifest.py first."
        )

    df = pd.read_csv(manifest_path)
    print(f"Precomputing features for {len(df)} videos -> {cache_dir}")

    failed = []
    skipped_existing = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(cache_dir, row["filename"].replace(".mp4", ".pt"))
        if os.path.exists(out_path):
            skipped_existing += 1
            continue
        try:
            cached = precompute_one_video(row.to_dict(), data_cfg)
            torch.save(cached, out_path)
        except Exception as e:
            print(f"FAILED on {row['filename']}: {e}")
            failed.append(row["filename"])

    print(f"\nDone. Skipped {skipped_existing} already-cached videos. "
          f"Failed on {len(failed)} videos.")
    if failed:
        fail_log = os.path.join(cache_dir, "_failed_videos.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed video list written to {fail_log} - re-run this script after "
              f"investigating, it will skip already-succeeded videos.")


if __name__ == "__main__":
    main()
