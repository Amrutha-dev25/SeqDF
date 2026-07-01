"""
Scans the flat generated_videos/ folder, parses every filename into labels via
filename_parser, and writes:
  1. A full manifest CSV (one row per video, with full path + all label fields)
  2. train/val/test split CSVs, split by SOURCE VIDEO ID (not by output video)

Why split by source id and not by output video:
Each of the 93 source videos produces up to 60 output videos (one per sequence).
All 60 outputs from the same source share the same underlying face/identity/background.
If train and val both contain outputs from source "007", the model can learn to
recognize "this is source 007's face" rather than learning manipulation-method
fingerprints, and validation accuracy will be inflated and won't reflect real
generalization. Splitting by source id guarantees no source's face appears in
more than one split.
"""
import os
import sys
import glob
import random

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.filename_parser import try_parse_filename


def load_config():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "configs", "paths.yaml")) as f:
        paths = yaml.safe_load(f)
    with open(os.path.join(base, "configs", "data_config.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    return paths, data_cfg


def scan_generated_videos(generated_videos_dir: str) -> pd.DataFrame:
    """Scans the flat folder of generated videos and parses every filename."""
    mp4_paths = glob.glob(os.path.join(generated_videos_dir, "*.mp4"))
    if not mp4_paths:
        raise FileNotFoundError(
            f"No .mp4 files found in {generated_videos_dir}. "
            f"Check configs/paths.yaml -> generated_videos_dir is correct."
        )

    rows = []
    skipped = []
    for path in mp4_paths:
        parsed = try_parse_filename(os.path.basename(path))
        if parsed is None:
            skipped.append(os.path.basename(path))
            continue
        parsed["filepath"] = path
        rows.append(parsed)

    if skipped:
        print(f"WARNING: skipped {len(skipped)} files that didn't match the expected "
              f"naming pattern. First few: {skipped[:5]}")

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} videos from {len(df['video_id'].unique())} unique source ids.")
    return df


def make_splits(df: pd.DataFrame, data_cfg: dict) -> dict:
    """Splits by unique source video_id so no source's outputs cross split boundaries."""
    split_cfg = data_cfg["split"]
    rng = random.Random(split_cfg["random_seed"])

    unique_ids = sorted(df["video_id"].unique())
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(n * split_cfg["train_frac"])
    n_val = int(n * split_cfg["val_frac"])

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    splits = {
        "train": df[df["video_id"].isin(train_ids)].reset_index(drop=True),
        "val": df[df["video_id"].isin(val_ids)].reset_index(drop=True),
        "test": df[df["video_id"].isin(test_ids)].reset_index(drop=True),
    }

    for name, split_df in splits.items():
        print(f"  {name}: {len(split_df)} videos from {split_df['video_id'].nunique()} source ids")

    return splits


def main():
    paths, data_cfg = load_config()

    print(f"Scanning {paths['generated_videos_dir']} ...")
    df = scan_generated_videos(paths["generated_videos_dir"])

    os.makedirs(os.path.dirname(paths["manifest_csv"]), exist_ok=True)
    df.to_csv(paths["manifest_csv"], index=False)
    print(f"Wrote full manifest -> {paths['manifest_csv']}")

    # class balance sanity check on SETS (Stage A's actual target) - flag if any of
    # the 10 unordered method-triples are badly underrepresented
    set_counts = df["set_sorted"].value_counts()
    print(f"\nSet balance (10 possible unordered method-triples): "
          f"min={set_counts.min()}, max={set_counts.max()}, mean={set_counts.mean():.1f} videos/set")
    if set_counts.min() < 20:
        under = set_counts[set_counts < 20]
        print(f"WARNING: {len(under)} method-sets have fewer than 20 examples total. "
              f"Consider oversampling these for Stage A training.")

    # secondary check on the full 60-way ordered sequence distribution, relevant
    # only once you get to Stage B (sequence-within-set) training
    seq_counts = df["sequence_class"].value_counts()
    print(f"\nSequence balance (60 possible ordered triples, relevant for Stage B): "
          f"min={seq_counts.min()}, max={seq_counts.max()}, mean={seq_counts.mean():.1f} videos/class")

    print("\nBuilding splits (by source video id) ...")
    splits = make_splits(df, data_cfg)

    os.makedirs(paths["splits_dir"], exist_ok=True)
    for name, split_df in splits.items():
        out_path = os.path.join(paths["splits_dir"], f"{name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"Wrote {name} split -> {out_path}")

    print("\nDone. Per-position label distribution in train split:")
    train_df = splits["train"]
    for pos in ["m1", "m2", "m3"]:
        print(f"  {pos}: {dict(train_df[pos].value_counts())}")


if __name__ == "__main__":
    main()
