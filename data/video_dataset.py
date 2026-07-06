"""
PyTorch Dataset serving (RGB clip, SRM clip, DCT clip, labels) tuples.

Uses the precomputed disk cache (data/preprocessing/cache_features.py) when available
and configured on in configs/paths.yaml; otherwise extracts everything live from the
raw .mp4 on each call (much slower - fine for quick testing, not for real training).
"""
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yaml

from data.frame_sampler import sample_frame_indices, extract_frames_at_indices
from data.preprocessing.srm_filters import extract_srm_residual_batch
from data.preprocessing.dct_features import extract_dct_energy_map_batch


class SeqDeepFakeDataset(Dataset):
    def __init__(self, split_csv_path: str, paths_cfg: dict, data_cfg: dict):
        """
        Args:
            split_csv_path: path to train.csv / val.csv / test.csv (from dataset_builder.py)
            paths_cfg: loaded configs/paths.yaml dict
            data_cfg: loaded configs/data_config.yaml dict
        """
        self.df = pd.read_csv(split_csv_path)
        self.paths_cfg = paths_cfg
        self.data_cfg = data_cfg
        self.use_cache = paths_cfg.get("use_feature_cache", True)
        self.cache_dir = paths_cfg.get("feature_cache_dir", None)
        self.resize_to = tuple(data_cfg["frame_sampling"]["resize_to"])
        self.num_frames = data_cfg["frame_sampling"]["num_frames"]
        self.sample_method = data_cfg["frame_sampling"]["method"]

    def __len__(self):
        return len(self.df)

    def _load_from_cache(self, row) -> dict:
        cache_path = os.path.join(self.cache_dir, row["filename"].replace(".mp4", ".pt"))
        return torch.load(cache_path, weights_only=False)

    def _load_live(self, row) -> dict:
        indices = sample_frame_indices(row["filepath"], num_frames=self.num_frames, method=self.sample_method)
        rgb_frames = extract_frames_at_indices(row["filepath"], indices, resize_to=self.resize_to)
        srm_frames = extract_srm_residual_batch(rgb_frames)
        dct_frames = extract_dct_energy_map_batch(rgb_frames, output_size=self.resize_to)
        return {
            "rgb": torch.from_numpy(rgb_frames),
            "srm": torch.from_numpy(srm_frames),
            "dct": torch.from_numpy(dct_frames),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cache_path = None
        if self.use_cache and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, row["filename"].replace(".mp4", ".pt"))

        if cache_path and os.path.exists(cache_path):
            data = self._load_from_cache(row)
            rgb, srm, dct = data["rgb"], data["srm"], data["dct"]
        else:
            data = self._load_live(row)
            rgb, srm, dct = data["rgb"], data["srm"], data["dct"]

        # RGB comes in as uint8 (T,H,W,3) -> normalize to [0,1] float, channel-first (T,3,H,W)
        rgb = rgb.float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2)  # (T, 3, H, W)

        # SRM and DCT are already float32, normalized at extraction time
        srm = srm.permute(0, 3, 1, 2) if srm.dim() == 4 else srm  # (T, 3, H, W)
        dct = dct.permute(0, 3, 1, 2) if dct.dim() == 4 else dct  # (T, 3, H, W)

        set_vector = torch.tensor(
            [int(v) for v in str(row["set_vector"]).split(",")], dtype=torch.float32
        )  # (5,) multi-hot, e.g. [1,1,0,1,0] - PRIMARY label for Stage A

        labels = {
            "m1_idx": torch.tensor(row["m1_idx"], dtype=torch.long),
            "m2_idx": torch.tensor(row["m2_idx"], dtype=torch.long),
            "m3_idx": torch.tensor(row["m3_idx"], dtype=torch.long),
            "sequence_class": torch.tensor(row["sequence_class"], dtype=torch.long),
            "set_vector": set_vector,
            "ordering_within_set": torch.tensor(row["ordering_within_set"], dtype=torch.long),  # used by Stage B only
        }

        return {
            "rgb": rgb, "srm": srm, "dct": dct,
            "labels": labels,
            "video_id": row["video_id"],
            "filename": row["filename"],
            "set_sorted": row["set_sorted"],
        }


def build_dataloaders(paths_cfg: dict, data_cfg: dict):
    """Convenience function: builds train/val/test DataLoaders from the split CSVs."""
    from torch.utils.data import DataLoader

    dl_cfg = data_cfg["dataloader"]
    num_workers = dl_cfg["num_workers"]

    # persistent_workers and prefetch_factor are only valid when num_workers > 0
    # (PyTorch raises if you pass them with num_workers=0). Using .get() with
    # defaults keeps this backward-compatible with older data_config.yaml files
    # that don't have these keys at all.
    persistent_workers = dl_cfg.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = dl_cfg.get("prefetch_factor", None) if num_workers > 0 else None

    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(paths_cfg["splits_dir"], f"{split}.csv")
        dataset = SeqDeepFakeDataset(csv_path, paths_cfg, data_cfg)

        dl_kwargs = dict(
            batch_size=dl_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=dl_cfg["pin_memory"],
            drop_last=(split == "train"),
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = prefetch_factor

        loaders[split] = DataLoader(dataset, **dl_kwargs)

    print(f"[DataLoader config] num_workers={num_workers}  pin_memory={dl_cfg['pin_memory']}  "
          f"persistent_workers={persistent_workers}  prefetch_factor={prefetch_factor}  "
          f"batch_size={dl_cfg['batch_size']}")

    return loaders


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "configs", "paths.yaml")) as f:
        paths_cfg = yaml.safe_load(f)
    with open(os.path.join(base, "configs", "data_config.yaml")) as f:
        data_cfg = yaml.safe_load(f)

    loaders = build_dataloaders(paths_cfg, data_cfg)
    batch = next(iter(loaders["train"]))
    print("RGB shape:", batch["rgb"].shape)
    print("SRM shape:", batch["srm"].shape)
    print("DCT shape:", batch["dct"].shape)
    print("Labels:", {k: v.shape for k, v in batch["labels"].items()})