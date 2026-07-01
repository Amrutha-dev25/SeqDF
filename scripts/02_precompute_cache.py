"""
Run this after 01_build_manifest.py and before training, if use_feature_cache is
true in configs/paths.yaml (recommended). Precomputes RGB/SRM/DCT tensors for every
video in the manifest and saves them to disk, so training doesn't have to redo
expensive video decoding + SRM/DCT extraction on every epoch across all 4 stages.

This step can take a long time on first run (5580 videos x frame sampling + SRM +
per-block DCT extraction). It is safe to interrupt and re-run - already-cached
videos are skipped automatically.

Usage:
    python scripts/02_precompute_cache.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing.cache_features import main

if __name__ == "__main__":
    main()
