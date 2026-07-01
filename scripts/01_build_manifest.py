"""
Run this first. Scans configs/paths.yaml -> generated_videos_dir, parses every
filename into labels, writes the manifest CSV, and builds train/val/test splits
(split by source video id - see data/dataset_builder.py for why this matters).

Usage:
    python scripts/01_build_manifest.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset_builder import main

if __name__ == "__main__":
    main()
