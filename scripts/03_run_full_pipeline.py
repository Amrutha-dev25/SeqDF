"""
Runs the full two-stage training pipeline:
  Stage A: 3-stream model (RGB/SRM/DCT) -> multi-label set detection
           ("which 3 of 5 methods were used, order-agnostic")
  Stage B: small decoder on frozen Stage A embeddings -> ordering within
           the known/predicted set (6-way: 3! orderings)

Usage:
    python scripts/03_run_full_pipeline.py
    python scripts/03_run_full_pipeline.py --stage A     # Stage A only
    python scripts/03_run_full_pipeline.py --stage B     # Stage B only (needs Stage A done first)
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.stageA_train_set_detection import run_stage_a, load_configs as load_cfgs_a
from training.stageB_train_sequence import run_stage_b, load_configs as load_cfgs_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="ALL", choices=["ALL", "A", "B"],
                         help="Run Stage A only, Stage B only, or ALL (default).")
    args = parser.parse_args()

    if args.stage in ("ALL", "A"):
        print("\n" + "=" * 60)
        print("STAGE A: 3-stream model -> multi-label set detection")
        print("=" * 60)
        run_stage_a(load_cfgs_a())

    if args.stage in ("ALL", "B"):
        print("\n" + "=" * 60)
        print("STAGE B: sequence-within-known-set decoder")
        print("=" * 60)
        run_stage_b(load_cfgs_b())

    print("\nPipeline complete. Run training/evaluate_stageA.py for Stage A test metrics, "
          "or scripts/05_predict_single_two_stage.py for end-to-end single-video inference.")


if __name__ == "__main__":
    main()
