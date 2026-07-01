# Sequential DeepFake Manipulation Detection (Set-Detection-First)

Detects which manipulation methods were chained into a composite deepfake video
built from FaceForensics++ (DF, F2F, FSh, FS, NT), using a two-stage approach:

- **Stage A (primary, well-posed):** predict WHICH 3 of 5 methods were used,
  order-agnostic. Multi-label classification.
- **Stage B (secondary, smaller):** given a known/predicted 3-method set,
  predict the ORDER they were applied in. 6-way classification (3! orderings).

Filename convention: `{video_id}_{m1}_{m2}_{m3}.mp4`  e.g. `000_DF_FS_F2F.mp4`

## Why set-detection-first

The original framing (predict the full ordered 60-way sequence directly) runs
into a real information bottleneck: your generation pipeline alpha-blends each
step on top of the last, so the *first* manipulation's evidence is degraded by
two further rounds of overwriting before any model sees it. That makes ordered
prediction of m1 specifically very hard (~25-35% accuracy ceiling on a 5-way
sub-task), dragging down full-sequence accuracy to ~13-18% exact match.

Set detection sidesteps this: a method's evidence doesn't need to survive at a
*specific position* in the chain - it only needs to survive *somewhere*, in any
of the 3 feature streams (RGB/SRM/DCT), at any frame. That's a fundamentally
lower bar, turns the problem into well-posed multi-label classification, and
gives much stronger, more defensible numbers. Sequence ordering is then handled
as a separate, smaller, downstream problem conditioned on the now-known set,
which shrinks the search space from 60 classes to 6.

## Architecture

```
                         INPUT: one .mp4 video
                                  |
                  Diff-based 16-frame sampling, resize 224x224
                                  |
        +-------------------------+-------------------------+
        v                         v                         v
   RGB frames               SRM residual              DCT energy map
  16x224x224x3            (3 high-pass kernels)      (8x8 block, 3 bands)
        |                         |                         |
        v                         v                         v
  Video Swin-T              EfficientNet-B4            ResNet-18 (modified)
  (fallback: 3D-CNN)        x16 frames -> TCN          x16 frames -> Transformer
        | 768-d                   | 512-d                   | 512-d
        +-----------------+--------+-----------------+-------+
                          v
              concat (1792-d) -> Fusion MLP -> 512 -> 256
                          |
        +------------------+-----------------------+
        v  STAGE A (trained first)                  v  STAGE B (trained after, frozen Stage A)
  Multi-label set head                  fused embedding (256-d, frozen)
  256 -> 256 -> 5 sigmoid outputs       + one-hot of predicted/known set (5-d)
  "is method X present anywhere?"                    |
        |                                            v
        v                                  small Transformer (1 layer, 2 heads)
  top-3 selection                                    |
  -> predicted 3-method SET                          v
                                            6-way classifier
                                            -> predicted ORDER within that set
```

## Folder structure

```
seqdf/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- paths.yaml              # <- EDIT THIS FIRST (all your Windows paths)
|   |-- data_config.yaml        # frame sampling, resolution, splits, method list
|   |-- model_config.yaml       # backbone choices, set_head, stage_b_decoder
|   |-- train_config.yaml       # LR, epochs, batch size, stage_a / stage_b schedule
|
|-- data/
|   |-- filename_parser.py      # parses filenames -> set_vector + ordering labels
|   |-- dataset_builder.py      # scans generated_videos/, builds manifest + splits
|   |-- frame_sampler.py        # 16-frame diff-based sampling
|   |-- video_dataset.py        # PyTorch Dataset: video -> RGB/SRM/DCT + labels
|   |-- preprocessing/
|       |-- srm_filters.py      # SRM residual extraction
|       |-- dct_features.py     # block DCT energy map extraction
|       |-- cache_features.py   # precompute + cache tensors to disk as .pt
|
|-- models/
|   |-- backbones/
|   |   |-- rgb_backbone.py     # Video Swin-T wrapper
|   |   |-- srm_backbone.py     # EfficientNet-B4 + TCN
|   |   |-- dct_backbone.py     # ResNet-18 + Transformer pool
|   |-- fusion/
|   |   |-- fusion_mlp.py       # concat + MLP fusion
|   |   |-- set_detection_head.py  # Stage A: multi-label set head
|   |-- stage_b_sequence/
|   |   |-- sequence_decoder.py # Stage B: ordering-within-set decoder
|   |-- seqdf_model.py          # full Stage A model wiring everything together
|
|-- training/
|   |-- losses.py                       # BCE for Stage A, CE for Stage B
|   |-- stageA_train_set_detection.py   # trains the full 3-stream model jointly
|   |-- stageB_train_sequence.py        # trains the small ordering decoder
|   |-- evaluate_stageA.py              # test-set evaluation for Stage A
|
|-- utils/
|   |-- seed.py / checkpoint.py / logger.py
|   |-- metrics.py               # per-method acc/F1, exact-set-match, ordering acc
|
|-- scripts/
|   |-- 01_build_manifest.py            # scan generated_videos/ -> manifest + splits
|   |-- 02_precompute_cache.py          # optional: precompute frame/SRM/DCT cache
|   |-- 03_run_full_pipeline.py         # runs Stage A then Stage B training
|   |-- 05_predict_single_two_stage.py  # inference: Stage A -> Stage B chained
|
|-- outputs/
    |-- checkpoints/   # stageA_best.pt, stageB_best.pt, per-epoch checkpoints
    |-- logs/          # tensorboard + text logs
    |-- predictions/   # test-set CSVs
```

## Commands to run (in order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit configs/paths.yaml - confirm ffpp_root and generated_videos_dir
#    match your actual Windows folders

# 3. Build the manifest (parses all 5580 filenames into set + ordering labels,
#    splits train/val/test by SOURCE video id so no face/identity leaks across splits)
python scripts/01_build_manifest.py

# 4. (Recommended) Precompute RGB/SRM/DCT feature cache - slow on first run,
#    safe to interrupt and resume, dramatically speeds up every epoch after
python scripts/02_precompute_cache.py

# 5. Train Stage A (set detection) - the main, well-posed task
python training/stageA_train_set_detection.py
#    --- OR run the whole pipeline (Stage A then Stage B) in one command: ---
python scripts/03_run_full_pipeline.py

# 6. Train Stage B (ordering within known set) - run only after Stage A
#    has a good checkpoint at outputs/checkpoints/stageA_best.pt
python training/stageB_train_sequence.py

# 7. Evaluate Stage A on the held-out test set
python training/evaluate_stageA.py

# 8. Run end-to-end inference on a single new video (Stage A -> Stage B chained)
python scripts/05_predict_single_two_stage.py --video "C:\Users\Student\Desktop\spoorthi\generated_videos\000_DF_FS_F2F.mp4"

#    To run Stage A only (skip ordering prediction):
python scripts/05_predict_single_two_stage.py --video "<path>.mp4" --skip_stage_b
```

## Expected accuracy

| Task | Metric | Expected range | Random chance baseline |
|---|---|---|---|
| Stage A | per-method accuracy (avg of 5) | 75-88% | 50% (independent binary) |
| Stage A | exact set match (all 3 correct) | 35-55% | ~0.84% (choosing 3-of-5) |
| Stage B | ordering top-1 (6-way, true set known) | 30-50%* | ~16.7% |
| End-to-end | set correct AND order correct | roughly Stage A exact-match x Stage B top-1, ~15-25% | ~0.14% |

*Stage B's range is a rough prior, not validated against external literature the
way Stage A's is - check actual numbers once trained, since how much positional
information survives Stage A's set-level supervision is genuinely uncertain in
advance.

If Stage A's exact-set-match comes out much above ~65%, or Stage B's top-1
comes out much above ~60%, treat that as a prompt to check for label leakage
(e.g. filename or metadata reaching the model) before reporting the number,
not as a result to celebrate immediately.

## Notes

- Stage A trains the full 3-stream backbone (RGB/SRM/DCT) jointly from the start
  - there's no per-position freezing schedule here, since "is method X present
  anywhere" doesn't have the same difficulty gradient across streams that
  ordered first/middle/last prediction did.
- Stage B is cheap to train: it reuses Stage A's frozen fused embedding as input
  and only trains a small 1-layer Transformer + linear classifier on top.
- Stage B's validation metric uses the TRUE set as conditioning (isolating its
  own quality from Stage A's errors). `scripts/05_predict_single_two_stage.py`
  chains Stage A's actual predicted set into Stage B for realistic end-to-end
  numbers - expect this to be somewhat lower than Stage B's own validation
  accuracy, proportional to Stage A's error rate.
