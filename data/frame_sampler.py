"""
Samples 16 frames per video using difference-based sampling: computes inter-frame
absolute difference across the whole video, and picks the frames with the highest
cumulative difference score in a sliding window, since these tend to coincide with
motion/artifact-rich regions (blink, mouth movement, blending seam flicker).

Falls back to uniform sampling if the video is too short, corrupt, or has
near-zero motion throughout (e.g. a static face with negligible inter-frame diff).
"""
import cv2
import numpy as np


def _read_all_frames_grayscale_downsampled(video_path: str, downsample_to=(160, 120)):
    """Reads every frame at low res in grayscale just for diff scoring - keeps this
    pass cheap since we don't need full quality until we know which frames to keep."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames_gray = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, downsample_to, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
    cap.release()
    return frames_gray


def compute_diff_scores(frames_gray: list) -> np.ndarray:
    """Per-frame score = abs difference to previous frame, summed over pixels.
    First frame gets the same score as the second (no previous frame to diff against)."""
    if len(frames_gray) < 2:
        return np.zeros(len(frames_gray))

    diffs = [0.0]  # placeholder for frame 0
    for i in range(1, len(frames_gray)):
        diff = np.abs(frames_gray[i].astype(np.int16) - frames_gray[i - 1].astype(np.int16))
        diffs.append(float(diff.sum()))
    diffs[0] = diffs[1] if len(diffs) > 1 else 0.0
    return np.array(diffs)


def select_diff_based_frame_indices(num_total_frames: int, scores: np.ndarray, num_frames: int) -> list:
    """
    Picks num_frames indices with highest diff score, but enforces a minimum spacing
    so we don't end up with 16 frames all clustered in one 2-second burst of motion -
    that would give poor temporal coverage of the whole ~12s clip.
    """
    if num_total_frames <= num_frames:
        return list(range(num_total_frames))

    min_spacing = max(1, num_total_frames // (num_frames * 2))
    sorted_idx = np.argsort(-scores)  # descending by score

    selected = []
    for idx in sorted_idx:
        if all(abs(idx - s) >= min_spacing for s in selected):
            selected.append(int(idx))
        if len(selected) == num_frames:
            break

    # if spacing constraint left us short (very short/low-motion video), top up with
    # whatever highest-scoring frames remain regardless of spacing
    if len(selected) < num_frames:
        for idx in sorted_idx:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == num_frames:
                break

    return sorted(selected)


def select_uniform_frame_indices(num_total_frames: int, num_frames: int) -> list:
    if num_total_frames <= num_frames:
        return list(range(num_total_frames))
    return list(np.linspace(0, num_total_frames - 1, num_frames, dtype=int))


def sample_frame_indices(video_path: str, num_frames: int = 16, method: str = "diff_based") -> list:
    """
    Main entry point. Returns a sorted list of frame indices to extract at full
    resolution later. Falls back to uniform sampling on any failure.
    """
    try:
        frames_gray = _read_all_frames_grayscale_downsampled(video_path)
        num_total = len(frames_gray)

        if num_total == 0:
            raise ValueError(f"Video {video_path} has zero readable frames.")

        if method != "diff_based":
            return select_uniform_frame_indices(num_total, num_frames)

        scores = compute_diff_scores(frames_gray)

        # near-static video guard: if max diff score is negligible, diff-based
        # selection won't be meaningful - fall back to uniform for even coverage
        if scores.max() < 1.0:
            return select_uniform_frame_indices(num_total, num_frames)

        return select_diff_based_frame_indices(num_total, scores, num_frames)

    except Exception as e:
        print(f"WARNING: diff-based sampling failed for {video_path} ({e}); "
              f"falling back to uniform sampling.")
        cap = cv2.VideoCapture(video_path)
        num_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if num_total <= 0:
            num_total = 300  # last-resort assumption matching your ~300 frame average
        return select_uniform_frame_indices(num_total, num_frames)


def extract_frames_at_indices(video_path: str, indices: list, resize_to=(224, 224)) -> np.ndarray:
    """Extracts specific frame indices at full resolution, resized, as RGB.
    Returns array of shape (num_frames, H, W, 3), dtype uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    indices_set = set(indices)
    max_idx = max(indices) if indices else -1
    frames = {}

    frame_idx = 0
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_set:
            resized = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames[frame_idx] = rgb
        frame_idx += 1
    cap.release()

    # Some requested indices might exceed actual frame count if video was shorter
    # than expected - pad by repeating the last successfully read frame.
    ordered = []
    last_good = None
    for idx in indices:
        if idx in frames:
            last_good = frames[idx]
            ordered.append(frames[idx])
        elif last_good is not None:
            ordered.append(last_good)
        else:
            ordered.append(np.zeros((resize_to[1], resize_to[0], 3), dtype=np.uint8))

    return np.stack(ordered, axis=0)
