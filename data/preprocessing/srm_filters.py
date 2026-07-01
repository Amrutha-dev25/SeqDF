"""
SRM (Spatial Rich Model) high-pass residual extraction.

These are the same 3 classic SRM kernels widely used in steganalysis and deepfake
forensics literature (e.g. used in Zhou et al. 2017 "Two-Stream Neural Networks for
Tampered Face Detection", and many subsequent forensics papers) to suppress low-frequency
image content and expose noise-level inconsistencies - exactly the kind of subtle
blending-seam artifact that should survive from your m2 (middle) manipulation step
even after m3's blend is applied on top.
"""
import numpy as np
import torch
import torch.nn.functional as F

# Three standard SRM high-pass kernels (5x5), normalized.
_SRM_KERNEL_1 = np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32) / 4.0

_SRM_KERNEL_2 = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
], dtype=np.float32) / 12.0

_SRM_KERNEL_3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.float32) / 2.0

SRM_KERNELS = np.stack([_SRM_KERNEL_1, _SRM_KERNEL_2, _SRM_KERNEL_3], axis=0)  # (3, 5, 5)


def _build_srm_conv_weight(num_filters: int = 3) -> torch.Tensor:
    """Builds a (num_filters, 3, 5, 5) conv weight applying each SRM kernel to each
    RGB channel independently (depthwise-style via grouping handled in apply fn)."""
    kernels = SRM_KERNELS[:num_filters]  # (F, 5, 5)
    weight = np.tile(kernels[:, None, :, :], (1, 3, 1, 1))  # (F, 3, 5, 5) - same kernel all channels
    return torch.from_numpy(weight)


_SRM_WEIGHT = _build_srm_conv_weight(num_filters=3)  # cached module-level


def extract_srm_residual(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Single-frame SRM extraction.
    Input: (H, W, 3) uint8 RGB frame.
    Output: (H, W, 3) float32 array - 3 SRM filter responses stacked as pseudo-RGB
    channels, normalized to roughly [-1, 1], suitable as direct input to a standard
    3-channel CNN backbone (EfficientNet-B4 etc) without architecture modification.
    """
    img = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,H,W)
    with torch.no_grad():
        out = F.conv2d(img, _SRM_WEIGHT, padding=2)  # (1, 3, H, W) - one channel per SRM kernel
    out = out.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

    # normalize per-channel to stabilize scale across videos/lighting conditions
    for c in range(out.shape[-1]):
        ch = out[..., c]
        std = ch.std()
        if std > 1e-6:
            out[..., c] = (ch - ch.mean()) / (std + 1e-6)
        else:
            out[..., c] = 0.0

    return out.astype(np.float32)


def extract_srm_residual_batch(frames_rgb: np.ndarray) -> np.ndarray:
    """
    Batched version for a full sampled clip.
    Input: (T, H, W, 3) uint8.
    Output: (T, H, W, 3) float32 SRM residual maps.
    """
    img = torch.from_numpy(frames_rgb).float().permute(0, 3, 1, 2) / 255.0  # (T,3,H,W)
    with torch.no_grad():
        out = F.conv2d(img, _SRM_WEIGHT, padding=2)  # (T,3,H,W)
    out = out.permute(0, 2, 3, 1).numpy()  # (T,H,W,3)

    # normalize per-frame, per-channel
    for t in range(out.shape[0]):
        for c in range(out.shape[-1]):
            ch = out[t, ..., c]
            std = ch.std()
            if std > 1e-6:
                out[t, ..., c] = (ch - ch.mean()) / (std + 1e-6)
            else:
                out[t, ..., c] = 0.0

    return out.astype(np.float32)
