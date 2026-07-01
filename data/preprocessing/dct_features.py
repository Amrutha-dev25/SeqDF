"""
Block-DCT energy map extraction, the weakest but only available signal for m1
(first manipulation step) - re-encoding artifacts and deep frequency fingerprints
that survive (faintly) underneath two further rounds of blending.

For each non-overlapping 8x8 block (standard JPEG-style blocking, matching FF++'s
H.264 macroblock-adjacent structure), computes the 2D DCT and splits its energy
into low/mid/high frequency bands. The 3 band-energy values per block become 3
"pseudo-RGB" channels in a downsampled energy map, so this can again be fed into a
standard 3-channel CNN backbone without modification.
"""
import numpy as np
import cv2


def _get_zigzag_band_masks(block_size: int = 8, num_bands: int = 3) -> list:
    """
    Splits an 8x8 DCT coefficient block into num_bands frequency bands by
    Manhattan distance from the DC term (top-left), which approximates the
    standard zigzag low->high frequency ordering used in JPEG-style coding.
    Returns a list of boolean masks, each (block_size, block_size).
    """
    coords = np.indices((block_size, block_size)).sum(axis=0)  # i+j distance from DC
    max_dist = coords.max()
    band_edges = np.linspace(0, max_dist, num_bands + 1)

    masks = []
    for b in range(num_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        if b == num_bands - 1:
            mask = (coords >= lo) & (coords <= hi)
        else:
            mask = (coords >= lo) & (coords < hi)
        masks.append(mask)
    return masks


def extract_dct_energy_map(frame_rgb: np.ndarray, block_size: int = 8, num_bands: int = 3) -> np.ndarray:
    """
    Single-frame block-DCT energy extraction.
    Input: (H, W, 3) uint8 RGB frame.
    Output: (H//block_size, W//block_size, num_bands) float32 energy map.
    Operates on luminance (matches how H.264/JPEG-style compression artifacts
    actually manifest - chroma subsampling makes luma the more informative channel).
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    h_blocks, w_blocks = h // block_size, w // block_size
    gray = gray[:h_blocks * block_size, :w_blocks * block_size]  # crop to exact multiple

    band_masks = _get_zigzag_band_masks(block_size, num_bands)
    energy_map = np.zeros((h_blocks, w_blocks, num_bands), dtype=np.float32)

    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = gray[bi * block_size:(bi + 1) * block_size,
                          bj * block_size:(bj + 1) * block_size]
            dct_block = cv2.dct(block)
            for band_idx, mask in enumerate(band_masks):
                energy_map[bi, bj, band_idx] = np.sum(dct_block[mask] ** 2)

    # log-scale energy (DCT energy is heavy-tailed) then per-channel normalize
    energy_map = np.log1p(energy_map)
    for c in range(num_bands):
        ch = energy_map[..., c]
        std = ch.std()
        if std > 1e-6:
            energy_map[..., c] = (ch - ch.mean()) / (std + 1e-6)
        else:
            energy_map[..., c] = 0.0

    return energy_map.astype(np.float32)


def extract_dct_energy_map_resized(frame_rgb: np.ndarray, output_size=(224, 224),
                                     block_size: int = 8, num_bands: int = 3) -> np.ndarray:
    """
    Same as extract_dct_energy_map, but resizes the resulting low-res block-grid
    map up to output_size (e.g. 224x224) via nearest-neighbor, so it has the same
    spatial dims as the RGB/SRM streams and can share a standard CNN backbone
    input pipeline without separate handling.
    """
    energy_map = extract_dct_energy_map(frame_rgb, block_size, num_bands)
    resized = cv2.resize(energy_map, output_size, interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)


def extract_dct_energy_map_batch(frames_rgb: np.ndarray, output_size=(224, 224),
                                   block_size: int = 8, num_bands: int = 3) -> np.ndarray:
    """Batched version for a full sampled clip.
    Input: (T, H, W, 3) uint8. Output: (T, output_h, output_w, num_bands) float32."""
    maps = [
        extract_dct_energy_map_resized(frames_rgb[t], output_size, block_size, num_bands)
        for t in range(frames_rgb.shape[0])
    ]
    return np.stack(maps, axis=0)
