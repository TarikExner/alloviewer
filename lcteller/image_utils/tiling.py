from __future__ import annotations
from typing import Callable, Optional, Tuple, Iterable
import numpy as np
from skimage.measure import label as sklabel


def iter_sliding_windows(H: int, W: int, tile: int, overlap: int):
    stride = tile - overlap
    assert stride > 0, "overlap must be smaller than tile"

    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)

    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)

    for y0 in ys:
        y1 = y0 + tile
        for x0 in xs:
            x1 = x0 + tile
            yield (y0, y1, x0, x1)

def tile_image_numpy(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Split a [C, H, W] numpy image into overlapping tiles of shape [N, C, tile_size, tile_size].

    Parameters
    ----------
    image : np.ndarray
        Array of shape [C, H, W].
    tile_size : int
        Spatial size of each tile (default 512).
    overlap : int
        Overlap in pixels between tiles (default 64).
    pad_value : float
        Value used to pad tiles at the borders if H or W < tile_size.

    Returns
    -------
    tiles : np.ndarray
        Array of tiles with shape [N, C, tile_size, tile_size].
    orig_hw : (int, int)
        Original (H, W) so you can reconstruct later.
    """
    assert image.ndim == 3, "image must be [C, H, W]"
    C, H, W = image.shape

    tiles = []
    for (y0, y1, x0, x1) in iter_sliding_windows(H, W, tile_size, overlap):
        crop = image[:, y0:y1, x0:x1]         # [C, th, tw]
        th, tw = crop.shape[1], crop.shape[2]

        if th < tile_size or tw < tile_size:
            pad_bottom = tile_size - th
            pad_right = tile_size - tw
            crop = np.pad(
                crop,
                ((0, 0), (0, pad_bottom), (0, pad_right)),
                mode="constant",
                constant_values=pad_value,
            )

        tiles.append(crop)

    if not tiles:
        raise RuntimeError("No tiles produced. Check tile_size / overlap / image size.")

    tiles_arr = np.stack(tiles, axis=0)       # [N, C, tile_size, tile_size]
    return tiles_arr


def reconstruct_from_tiles_numpy(
    tiles: np.ndarray,
    orig_hw: Tuple[int, int],
    tile_size: int = 512,
    overlap: int = 64,
    mode: str = "image",
    threshold: float = 0.5,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Reconstruct a full image from tiles.

    Parameters
    ----------
    tiles : np.ndarray
        Array of shape [N, C, tile_size, tile_size].
        Works for:
          - C = 3: RGB image (mode="image")
          - C = 1: mask (mode="mask")
          - C = 1: instance labels (mode="instance")
    orig_hw : (int, int)
        Original (H, W) of the image before tiling.
    tile_size : int
        Spatial size of each tile (must match tiling).
    overlap : int
        Overlap used during tiling (must match tiling).
    mode : {"image", "mask", "instance"}
        - "image": numeric data (float / uint8), averaged in overlaps.
        - "mask":  boolean or 0/1 mask, merged and thresholded.
        - "instance": instance label map; tiles > 0 are merged to a
                      global binary mask, then relabeled with skimage.measure.label.
    threshold : float
        Threshold used in "mask" mode after averaging.
    connectivity : int
        Connectivity for skimage.measure.label in "instance" mode.

    Returns
    -------
    out : np.ndarray
        Reconstructed array of shape [C, H, W].
        For "instance" mode, C = 1 and labels are relabeled globally.
    """
    assert tiles.ndim == 4, "tiles must be [N, C, tile_size, tile_size]"
    N, C, tH, tW = tiles.shape
    assert tH == tile_size and tW == tile_size, "tile_size mismatch"

    H, W = orig_hw

    # helper: iterate tiles in the same order as tiling
    def _iter_tiles():
        idx = 0
        for (y0, y1, x0, x1) in iter_sliding_windows(H, W, tile_size, overlap):
            if idx >= N:
                raise RuntimeError("Not enough tiles for given H/W/tile_size/overlap.")
            th = min(tile_size, H - y0)
            tw = min(tile_size, W - x0)
            patch = tiles[idx, :, :th, :tw]  # [C, th, tw]
            yield idx, y0, x0, th, tw, patch
            idx += 1
        if idx != N:
            raise RuntimeError(
                f"Number of tiles ({N}) does not match tiling scheme ({idx})."
            )

    if mode == "instance":
        # We ignore the original instance IDs and just reconstruct a binary mask,
        # then relabel globally so IDs are consistent across the whole image.
        assert C == 1, "Instance mode expects tiles with C = 1"
        mask = np.zeros((H, W), dtype=bool)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            mask[y0:y0 + th, x0:x0 + tw] |= (patch[0] > 0)

        labeled = sklabel(mask, connectivity=connectivity)
        return labeled[np.newaxis, ...]        # [1, H, W]

    elif mode == "mask":
        # 0/1 or boolean masks. We average in overlaps and then threshold.
        acc = np.zeros((C, H, W), dtype=np.float32)
        acc_w = np.zeros((1, H, W), dtype=np.float32)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            acc[:, y0:y0 + th, x0:x0 + tw] += patch
            acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        acc_w[acc_w == 0] = 1.0
        out = acc / acc_w
        out_bin = out >= threshold

        if tiles.dtype == np.bool_:
            return out_bin.astype(bool)
        elif np.issubdtype(tiles.dtype, np.integer):
            return out_bin.astype(tiles.dtype)
        else:
            return out_bin.astype(np.float32)

    else:  # mode == "image"
        # Generic numeric images (e.g. RGB, probability maps).
        acc = np.zeros((C, H, W), dtype=np.float32)
        acc_w = np.zeros((1, H, W), dtype=np.float32)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            acc[:, y0:y0 + th, x0:x0 + tw] += patch
            acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        acc_w[acc_w == 0] = 1.0
        out = acc / acc_w
        return out.astype(tiles.dtype)

