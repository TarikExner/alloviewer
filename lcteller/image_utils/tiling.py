from __future__ import annotations
from typing import Tuple, List
import numpy as np


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

class _UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def _make(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        # lazy create
        if x not in self.parent:
            self._make(x)
        # path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def reconstruct_from_tiles_numpy(
    tiles: np.ndarray,
    orig_hw: Tuple[int, int],
    tile_size: int = 512,
    overlap: int = 64,
    mode: str = "image",
    threshold: float = 0.5,
    connectivity: int = 1,  # kept for API, not used in new instance mode
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
          - C >= 1: probability map(s) (mode="probability_map")
    orig_hw : (int, int)
        Original (H, W) of the image before tiling.
    tile_size : int
        Spatial size of each tile (must match tiling).
    overlap : int
        Overlap used during tiling (must match tiling).
    mode : {"image", "mask", "instance", "probability_map"}
        - "image": numeric data (float / uint8), averaged in overlaps,
                   result cast back to tiles.dtype.
        - "mask":  boolean or 0/1 mask, merged with averaging+threshold.
        - "instance": per-tile instance label map; labels across tiles
                      are merged using overlap-based matching and a
                      union-find, then a global label map is built.
        - "probability_map": UNet-style probability map(s), averaged
                      in overlaps, returned as float32.
    threshold : float
        Threshold used in "mask" mode after averaging.
    connectivity : int
        Kept for compatibility; not used in the new instance mode.

    Returns
    -------
    out : np.ndarray
        Reconstructed array of shape [C, H, W].
        For "instance" mode, C = 1 and labels are global instance IDs.
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
        # Case 2: per-tile instance labels that may differ across tiles.
        # We merge labels across tiles using overlaps and a union-find.

        assert C == 1, "Instance mode expects tiles with C = 1"

        # First, collect tile positions (in the same order as tiles)
        tile_info = []  # (idx, y0, y1, x0, x1)
        idx = 0
        for (y0, y1, x0, x1) in iter_sliding_windows(H, W, tile_size, overlap):
            if idx >= N:
                raise RuntimeError("Not enough tiles for given H/W/tile_size/overlap.")
            tile_info.append((idx, y0, y1, x0, x1))
            idx += 1
        if idx != N:
            raise RuntimeError(
                f"Number of tiles ({N}) does not match tiling scheme ({idx})."
            )

        uf = _UnionFind()

        # Helper: union labels that overlap for a pair of tiles
        def _merge_tile_pair(i: int, j: int):
            idx_i, y0_i, y1_i, x0_i, x1_i = tile_info[i]
            idx_j, y0_j, y1_j, x0_j, x1_j = tile_info[j]

            # Check if bounding boxes overlap
            oy0 = max(y0_i, y0_j)
            oy1 = min(y1_i, y1_j)
            ox0 = max(x0_i, x0_j)
            ox1 = min(x1_i, x1_j)
            if oy1 <= oy0 or ox1 <= ox0:
                return  # no overlap

            # Local coords inside each tile
            li_y0 = oy0 - y0_i
            li_y1 = oy1 - y0_i
            li_x0 = ox0 - x0_i
            li_x1 = ox1 - x0_i

            lj_y0 = oy0 - y0_j
            lj_y1 = oy1 - y0_j
            lj_x0 = ox0 - x0_j
            lj_x1 = ox1 - x0_j

            tile_i = tiles[idx_i, 0]
            tile_j = tiles[idx_j, 0]

            patch_i = tile_i[li_y0:li_y1, li_x0:li_x1]
            patch_j = tile_j[lj_y0:lj_y1, lj_x0:lj_x1]

            both = (patch_i > 0) & (patch_j > 0)
            if not np.any(both):
                return

            li_vals = patch_i[both].astype(np.int64, copy=False)
            lj_vals = patch_j[both].astype(np.int64, copy=False)

            pairs = np.stack([li_vals, lj_vals], axis=1)
            # unique pairs (l_i, l_j) that share at least one pixel
            unique_pairs = np.unique(pairs, axis=0)

            for l_i, l_j in unique_pairs:
                if l_i == 0 or l_j == 0:
                    continue
                node_i = (idx_i, int(l_i))
                node_j = (idx_j, int(l_j))
                uf.union(node_i, node_j)

        # Build unions across overlapping tiles
        # Tiles are ordered by y, then x. We can break inner loop once y no longer overlaps.
        for i in range(N):
            idx_i, y0_i, y1_i, x0_i, x1_i = tile_info[i]
            for j in range(i + 1, N):
                idx_j, y0_j, y1_j, x0_j, x1_j = tile_info[j]
                if y0_j >= y1_i:
                    # further tiles will have even larger y0, no vertical overlap
                    break
                _merge_tile_pair(i, j)

        # Now assign global IDs to each (tile, local_label)
        global_id_for = {}
        root_to_gid = {}
        next_gid = 1

        for idx_t in range(N):
            tile_lab = tiles[idx_t, 0]
            labels_in_tile = np.unique(tile_lab)
            labels_in_tile = labels_in_tile[labels_in_tile > 0]
            for _l in labels_in_tile:
                node = (idx_t, int(_l))
                root = uf.find(node)
                if root not in root_to_gid:
                    root_to_gid[root] = next_gid
                    next_gid += 1
                global_id_for[node] = root_to_gid[root]

        # Build the global label map
        global_map = np.zeros((H, W), dtype=np.int32)
        for idx_t, y0, y1, x0, x1 in tile_info:
            tile_lab = tiles[idx_t, 0]
            th = y1 - y0
            tw = x1 - x0
            tile_lab = tile_lab[:th, :tw]
            region = global_map[y0:y1, x0:x1]

            labels_in_tile = np.unique(tile_lab)
            labels_in_tile = labels_in_tile[labels_in_tile > 0]

            for _l in labels_in_tile:
                node = (idx_t, int(_l))
                g = global_id_for[node]
                mask = (tile_lab == _l)

                # Only write where region is still zero; if region already
                # has non-zero there, it should be from the same union-root.
                write_mask = mask & (region == 0)
                region[write_mask] = g

        return global_map[np.newaxis, ...]  # [1, H, W]

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

    elif mode == "probability_map":
        # UNet-style probability maps; keep float and average.
        acc = np.zeros((C, H, W), dtype=np.float32)
        acc_w = np.zeros((1, H, W), dtype=np.float32)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            acc[:, y0:y0 + th, x0:x0 + tw] += patch
            acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        acc_w[acc_w == 0] = 1.0
        out = acc / acc_w
        return out.astype(np.float32)

    else:  # mode == "image"
        # Generic numeric images (e.g. RGB).
        acc = np.zeros((C, H, W), dtype=np.float32)
        acc_w = np.zeros((1, H, W), dtype=np.float32)

        for idx, y0, x0, th, tw, patch in _iter_tiles():
            acc[:, y0:y0 + th, x0:x0 + tw] += patch
            acc_w[:, y0:y0 + th, x0:x0 + tw] += 1.0

        acc_w[acc_w == 0] = 1.0
        out = acc / acc_w
        return out.astype(tiles.dtype)

def tile_images(imgs: List[np.ndarray]) -> List[np.ndarray]:
    return [
        tile_image_numpy(img)
        for img in imgs
    ]
