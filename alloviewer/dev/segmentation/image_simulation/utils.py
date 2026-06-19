import numpy as np
from typing import (
    Tuple,
    Union,
    Sequence,
    Mapping,
    Any,
    Dict,
    Optional,
    Callable
)
import inspect
import cv2

from .types import RNG

import numbers

def to_jsonable(x):
    """Convert common numeric / numpy types to plain Python so JSON dump works."""
    # simple numbers
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x

    # numpy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)

    # sequences (tuples/lists) of jsonable items
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return [to_jsonable(v) for v in x]

    # small numpy arrays (avoid dumping huge arrays by mistake)
    if isinstance(x, np.ndarray):
        # keep tiny shapes, otherwise store shape + dtype
        if x.size <= 64:
            return x.tolist()
        return {"__ndarray__": True, "shape": tuple(x.shape), "dtype": str(x.dtype)}

    # mappings
    if isinstance(x, Mapping):
        return {k: to_jsonable(v) for k, v in x.items()}

    # fallback to string
    return str(x)

def capture_params(func, locals_dict):
    """
    Return a dict of just the function's declared parameters
    with their current values, made JSON-safe.
    """
    sig = inspect.signature(func)
    out = {}
    for name in sig.parameters:
        if name in locals_dict:
            out[name] = to_jsonable(locals_dict[name])
    return out

def is_pair(x) -> bool:
    return isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, numbers.Number) for v in x)

def sample_number(
        rng: RNG,
        spec: Union[float, int, Tuple[float, float],
        Tuple[int, int]],
        integer: bool = False
) -> Union[float, int]:
    if is_pair(spec):
        lo, hi = spec  # inclusive-exclusive for floats; inclusive-inclusive for ints below
        if integer:
            return int(rng.integers(int(np.floor(lo)), int(np.ceil(hi)) + 1))
        else:
            return float(rng.uniform(float(lo), float(hi)))
    # scalar
    return int(spec) if integer else float(spec)

def sample_bool(rng: RNG, spec: Union[bool, float, Tuple[float, float]]) -> bool:
    """
    spec can be:
      - bool: returned directly
      - float p in [0,1]: Bernoulli(p)
      - (p_lo, p_hi): sample p ~ U[p_lo, p_hi], then Bernoulli(p)
    """
    if isinstance(spec, bool):
        return spec
    if is_pair(spec):
        p = float(rng.uniform(float(spec[0]), float(spec[1])))
        p = np.clip(p, 0.0, 1.0)
        return bool(rng.random() < p)
    # scalar prob
    p = float(spec)
    p = np.clip(p, 0.0, 1.0)
    return bool(rng.random() < p)

def round_to_multiple(x: float, m: int) -> int:
    return int(max(m, round(x / m) * m))

def choose_ratio(rng: RNG, ratios: Sequence[Tuple[int, int]], portrait_prob: float) -> float:
    num, den = ratios[int(rng.integers(0, len(ratios)))]
    r = num / den  # width / height
    if portrait_prob > 0 and rng.random() < portrait_prob:
        r = 1.0 / r  # swap to height / width
    return r

def apply_s_curve(img: np.ndarray, strength: float) -> np.ndarray:
    """
    strength in about [-0.25, 0.40]
    positive => stronger midtone contrast
    negative => flatter midtones
    """
    if abs(strength) < 1e-8:
        return img

    x = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    a = np.float32(1.0 + 8.0 * float(strength))

    y = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-a * (x - np.float32(0.5))).astype(np.float32)
    )

    y0 = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-a * (np.float32(0.0) - np.float32(0.5))).astype(np.float32)
    )
    y1 = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-a * (np.float32(1.0) - np.float32(0.5))).astype(np.float32)
    )

    y = (y - y0) / (y1 - y0 + np.float32(1e-8))
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def lift_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img

    img = img.astype(np.float32, copy=False)
    amount32 = np.float32(amount)

    w = (np.float32(1.0) - img) ** np.float32(2.0)
    out = img + amount32 * np.float32(0.35) * w

    return np.clip(out, 0.0, 1.0).astype(np.float32)

def compress_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    thr = np.float32(0.72)
    amount32 = np.float32(amount)
    out = img.astype(np.float32, copy=True)
    mask = out > thr
    if np.any(mask):
        x = (out[mask] - thr).astype(np.float32, copy=False)
        out[mask] = (
            thr
            + (np.float32(1.0) - np.exp(-x / (amount32 + np.float32(1e-6))).astype(np.float32))
            * (np.float32(1.0) - thr)
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def apply_channel_median_match(
    img: np.ndarray,
    target_device: str,
    quantile_band_cache: Dict[str, Any],
    strength: float = 0.5,
    per_channel_strength: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Shift each channel toward the target device median.

    Parameters
    ----------
    img : np.ndarray
        RGB image, HWC, float32 in [0,1]
    target_device : str
        "microscope", "iphone", or "googlepixel"
    quantile_band_cache : dict
        Cache from build_target_quantile_band_cache(...)
    strength : float
        Global median-match strength in [0,1]
    per_channel_strength : np.ndarray or None
        Optional shape (3,) multiplier for R/G/B.
        Example for microscope: np.array([0.3, 0.3, 1.0], dtype=np.float32)
        to hit blue harder than red/green.
    """
    if target_device not in quantile_band_cache["devices"]:
        return img

    device_ref = quantile_band_cache["devices"][target_device]
    q_center = device_ref["q_center"]   # [3, Q]

    # median of the target distribution
    q_probs = np.asarray(quantile_band_cache["q_probs"], dtype=np.float32)
    mid_idx = int(np.argmin(np.abs(q_probs - 0.5)))
    target_medians = q_center[:, mid_idx].astype(np.float32)

    current_medians = np.median(img.reshape(-1, 3), axis=0).astype(np.float32)

    delta = target_medians - current_medians

    if per_channel_strength is None:
        per_channel_strength = np.ones(3, dtype=np.float32)
    else:
        per_channel_strength = np.asarray(per_channel_strength, dtype=np.float32)

    strength32 = np.float32(strength)
    shift = (strength32 * per_channel_strength * delta).astype(np.float32)
    out = img.astype(np.float32, copy=False) + shift.reshape(1, 1, 3)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_global_blur(
    img: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply clean global Gaussian blur on float RGB data.
    """
    if sigma <= 0:
        return img

    img = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def apply_photon_noise(
    img: np.ndarray,
    rng: RNG,
    photon_level: float,
) -> np.ndarray:
    """
    Apply signal-dependent Poisson shot noise.

    photon_level acts like the approximate maximum photon count at img == 1.
    Higher values mean less visible photon noise.
    """
    if photon_level <= 0:
        return img

    photon_level = float(photon_level)
    counts = np.clip(img, 0.0, 1.0) * photon_level

    img = rng.poisson(counts).astype(np.float32) / max(1.0, photon_level)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def apply_read_noise(
    img: np.ndarray,
    rng: RNG,
    read_noise: float,
) -> np.ndarray:
    """
    Apply additive Gaussian read noise on float RGB data.
    """
    if read_noise <= 0:
        return img

    noise = rng.normal(
        loc=0.0,
        scale=float(read_noise),
        size=img.shape,
    ).astype(np.float32)

    img = img + noise

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def sample_channel_values(
    rng: RNG,
    ranges,
    dtype=np.float32,
) -> np.ndarray:
    """
    Sample one value per channel from ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)).
    """
    vals = [
        float(rng.uniform(float(lo), float(hi)))
        for lo, hi in ranges
    ]
    return np.asarray(vals, dtype=dtype)


def interpolate_cell_diameter_bounds(
    short_side: float,
    anchors: Sequence[tuple[float, float, float]],
) -> tuple[float, float]:
    """Interpolate realistic core-diameter limits from image-size anchors.

    Each anchor is ``(image_short_side_px, min_core_diameter_px,
    max_core_diameter_px)``. Values outside the measured image-size interval
    are clamped to the nearest anchor.
    """
    values = np.asarray(anchors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 3:
        raise ValueError(
            "anchors must contain at least one "
            "(short_side, min_diameter, max_diameter) row"
        )

    order = np.argsort(values[:, 0])
    values = values[order]

    image_sizes = values[:, 0]
    lower_bounds = values[:, 1]
    upper_bounds = values[:, 2]

    if np.any(image_sizes <= 0):
        raise ValueError("anchor short-side values must be positive")
    if np.any(lower_bounds <= 0) or np.any(upper_bounds <= 0):
        raise ValueError("diameter bounds must be positive")
    if np.any(lower_bounds > upper_bounds):
        raise ValueError("each minimum diameter must be <= its maximum")

    lower = float(
        np.interp(
            float(short_side),
            image_sizes,
            lower_bounds,
            left=lower_bounds[0],
            right=lower_bounds[-1],
        )
    )
    upper = float(
        np.interp(
            float(short_side),
            image_sizes,
            upper_bounds,
            left=upper_bounds[0],
            right=upper_bounds[-1],
        )
    )
    return lower, upper


def sample_calibrated_cell_diameters(
    *,
    rng: np.random.Generator,
    n_cells: int,
    short_side: float,
    anchors: Sequence[tuple[float, float, float]],
    center_margin_frac: float = 0.20,
    cell_sigma_frac: float = 0.18,
    min_sigma_px: float = 0.25,
    large_cell_frac: float = 0.0,
) -> tuple[np.ndarray, float, tuple[float, float], np.ndarray]:
    """Sample one realistic diameter distribution for a simulated image.

    A typical diameter is sampled once per image. Individual cells are then
    sampled around that value and clipped to image-size-dependent limits.
    Cells selected by ``large_cell_frac`` are sampled from the upper part of
    the same valid interval rather than multiplied beyond the measured range.
    """
    n_cells = int(n_cells)
    if n_cells < 0:
        raise ValueError("n_cells must be non-negative")

    lower, upper = interpolate_cell_diameter_bounds(short_side, anchors)
    span = max(0.0, upper - lower)

    margin_frac = float(np.clip(center_margin_frac, 0.0, 0.49))
    if span <= 1e-8:
        image_center = lower
    else:
        margin = margin_frac * span
        image_center = float(rng.uniform(lower + margin, upper - margin))

    sigma = max(float(min_sigma_px), float(cell_sigma_frac) * span)
    diameters = rng.normal(
        loc=image_center,
        scale=sigma,
        size=n_cells,
    ).astype(np.float32)

    large_fraction = float(np.clip(large_cell_frac, 0.0, 1.0))
    is_large = rng.random(n_cells) < large_fraction
    n_large = int(is_large.sum())

    if n_large > 0:
        upper_start = max(image_center, lower + 0.65 * span)
        if upper <= upper_start + 1e-8:
            diameters[is_large] = np.float32(upper)
        else:
            diameters[is_large] = rng.uniform(
                upper_start,
                upper,
                size=n_large,
            ).astype(np.float32)

    diameters = np.clip(diameters, lower, upper).astype(np.float32)
    return diameters, image_center, (lower, upper), is_large


def assign_cluster_ids(
    *,
    rng: np.random.Generator,
    n_cells: int,
    clustered_fraction: float,
    cluster_size_range: tuple[int, int],
) -> np.ndarray:
    """Assign a subset of cells to compact clusters in O(n_cells) time.

    ``-1`` marks an isolated cell. Non-negative values are cluster IDs.
    """
    n_cells = int(n_cells)
    cluster_ids = np.full(n_cells, -1, dtype=np.int32)
    if n_cells <= 1:
        return cluster_ids

    size_lo = max(2, int(cluster_size_range[0]))
    size_hi = max(size_lo, int(cluster_size_range[1]))

    target = int(round(np.clip(clustered_fraction, 0.0, 1.0) * n_cells))
    target = min(target, n_cells)
    if target < size_lo:
        return cluster_ids

    selected = rng.permutation(n_cells)[:target]
    cursor = 0
    cluster_id = 0

    while target - cursor >= size_lo:
        remaining = target - cursor
        size = min(int(rng.integers(size_lo, size_hi + 1)), remaining)

        # Avoid leaving an invalid one-cell remainder.
        if 0 < remaining - size < size_lo:
            size = remaining

        members = selected[cursor:cursor + size]
        cluster_ids[members] = cluster_id
        cursor += size
        cluster_id += 1

    return cluster_ids


def place_clustered_centers(
    *,
    rng: np.random.Generator,
    cluster_ids: np.ndarray,
    r_px: np.ndarray,
    H: int,
    W: int,
    cy: float,
    cx: float,
    R: float,
    wall_margin_px: float,
    rim_band: float,
    rim_min_sep_px: float,
    sample_radius: Callable[[], float],
    sample_theta: Callable[[float], float],
    contact_factor_range: tuple[float, float] = (0.90, 1.05),
    core_min_sep_factor: float = 0.84,
    chain_probability: float = 0.55,
    angle_jitter: float = 0.80,
    packed_probability: float = 0.55,
    packed_size_bias_range: tuple[int, int] = (3, 15),
    packed_contact_factor_range: tuple[float, float] = (0.88, 1.02),
    packed_candidate_count: int = 8,
    packed_contact_bonus: float = 1.50,
    packed_region_join_probability: float = 0.30,
    packed_region_contact_factor_range: tuple[float, float] = (0.95, 1.12),
    seed_tries: int = 120,
    member_tries: int = 32,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
]:
    """Place isolated cells plus explicit lengthy and packed clusters.

    Mode codes returned in ``cluster_modes``:
      - ``-1``: isolated or placement fallback
      - ``0``: lengthy cluster
      - ``1``: packed cluster

    Packed clusters use a small candidate set for each new cell. Valid
    candidates are scored by distance to the current cluster centre and by
    the number of nearby neighbours. This produces compact, irregular groups
    without a fixed circular boundary.

    With ``packed_region_join_probability > 0``, a packed cluster may seed
    next to an earlier packed cluster. Both clusters then share a region ID,
    allowing several labelled clusters to appear as one larger packed area.

    Work is local to each cluster. No full-image pairwise distance matrix is
    created; the later KD-tree pass still resolves global conflicts.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32).copy()
    r_px = np.asarray(r_px, dtype=np.float32)
    n_cells = int(cluster_ids.size)

    if r_px.shape != (n_cells,):
        raise ValueError("r_px and cluster_ids must have the same length")

    centers = np.full((n_cells, 2), np.nan, dtype=np.float32)
    cluster_modes = np.full(n_cells, -1, dtype=np.int8)
    cluster_region_ids = np.full(n_cells, -1, dtype=np.int32)
    cluster_edges: list[tuple[int, int]] = []

    H = int(H)
    W = int(W)
    cy = float(cy)
    cx = float(cx)
    R = float(R)
    wall_margin_px = float(wall_margin_px)
    rim_inner = (1.0 - float(rim_band)) * R
    rim_min_sep_sq = float(rim_min_sep_px) ** 2
    accepted_rim_seeds: list[tuple[float, float]] = []

    lengthy_contact_lo = float(min(contact_factor_range))
    lengthy_contact_hi = float(max(contact_factor_range))
    packed_contact_lo = float(min(packed_contact_factor_range))
    packed_contact_hi = float(max(packed_contact_factor_range))
    region_contact_lo = float(min(packed_region_contact_factor_range))
    region_contact_hi = float(max(packed_region_contact_factor_range))

    core_min_sep_factor = max(0.0, float(core_min_sep_factor))
    chain_probability = float(np.clip(chain_probability, 0.0, 1.0))
    angle_jitter = max(0.0, float(angle_jitter))
    packed_probability = float(np.clip(packed_probability, 0.0, 1.0))
    packed_candidate_count = max(1, int(packed_candidate_count))
    packed_contact_bonus = max(0.0, float(packed_contact_bonus))
    packed_region_join_probability = float(
        np.clip(packed_region_join_probability, 0.0, 1.0)
    )
    seed_tries = max(1, int(seed_tries))
    member_tries = max(1, int(member_tries))

    bias_lo = max(2, int(min(packed_size_bias_range)))
    bias_hi = max(bias_lo + 1, int(max(packed_size_bias_range)))

    # Each entry stores all cell indices currently assigned to that packed
    # region. Several separate cluster IDs may share one region.
    packed_regions: dict[int, list[int]] = {}
    next_region_id = 0

    def valid_center(y: float, x: float, index: int) -> bool:
        radius = float(r_px[index])
        if not (
            radius + 1.0 <= y < H - radius - 1.0
            and radius + 1.0 <= x < W - radius - 1.0
        ):
            return False

        max_center_radius = max(0.0, R - wall_margin_px - radius)
        return np.hypot(y - cy, x - cx) <= max_center_radius

    def independent_center(index: int) -> np.ndarray:
        for _ in range(seed_tries):
            radial_position = float(sample_radius())
            angle = float(sample_theta(radial_position))
            y = cy + radial_position * np.sin(angle)
            x = cx + radial_position * np.cos(angle)

            if not valid_center(y, x, index):
                continue

            is_rim_seed = radial_position >= rim_inner
            if is_rim_seed and rim_min_sep_px > 0 and accepted_rim_seeds:
                recent = accepted_rim_seeds[-200:]
                too_close = any(
                    (py - y) ** 2 + (px - x) ** 2 < rim_min_sep_sq
                    for py, px in recent
                )
                if too_close:
                    continue

            if is_rim_seed:
                accepted_rim_seeds.append((y, x))
            return np.array([y, x], dtype=np.float32)

        max_center_radius = max(0.0, R - wall_margin_px - float(r_px[index]))
        for _ in range(256):
            radial_position = max_center_radius * np.sqrt(rng.random())
            angle = rng.uniform(-np.pi, np.pi)
            y = cy + radial_position * np.sin(angle)
            x = cx + radial_position * np.cos(angle)
            if valid_center(y, x, index):
                return np.array([y, x], dtype=np.float32)

        radius = float(r_px[index])
        y = float(np.clip(cy, radius + 1.0, H - radius - 2.0))
        x = float(np.clip(cx, radius + 1.0, W - radius - 2.0))
        return np.array([y, x], dtype=np.float32)

    def clear_of_indices(
        y: float,
        x: float,
        index: int,
        other_indices: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        if other_indices.size == 0:
            return True, np.empty(0, dtype=np.float32)

        other_centers = centers[other_indices]
        finite = np.isfinite(other_centers).all(axis=1)
        if not np.any(finite):
            return True, np.empty(0, dtype=np.float32)

        other_indices = other_indices[finite]
        other_centers = other_centers[finite]
        dy = other_centers[:, 0] - np.float32(y)
        dx = other_centers[:, 1] - np.float32(x)
        distances = np.sqrt(dy * dy + dx * dx).astype(np.float32)
        minimum = np.float32(core_min_sep_factor) * (
            r_px[other_indices] + r_px[index]
        )
        return bool(np.all(distances >= minimum)), distances

    def packed_mode_probability(cluster_size: int) -> float:
        size_fraction = float(
            np.clip(
                (float(cluster_size) - bias_lo) / float(bias_hi - bias_lo),
                0.0,
                1.0,
            )
        )
        # Small groups are mostly lengthy. Large groups approach the sampled
        # packed_probability.
        return float(packed_probability * (0.25 + 0.75 * size_fraction))

    def seed_next_to_packed_region(index: int) -> tuple[np.ndarray | None, int]:
        if not packed_regions:
            return None, -1

        region_ids = np.fromiter(packed_regions.keys(), dtype=np.int32)
        region_id = int(rng.choice(region_ids))
        region_members = np.asarray(packed_regions[region_id], dtype=np.int32)
        if region_members.size == 0:
            return None, -1

        all_placed = np.flatnonzero(np.isfinite(centers).all(axis=1))
        best_position: np.ndarray | None = None
        best_score = np.inf

        # A few scored seed candidates are enough because this runs once per
        # packed cluster, not once per cell.
        for _ in range(min(seed_tries, max(8, 2 * packed_candidate_count))):
            parent = int(rng.choice(region_members))
            angle = float(rng.uniform(-np.pi, np.pi))
            target_distance = float(
                rng.uniform(region_contact_lo, region_contact_hi)
                * (r_px[parent] + r_px[index])
            )
            y = float(centers[parent, 0] + target_distance * np.sin(angle))
            x = float(centers[parent, 1] + target_distance * np.cos(angle))

            if not valid_center(y, x, index):
                continue

            clear, distances = clear_of_indices(y, x, index, all_placed)
            if not clear:
                continue

            placed_indices = all_placed
            pair_sum = r_px[placed_indices] + r_px[index]
            close_neighbors = int(np.sum(distances <= 1.15 * pair_sum))

            region_center = np.mean(centers[region_members], axis=0)
            scale = max(1.0, 2.0 * float(np.median(r_px[region_members])))
            centroid_distance = float(
                np.hypot(y - region_center[0], x - region_center[1]) / scale
            )
            score = centroid_distance - packed_contact_bonus * close_neighbors

            if score < best_score:
                best_score = score
                best_position = np.array([y, x], dtype=np.float32)

        return best_position, region_id

    valid_cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])

    for cluster_id in valid_cluster_ids:
        members = np.flatnonzero(cluster_ids == cluster_id)
        if members.size < 2:
            cluster_ids[members] = -1
            continue

        mode = 1 if rng.random() < packed_mode_probability(int(members.size)) else 0
        cluster_modes[members] = np.int8(mode)

        first = int(members[0])
        region_id = -1

        if (
            mode == 1
            and packed_regions
            and rng.random() < packed_region_join_probability
        ):
            joined_seed, joined_region_id = seed_next_to_packed_region(first)
            if joined_seed is not None:
                centers[first] = joined_seed
                region_id = joined_region_id

        if not np.isfinite(centers[first]).all():
            centers[first] = independent_center(first)
            if mode == 1:
                region_id = next_region_id
                next_region_id += 1

        if mode == 1:
            cluster_region_ids[members] = np.int32(region_id)
            packed_regions.setdefault(region_id, []).append(first)

        heading = float(rng.uniform(-np.pi, np.pi))

        for local_index in range(1, members.size):
            index = int(members[local_index])
            previous = members[:local_index]
            accepted = False
            chosen_parent = -1

            if mode == 0:
                # Lengthy mode: continue from the latest member most of the
                # time, with occasional branches from earlier members.
                for _ in range(member_tries):
                    previous_index = int(members[local_index - 1])
                    if rng.random() < chain_probability:
                        parent = previous_index
                        heading += float(rng.normal(0.0, angle_jitter))
                        angle = heading
                    else:
                        parent = int(rng.choice(previous))
                        angle = float(rng.uniform(-np.pi, np.pi))

                    target_distance = float(
                        rng.uniform(lengthy_contact_lo, lengthy_contact_hi)
                        * (r_px[parent] + r_px[index])
                    )
                    y = float(centers[parent, 0] + target_distance * np.sin(angle))
                    x = float(centers[parent, 1] + target_distance * np.cos(angle))

                    if not valid_center(y, x, index):
                        angle = float(
                            np.arctan2(
                                cy - float(centers[parent, 0]),
                                cx - float(centers[parent, 1]),
                            )
                            + rng.normal(0.0, angle_jitter)
                        )
                        y = float(centers[parent, 0] + target_distance * np.sin(angle))
                        x = float(centers[parent, 1] + target_distance * np.cos(angle))

                    if not valid_center(y, x, index):
                        continue

                    clear, _ = clear_of_indices(y, x, index, previous)
                    if not clear:
                        continue

                    centers[index] = (y, x)
                    chosen_parent = parent
                    accepted = True
                    break

            else:
                # Packed mode: score a small set of candidates and choose the
                # one that stays close to the cluster centre while touching
                # several earlier members.
                cluster_center = np.mean(centers[previous], axis=0)
                scale = max(1.0, 2.0 * float(np.median(r_px[previous])))
                best_position: np.ndarray | None = None
                best_parent = -1
                best_score = np.inf
                n_candidates = packed_candidate_count

                for _ in range(n_candidates):
                    parent = int(rng.choice(previous))
                    angle = float(rng.uniform(-np.pi, np.pi))
                    target_distance = float(
                        rng.uniform(packed_contact_lo, packed_contact_hi)
                        * (r_px[parent] + r_px[index])
                    )
                    y = float(centers[parent, 0] + target_distance * np.sin(angle))
                    x = float(centers[parent, 1] + target_distance * np.cos(angle))

                    if not valid_center(y, x, index):
                        continue

                    clear, distances = clear_of_indices(y, x, index, previous)
                    if not clear:
                        continue

                    pair_sum = r_px[previous] + r_px[index]
                    close_neighbors = int(np.sum(distances <= 1.15 * pair_sum))
                    centroid_distance = float(
                        np.hypot(y - cluster_center[0], x - cluster_center[1])
                        / scale
                    )
                    score = centroid_distance - packed_contact_bonus * close_neighbors

                    if score < best_score:
                        best_score = score
                        best_position = np.array([y, x], dtype=np.float32)
                        best_parent = parent

                if best_position is not None:
                    centers[index] = best_position
                    chosen_parent = best_parent
                    accepted = True

            if accepted:
                cluster_edges.append((chosen_parent, index))
                if mode == 1:
                    packed_regions[region_id].append(index)
                continue

            # A failed member becomes isolated rather than forcing an invalid
            # overlap into the target map.
            centers[index] = independent_center(index)
            cluster_ids[index] = -1
            cluster_modes[index] = -1
            cluster_region_ids[index] = -1

    # Remove cluster labels that ended with fewer than two valid members.
    remaining_cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])
    for cluster_id in remaining_cluster_ids:
        members = np.flatnonzero(cluster_ids == cluster_id)
        if members.size < 2:
            cluster_ids[members] = -1
            cluster_modes[members] = -1
            cluster_region_ids[members] = -1

    isolated = np.flatnonzero(cluster_ids < 0)
    for index in isolated:
        if not np.isfinite(centers[index]).all():
            centers[index] = independent_center(int(index))

    return (
        centers,
        cluster_ids,
        cluster_edges,
        cluster_modes,
        cluster_region_ids,
    )



def make_wormy_dirt_patch(
    rng,
    base_size,
    dirt_sigma,
    dirt_alpha,
    base_orange,
    base_green,
):
    """
    Create one worm-like debris patch.

    Parameters
    ----------
    rng : np.random.Generator
    base_size : int
        Main size control for the debris object.
    dirt_sigma : tuple[float, float]
        Blur sigma range.
    dirt_alpha : tuple[float, float]
        Opacity range.
    base_orange : np.ndarray shape (3,)
    base_green : np.ndarray shape (3,)

    Returns
    -------
    alpha_map : np.ndarray, float32, shape (H, W)
    color : np.ndarray, float32, shape (3,)
    """
    base_size = int(max(1, base_size))

    # Thickness stays modest so it looks like debris, not a cell blob.
    thickness = max(1, int(round(rng.uniform(0.20, 0.45) * base_size)))

    # Keep a short random worm.
    n_segments = int(rng.integers(2, 5))
    step_len = float(rng.uniform(0.45, 0.90) * base_size)

    # Smaller patch than before.
    max_path_length = n_segments * step_len
    patch_radius = int(np.ceil(0.65 * max_path_length + 1.5 * base_size))
    patch_radius = max(patch_radius, 2 * base_size)
    patch_size = 2 * patch_radius + 1

    canvas = np.zeros((patch_size, patch_size), dtype=np.uint8)

    cy = patch_radius
    cx = patch_radius

    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    y = float(cy)
    x = float(cx)

    pts = [(int(round(x)), int(round(y)))]

    for _ in range(n_segments):
        angle += float(rng.normal(0.0, 0.55))
        seg_len = float(step_len * rng.uniform(0.7, 1.3))

        y_new = y + seg_len * np.sin(angle)
        x_new = x + seg_len * np.cos(angle)

        y_new = float(np.clip(y_new, 1, patch_size - 2))
        x_new = float(np.clip(x_new, 1, patch_size - 2))

        pts.append((int(round(x_new)), int(round(y_new))))
        y, x = y_new, x_new

    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    # Main worm body.
    cv2.polylines(
        canvas,
        [pts_np],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Optional side branch.
    if rng.random() < 0.35 and len(pts) >= 2:
        branch_idx = int(rng.integers(1, len(pts)))
        bx, by = pts[branch_idx]

        branch_angle = angle + float(rng.uniform(-1.6, 1.6))
        branch_len = float(step_len * rng.uniform(0.35, 0.75))

        bx2 = int(round(np.clip(bx + branch_len * np.cos(branch_angle), 1, patch_size - 2)))
        by2 = int(round(np.clip(by + branch_len * np.sin(branch_angle), 1, patch_size - 2)))

        cv2.line(
            canvas,
            (bx, by),
            (bx2, by2),
            color=255,
            thickness=max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )

    # Optional short fragment nearby.
    if rng.random() < 0.25:
        fx = int(rng.integers(max(1, cx - base_size), min(patch_size - 1, cx + base_size + 1)))
        fy = int(rng.integers(max(1, cy - base_size), min(patch_size - 1, cy + base_size + 1)))
        frag_len = max(1, int(round(rng.uniform(0.3, 0.8) * base_size)))
        frag_angle = float(rng.uniform(0.0, 2.0 * np.pi))

        fx2 = int(round(np.clip(fx + frag_len * np.cos(frag_angle), 1, patch_size - 2)))
        fy2 = int(round(np.clip(fy + frag_len * np.sin(frag_angle), 1, patch_size - 2)))

        cv2.line(
            canvas,
            (fx, fy),
            (fx2, fy2),
            color=255,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Fast local blur.
    sig_edge = float(rng.uniform(*dirt_sigma))
    alpha_base = canvas.astype(np.float32) / 255.0

    if sig_edge > 0:
        alpha_soft = cv2.GaussianBlur(
            alpha_base,
            ksize=(0, 0),
            sigmaX=sig_edge,
            sigmaY=sig_edge,
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        alpha_soft = alpha_base

    max_val = float(alpha_soft.max())
    if max_val > 0:
        alpha_soft /= max_val

    a = float(rng.uniform(*dirt_alpha))
    alpha_map = (a * alpha_soft).astype(np.float32)

    hmix = float(rng.uniform(0.0, 1.0))
    base_mix = ((1.0 - hmix) * base_orange + hmix * base_green).astype(np.float32)
    bright = float(0.85 + 0.3 * rng.random())
    color = (bright * base_mix).clip(0.0, 1.0).astype(np.float32)

    return alpha_map, color
