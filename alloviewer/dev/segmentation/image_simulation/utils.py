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
    contact_factor_range: tuple[float, float] = (0.95, 1.08),
    core_min_sep_factor: float = 0.80,
    chain_probability: float = 0.70,
    angle_jitter: float = 0.65,
    seed_tries: int = 120,
    member_tries: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Place isolated cells and small chain-like clusters efficiently.

    Cluster members are checked only against earlier members of their own
    cluster. Global conflicts are left to the later KD-tree packing pass.
    This keeps placement cost close to O(n_cells) for small cluster sizes.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32).copy()
    r_px = np.asarray(r_px, dtype=np.float32)
    n_cells = int(cluster_ids.size)

    if r_px.shape != (n_cells,):
        raise ValueError("r_px and cluster_ids must have the same length")

    centers = np.full((n_cells, 2), np.nan, dtype=np.float32)
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

    contact_lo = float(min(contact_factor_range))
    contact_hi = float(max(contact_factor_range))
    core_min_sep_factor = max(0.0, float(core_min_sep_factor))
    chain_probability = float(np.clip(chain_probability, 0.0, 1.0))
    angle_jitter = max(0.0, float(angle_jitter))

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
        for _ in range(max(1, int(seed_tries))):
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

        # Rare fallback: sample uniformly inside the valid well radius.
        max_center_radius = max(0.0, R - wall_margin_px - float(r_px[index]))
        for _ in range(256):
            radial_position = max_center_radius * np.sqrt(rng.random())
            angle = rng.uniform(-np.pi, np.pi)
            y = cy + radial_position * np.sin(angle)
            x = cx + radial_position * np.cos(angle)
            if valid_center(y, x, index):
                return np.array([y, x], dtype=np.float32)

        # Last-resort centre placement, clipped to image limits.
        radius = float(r_px[index])
        y = float(np.clip(cy, radius + 1.0, H - radius - 2.0))
        x = float(np.clip(cx, radius + 1.0, W - radius - 2.0))
        return np.array([y, x], dtype=np.float32)

    valid_cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])

    for cluster_id in valid_cluster_ids:
        members = np.flatnonzero(cluster_ids == cluster_id)
        if members.size < 2:
            cluster_ids[members] = -1
            continue

        first = int(members[0])
        centers[first] = independent_center(first)
        heading = float(rng.uniform(-np.pi, np.pi))

        for local_index in range(1, members.size):
            index = int(members[local_index])
            accepted = False

            for _ in range(max(1, int(member_tries))):
                previous_members = members[:local_index]
                eligible_parents = previous_members[
                    cluster_ids[previous_members] == cluster_id
                ]
                if eligible_parents.size == 0:
                    break

                previous_index = int(members[local_index - 1])
                use_chain_parent = (
                    rng.random() < chain_probability
                    and cluster_ids[previous_index] == cluster_id
                )

                if use_chain_parent:
                    parent = previous_index
                    heading += float(rng.normal(0.0, angle_jitter))
                    angle = heading
                else:
                    parent = int(rng.choice(eligible_parents))
                    angle = float(rng.uniform(-np.pi, np.pi))

                target_distance = float(
                    rng.uniform(contact_lo, contact_hi)
                    * (r_px[parent] + r_px[index])
                )

                y = float(centers[parent, 0] + target_distance * np.sin(angle))
                x = float(centers[parent, 1] + target_distance * np.cos(angle))

                if not valid_center(y, x, index):
                    # A centre-directed retry keeps rim clusters inside the well.
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

                previous = members[:local_index]
                previous_centers = centers[previous]
                deltas = previous_centers - np.array([y, x], dtype=np.float32)
                distances = np.sqrt(np.sum(deltas * deltas, axis=1))
                minimum_distances = core_min_sep_factor * (
                    r_px[previous] + r_px[index]
                )

                if np.any(distances < minimum_distances):
                    continue

                centers[index] = (y, x)
                cluster_edges.append((parent, index))
                accepted = True
                break

            if not accepted:
                centers[index] = independent_center(index)
                cluster_ids[index] = -1

    # Remove cluster labels that ended up with fewer than two members.
    remaining_cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])
    for cluster_id in remaining_cluster_ids:
        members = np.flatnonzero(cluster_ids == cluster_id)
        if members.size < 2:
            cluster_ids[members] = -1

    isolated = np.flatnonzero(cluster_ids < 0)
    for index in isolated:
        if not np.isfinite(centers[index]).all():
            centers[index] = independent_center(int(index))

    return centers, cluster_ids, cluster_edges

