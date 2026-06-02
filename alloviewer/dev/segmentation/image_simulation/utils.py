import numpy as np
from typing import (
    Tuple,
    Union,
    Sequence,
    Mapping,
    Any,
    Dict,
    Optional
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

    x = np.clip(img, 0.0, 1.0)
    a = 1.0 + 8.0 * float(strength)

    y = 1.0 / (1.0 + np.exp(-a * (x - 0.5)))
    y0 = 1.0 / (1.0 + np.exp(-a * (0.0 - 0.5)))
    y1 = 1.0 / (1.0 + np.exp(-a * (1.0 - 0.5)))
    y = (y - y0) / (y1 - y0 + 1e-8)
    return np.clip(y, 0.0, 1.0)


def lift_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    w = (1.0 - img) ** 2
    out = img + amount * 0.35 * w
    return np.clip(out, 0.0, 1.0)


def compress_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    thr = 0.72
    out = img.copy()
    mask = out > thr
    if np.any(mask):
        x = out[mask] - thr
        out[mask] = thr + (1.0 - np.exp(-x / (amount + 1e-6))) * (1.0 - thr)
    return np.clip(out, 0.0, 1.0)

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

    shift = float(strength) * per_channel_strength * delta
    out = img + shift.reshape(1, 1, 3)

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

def apply_overexposure_halo(
    img: np.ndarray,
    *,
    threshold: float,
    sigma: float,
    strength: float,
    wash_strength: float = 0.0,
    cell_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Add a local overexposure halo around bright cell signal.

    The effect is driven by bright pixels, optionally restricted by the
    simulated cell mask. Nearby halos naturally overlap because the halo
    field is built on the whole image at once.

    Parameters
    ----------
    img:
        RGB HWC float32 image in [0, 1].
    threshold:
        Only signal above this threshold contributes to the halo.
    sigma:
        Gaussian blur sigma for the halo spread.
    strength:
        Strength of the added halo field.
    wash_strength:
        Optional local contrast washout around the halo.
    cell_mask:
        Optional binary/float cell mask [H, W]. If given, the halo is focused
        on cell regions.
    """
    if sigma <= 0.0 or strength <= 0.0:
        return img

    img = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)

    bright = np.maximum(img - float(threshold), 0.0)

    if cell_mask is not None:
        mask = np.asarray(cell_mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError(f"cell_mask must be 2D, got shape {mask.shape}")

        mask = np.clip(mask, 0.0, 1.0)

        # soften the mask a bit so the halo can spill just beyond the cell body
        mask_soft = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=max(0.5, float(sigma) * 0.75),
            sigmaY=max(0.5, float(sigma) * 0.75),
        )
        mask_soft = np.clip(mask_soft, 0.0, 1.0)

        bright = bright * mask_soft[..., None]

    halo = cv2.GaussianBlur(
        bright,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    ).astype(np.float32)

    # Add the halo in a saturating way so overlap matters,
    # but does not blow up too hard.
    img = img + float(strength) * halo * (1.0 - img)
    img = np.clip(img, 0.0, 1.0)

    # Optional local washout: makes boundaries less crisp in bright regions.
    if wash_strength > 0.0:
        halo_map = halo.max(axis=2, keepdims=True)
        halo_scale = float(halo_map.max()) + 1e-6
        wash_mask = np.clip(halo_map / halo_scale, 0.0, 1.0)

        local_blur = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        ).astype(np.float32)

        alpha = float(wash_strength) * wash_mask
        img = (1.0 - alpha) * img + alpha * local_blur
        img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32)

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

