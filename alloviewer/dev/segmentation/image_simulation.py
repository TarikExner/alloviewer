import numpy as np
from skimage import segmentation, morphology
from scipy import ndimage as ndi
import inspect
from typing import Mapping, Sequence, Optional, Any

import cv2

from .camera_styles import (
    CameraStyleConfig,
    CameraStyleParams,
    apply_device_quantile_band_match
)
from typing import Dict

RNG = np.random.Generator

def _to_jsonable(x):
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
        return type(x)(_to_jsonable(v) for v in x)

    # small numpy arrays (avoid dumping huge arrays by mistake)
    if isinstance(x, np.ndarray):
        # keep tiny shapes, otherwise store shape + dtype
        if x.size <= 64:
            return x.tolist()
        return {"__ndarray__": True, "shape": tuple(x.shape), "dtype": str(x.dtype)}

    # mappings
    if isinstance(x, Mapping):
        return {k: _to_jsonable(v) for k, v in x.items()}

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
            out[name] = _to_jsonable(locals_dict[name])
    return out


def simulate_image(
    # --- size / geometry ---
    H=512, W=512,
    well_radius_frac=0.42,
    well_center_jitter=0.02,

    # --- radial look of the well ---
    background_level=0.08,
    edge_boost=0.25,
    radial_gamma=1.2,
    vignette_strength=0.20,

    # --- color mix ---
    bg_hue=0.25,             # 0=orange, 1=green

    # --- cells (sharp ones inside the well) ---
    n_cells=150,
    cell_diameter=20,

    large_cell_frac=0.0,              # fraction of inside-well cells that are "large"
    large_cell_diameter_factor=1.5,   # large size = factor * cell_diameter

    # --- cells: shape + brightness ---
    cell_ellipse_enable=True,
    cell_axis_jitter=0.20,          # ±20% axis ratio
    cell_random_rotation=True,      # random rotation angle
    cell_intensity_range=(0.70, 1.05),  # per-cell brightness multiplier (was ~0.9..1.1)

    frac_positive=0.5,
    color_jitter=0.07,
    sigma_in=(0.5, 1.0),
    sigma_out=(1.6, 3.2),    # used if focus_frac_in<1
    focus_frac_in=1.0,       # default: draw sharp; ghosts carry the blur
    in_focus_sigma_thresh=None,
    boundary_width=1,

    # crowd cells near the *outer* wall, but keep a filled center
    rim_bias=0.85,
    rim_band=0.12,
    edge_clamp=0.65,

    # --- collision / packing control ---
    min_cell_sep_px=None,   # if None -> 0.9 * cell_diameter
    rim_min_sep_px=4,
    pack_iters=20,          # fewer iters thanks to vectorized packing
    pack_strength=0.45,     # 0..1, how far to push per step
    wall_margin_px=2.0,     # keep centers this far from the wall

    # --- sidedness (pile-up on one side near the rim) ---
    side_bias_enable=False,     # set True to activate
    side_bias_theta=0.0,        # radians; 0=right, +pi/2=up, pi=left, -pi/2=down
    side_bias_strength=0.75,    # 0..1 mixture with uniform (higher = stronger bias)
    side_bias_kappa=5.0,        # von Mises concentration at the rim (higher = tighter)
    side_bias_inner_frac=0.55,  # start fading bias below this fraction of R (center stays even)

    # --- visual wall (soft rim) ---
    wall_blur_sigma=12.0,
    ring_artifacts=0,
    ring_sigma_range=(6.0, 18.0),
    ring_alpha_range=(0.03, 0.12),

    # --- “ghost cells” OUTSIDE the well (big, elongated, not in masks) ---
    ghost_enable=True,
    ghost_density=0.50,      # fraction relative to number of rim cells
    ghost_offset_px=10.0,
    ghost_offset_jitter=6.0,
    ghost_sigma=(2.5, 6.0),  # base sigma (minor axis)
    ghost_dilate=1.0,
    ghost_intensity=(0.8, 1.4),

    # NEW: outward elongation and short trail
    ghost_stretch=3.0,       # major/minor axis ratio (>1 stretches outward)
    ghost_trail=3,           # number of faded lobes outward
    ghost_trail_decay=0.6,   # amplitude decay per lobe (0..1)

    # --- debris INSIDE the well (small + dim) ---
    dirt_density=0.0007,
    dirt_size=(2, 4),
    dirt_sigma=(1.2, 2.0),
    dirt_alpha=(0.01, 0.04),

    # --- noise / camera ---
    blur_sigma_global=0.0,
    photon_level=2500,
    read_noise=0.003,

    # --- radial reflections on the wall (outside the well) ---
    reflect_enable=True,
    reflect_n=6,                 # number of streak groups
    reflect_theta_sigma=0.10,    # angular width of a streak (radians)
    reflect_radial_sigma=8.0,    # radial softness (pixels)
    reflect_offset_range=(6.0, 24.0),   # how far outside R the streak sits
    reflect_alpha_range=(0.05, 0.20),   # strength
    reflect_wobble=0.35,         # small angular wiggle per streak (radians)
    reflect_harmonics=2,         # add faint copies to get a comb feel
    reflect_harmonic_decay=0.55, # falloff for those copies

    seed=None,
    return_targets=True,
):
    """
    Returns:
      image: (H, W, 3) float32 in [0,1]
      meta: dict (centers, labels, well_center, radius_px, final_sigmas)
      targets (if return_targets):
        instance_labels: int32 (H,W) 0=bg, 1..K for *all inside-well cells*
        cell_mask: float32 (H,W) 1 for in-focus cells only
        boundary: float32 (H,W)
    """
    rng = np.random.default_rng(seed)
    H = int(H)
    W = int(W)

    # ---------- fast blur helper (SciPy) ----------
    def blur(a, sigma):
        if sigma is None or sigma <= 0:
            return a
        if a.ndim == 2:
            return ndi.gaussian_filter(a, sigma=float(sigma), mode='reflect')
        elif a.ndim == 3:
            return ndi.gaussian_filter(a, sigma=(float(sigma), float(sigma), 0.0), mode='reflect')
        else:
            return a

    # ---------- helpers ----------
    def jitter_color(base_rgb):
        j = rng.normal(1.0, color_jitter, size=3).astype(np.float32)
        c = (base_rgb * j).clip(0, 1)
        scale = np.linalg.norm(base_rgb) / max(1e-6, np.linalg.norm(c))
        return (c * scale).clip(0, 1)

    base_orange = np.array([1.00, 0.62, 0.08], dtype=np.float32)
    base_green  = np.array([0.05, 0.95, 0.35], dtype=np.float32)
    bg_color = ((1.0 - bg_hue) * base_orange + bg_hue * base_green).astype(np.float32)

    # ---------- grids / well ----------
    yy, xx = np.mgrid[0:H, 0:W]
    cy = H/2 + rng.normal(0, well_center_jitter * min(H, W))
    cx = W/2 + rng.normal(0, well_center_jitter * min(H, W))
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    R = well_radius_frac * min(H, W)
    inside = rr <= R
    r_norm = np.clip(rr / R, 0, 1)

    illum = r_norm**radial_gamma
    bg_inside = (background_level + edge_boost * illum).astype(np.float32)
    img = (bg_inside[..., None] * bg_color[None, None, :]).astype(np.float32)

    # vignette
    ry = (yy - H/2) / (0.5 * H)
    rx = (xx - W/2) / (0.5 * W)
    v = np.sqrt(rx**2 + ry**2)
    img *= (1.0 - vignette_strength * (v**2))[..., None].clip(0.15, 1.0).astype(np.float32)

    # soft wall glow (background only)
    rim = np.exp(-((rr - R)**2) / (2 * (0.8)**2)).astype(np.float32)
    rim = blur(rim, wall_blur_sigma)
    rim = (rim / (rim.max() + 1e-8))
    img += (0.06 * rim)[..., None] * bg_color[None, None, :]

    if ring_artifacts > 0:
        rings = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(ring_artifacts)):
            r_shift = rng.uniform(-0.04, 0.06) * R
            sig = rng.uniform(*ring_sigma_range)
            a = rng.uniform(*ring_alpha_range)
            ring = np.exp(-((rr - (R + r_shift))**2) / (2 * 1.0**2))
            ring = blur(ring, sig)
            ring /= ring.max() + 1e-8
            rings += a * ring
        img += rings[..., None] * bg_color[None, None, :]

    # small background texture inside the well
    tex1 = rng.normal(0, 1.0, size=(H, W)).astype(np.float32)   # fine
    tex2 = rng.normal(0, 1.0, size=(H, W)).astype(np.float32)   # coarse

    tex1 = blur(tex1, 2.0)
    tex2 = blur(tex2, 12.0)

    tex = 0.65 * tex1 + 0.35 * tex2
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)

    # multiplicative speckle strength (increase for more “grain”)
    tex_mul = 0.18
    tex = np.clip(1.0 + tex_mul * tex, 0.1, 1.6).astype(np.float32)

    img[inside] *= tex[inside, None]

    # ---------- place sharp cells inside the well ----------
    n_pos = int(round(frac_positive * n_cells))
    labels = np.array([1]*n_pos + [0]*(n_cells - n_pos), dtype=np.int32)
    rng.shuffle(labels)

    # ---------- per-cell diameters (two populations) + per-cell jitter ----------
    # base diameters: small vs large
    large_cell_frac = float(np.clip(large_cell_frac, 0.0, 1.0))
    is_large = (rng.random(n_cells) < large_cell_frac)

    diameters = np.full(n_cells, float(cell_diameter), dtype=np.float32)
    diameters[is_large] *= float(large_cell_diameter_factor)

    # per-cell jitter: sample N(1, 0.10) then clip to [0.8, 1.2] == ±20%
    # (0.10 std => about 95% within ±20% before clipping)
    jitter = rng.normal(1.0, 0.10, size=n_cells).astype(np.float32)
    jitter = np.clip(jitter, 0.8, 1.2)
    diameters *= jitter

    # integer radii in pixels for packing/painting (>=2 px)
    radii = np.maximum(2, np.round(diameters / 2.0).astype(np.int32))

    # ---------- per-cell ellipse params ----------
    if cell_ellipse_enable:
        # axis ratio around 1.0, clipped to [1-ax, 1+ax]
        axis_ratio = rng.uniform(1.0 - cell_axis_jitter, 1.0 + cell_axis_jitter, size=n_cells).astype(np.float32)
        axis_ratio = np.clip(axis_ratio, 1.0 - cell_axis_jitter, 1.0 + cell_axis_jitter)

        if cell_random_rotation:
            theta = rng.uniform(0.0, 2.0*np.pi, size=n_cells).astype(np.float32)
        else:
            theta = np.zeros(n_cells, dtype=np.float32)
    else:
        axis_ratio = np.ones(n_cells, dtype=np.float32)
        theta = np.zeros(n_cells, dtype=np.float32)

    # area-uniform radius sampling with optional rim pull
    def sample_radius():
        a = (1.0 - rim_band) * R
        if rng.random() < rim_bias:
            # [a, R] uniform by area
            r = np.sqrt(a*a + rng.random() * (R*R - a*a))
            if edge_clamp > 0:
                r = (1.0 - edge_clamp) * r + edge_clamp * R
                r -= rng.uniform(0.0, 0.02 * rim_band * R)  # jitter
        else:
            r = a * np.sqrt(rng.random())
        return float(np.clip(r, 0.05*R, 0.985*R))

    def sample_theta(r):
        """Angle with sidedness near rim; fades to uniform toward center."""
        if not side_bias_enable:
            return rng.uniform(-np.pi, np.pi)
        f0 = float(np.clip(side_bias_inner_frac, 0.0, 0.99))
        w_rad = np.clip((r / R - f0) / (1.0 - f0 + 1e-6), 0.0, 1.0)
        if w_rad <= 0:
            return rng.uniform(-np.pi, np.pi)
        kappa = max(1e-3, side_bias_kappa * w_rad)
        mix_p = np.clip(side_bias_strength * w_rad, 0.0, 1.0)
        if rng.random() < mix_p:
            return rng.vonmises(side_bias_theta, kappa)
        else:
            return rng.uniform(-np.pi, np.pi)

    centers = []
    is_rim = []
    tries, max_tries = 0, 400 * n_cells
    while len(centers) < n_cells and tries < max_tries:
        tries += 1
        r_s = sample_radius()
        th = sample_theta(r_s)
        y = int(round(cy + r_s * np.sin(th)))
        x = int(round(cx + r_s * np.cos(th)))
        i = len(centers)
        ri = int(radii[i])

        if not (ri < y < H - ri - 1 and ri < x < W - ri - 1):
            continue

        # enforce spacing only for rim-band points (sampling-time hint)
        rim_inner = (1.0 - rim_band) * R
        is_this_rim = (r_s >= rim_inner)
        ok = True
        if is_this_rim and rim_min_sep_px > 0:
            for (py, px), rim_flag in zip(centers[-200:], is_rim[-200:]):
                if rim_flag:
                    if (py - y)**2 + (px - x)**2 < rim_min_sep_px**2:
                        ok = False
                        break
        if ok:
            centers.append((y, x))
            is_rim.append(is_this_rim)

    while len(centers) < n_cells:
        i = len(centers)
        ri = int(radii[i])
        for _ in range(2000):
            y = int(rng.integers(ri+1, H-ri-1))
            x = int(rng.integers(ri+1, W-ri-1))
            if np.sqrt((y - cy)**2 + (x - cx)**2) <= (R - wall_margin_px - ri):
                centers.append((y, x))
                is_rim.append(False)
                break
        else:
            # if we failed a lot, just accept (rare)
            centers.append((y, x))
            is_rim.append(False)

    # ---------- resolve overlaps: push-apart circle packing (vectorized) ----------
    # effective radius = major axis scale * radius (keeps overlaps similar to round case)
    r_px = radii.astype(np.float32) * np.sqrt(np.maximum(axis_ratio, 1.0/axis_ratio)).astype(np.float32)
    eps = 1e-6


    # default: scale separation with cell size; preserves old "0.9*diameter" feel
    if min_cell_sep_px is None:
        min_sep_mat = 0.9 * (r_px[:, None] + r_px[None, :])  # (N,N)
    else:
        min_sep_scalar = float(min_cell_sep_px)

    cf = np.array(centers, dtype=np.float32)  # (N, 2): (y, x)
    N = cf.shape[0]

    for _ in range(int(pack_iters)):
        v = cf[:, None, :] - cf[None, :, :]            # (N, N, 2)
        d = np.sqrt((v**2).sum(axis=2))                # (N, N)
        np.fill_diagonal(d, np.inf)                    # exclude self-pairs

        if min_cell_sep_px is None:
            M = d < min_sep_mat
        else:
            M = d < min_sep_scalar

        if not M.any():
            break

        # safe division (inf on diagonal -> 0 direction there)
        u = v / (d[..., None] + eps)

        # IMPORTANT: avoid (±inf) * 0 -> NaN
        overlap = np.zeros_like(d, dtype=np.float32)
        if min_cell_sep_px is None:
            overlap[M] = (min_sep_mat[M] - d[M]).astype(np.float32)
        else:
            overlap[M] = (min_sep_scalar - d[M]).astype(np.float32)

        disp = (u * overlap[..., None]).sum(axis=1)    # (N,2)
        cf += (pack_strength * 0.5) * disp

        # keep inside image bounds (per-cell)
        cf[:, 0] = np.clip(cf[:, 0], r_px + 1, H - r_px - 2)
        cf[:, 1] = np.clip(cf[:, 1], r_px + 1, W - r_px - 2)

        # keep inside well margin (per-cell)
        vy = cf[:, 0] - cy
        vx = cf[:, 1] - cx
        rr_c = np.sqrt(vy*vy + vx*vx) + eps

        max_r_center = (R - wall_margin_px - r_px).astype(np.float32)
        max_r_center = np.maximum(0.0, max_r_center)

        too_far = rr_c > max_r_center
        if np.any(too_far):
            s = (max_r_center[too_far] / rr_c[too_far]).astype(np.float32)
            cf[too_far, 0] = cy + vy[too_far] * s
            cf[too_far, 1] = cx + vx[too_far] * s

    centers = [(int(round(y)), int(round(x))) for (y, x) in cf]

    # ---------- instance map (fast paint) ----------
    inst = np.zeros((H, W), dtype=np.int32)

    def ellipse_mask(radius, ratio, angle):
        # keep area roughly constant: a*b = radius^2
        s = float(np.sqrt(ratio))
        a = float(radius) * s       # semi-major
        b = float(radius) / s       # semi-minor

        r_box = int(np.ceil(max(a, b)))
        yy_p, xx_p = np.mgrid[-r_box:r_box+1, -r_box:r_box+1].astype(np.float32)
        ca, sa = np.cos(angle), np.sin(angle)
        xr =  ca*xx_p + sa*yy_p
        yr = -sa*xx_p + ca*yy_p
        m = (xr*xr)/(a*a + 1e-8) + (yr*yr)/(b*b + 1e-8) <= 1.0
        return m, r_box

    for k_id, (y, x) in enumerate(centers, start=1):
        r = int(radii[k_id - 1])
        m, r_box = ellipse_mask(r, float(axis_ratio[k_id - 1]), float(theta[k_id - 1]))

        y0, y1 = y - r_box, y + r_box + 1
        x0, x1 = x - r_box, x + r_box + 1
        if y1 <= 0 or x1 <= 0 or y0 >= H or x0 >= W:
            continue
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        sy0, sy1 = y0c - y0, y1c - y0
        sx0, sx1 = x0c - x0, x1c - x0

        m_loc = m[sy0:sy1, sx0:sx1]
        sl = (slice(y0c, y1c), slice(x0c, x1c))

        write_mask = m_loc & (rr[y0c:y1c, x0c:x1c] <= R)
        inst[sl] = np.where(write_mask, k_id, inst[sl])

    # ---------- render sharp cells ----------
    final_sigmas = np.zeros(n_cells, dtype=np.float32)

    for k_id, (y, x) in enumerate(centers):
        base_col = base_orange if labels[k_id] == 1 else base_green
        col = jitter_color(base_col)
        sig = rng.uniform(*sigma_in) if rng.random() < focus_frac_in else rng.uniform(*sigma_out)
        final_sigmas[k_id] = sig

        d0 = float(diameters[k_id])
        radius = int(np.ceil(max(2, 0.5 * d0 + 3 * sig)))
        yx = np.mgrid[-radius:radius+1, -radius:radius+1]
        gsig = (d0 / 6.0 + sig)
        g = np.exp(-(yx[0]**2 + yx[1]**2) / (2 * (gsig**2))).astype(np.float32)
        g /= g.max() + 1e-8
        amp = float(rng.uniform(*cell_intensity_range))

        y0, y1 = y - radius, y + radius + 1
        x0, x1 = x - radius, x + radius + 1
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        sy0, sy1 = y0c - y0, y1c - y0
        sx0, sx1 = x0c - x0, x1c - x0
        img[y0c:y1c, x0c:x1c, :] += amp * g[sy0:sy1, sx0:sx1][..., None] * col[None, None, :]

    # ---------- elongated ghosts OUTSIDE the wall (not in masks) ----------
    def draw_elliptical_gaussian(dst, gy, gx, sig_minor, stretch, angle_rad, amp, col, ref_diameter):
        sig_x = sig_minor * stretch
        sig_y = sig_minor
        radius = int(np.ceil(ghost_dilate * (0.5 * float(ref_diameter) + 3 * max(sig_x, sig_y))))
        yy_l, xx_l = np.mgrid[-radius:radius+1, -radius:radius+1].astype(np.float32)
        ca, sa = np.cos(angle_rad), np.sin(angle_rad)
        xr =  ca*xx_l + sa*yy_l
        yr = -sa*xx_l + ca*yy_l
        g = np.exp(-0.5 * ((xr/sig_x)**2 + (yr/sig_y)**2)).astype(np.float32)
        g /= g.max() + 1e-8
        y0, y1 = gy - radius, gy + radius + 1
        x0, x1 = gx - radius, gx + radius + 1
        if y1 <= 0 or x1 <= 0 or y0 >= H or x0 >= W:
            return
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        sy0, sy1 = y0c - y0, y1c - y0
        sx0, sx1 = x0c - x0, x1c - x0
        dst[y0c:y1c, x0c:x1c, :] += amp * g[sy0:sy1, sx0:sx1][..., None] * col[None, None, :]

    if ghost_enable and ghost_density > 0:
        rim_idx = [i for i, flag in enumerate(is_rim) if flag]
        if rim_idx:
            n_ghost = max(1, int(len(rim_idx) * ghost_density))
            pick = rng.choice(rim_idx, size=n_ghost, replace=True)
            for i in pick:
                ang = np.arctan2(centers[i][0] - cy, centers[i][1] - cx)
                roff = max(1.0, rng.normal(ghost_offset_px, ghost_offset_jitter))
                base_y = int(round(cy + (R + roff) * np.sin(ang)))
                base_x = int(round(cx + (R + roff) * np.cos(ang)))
                base_col = base_green if labels[i] == 0 else base_orange
                col = jitter_color(base_col)

                sig0 = rng.uniform(*ghost_sigma)

                amp0 = float(rng.uniform(*ghost_intensity))
                draw_elliptical_gaussian(img, base_y, base_x, sig0, ghost_stretch, ang, amp0, col, diameters[i])

                step = max(2.0, 0.6 * float(diameters[i]))  # outward spacing per lobe
                amp = amp0 * ghost_trail_decay
                for t in range(1, int(ghost_trail)):
                    gy = int(round(base_y + t * step * np.sin(ang)))
                    gx = int(round(base_x + t * step * np.cos(ang)))
                    sig_t = sig0 * (1.0 + 0.4*t)  # grow a bit along the trail
                    draw_elliptical_gaussian(img, gy, gx, sig_t, ghost_stretch, ang, amp, col, diameters[i])
                    amp *= ghost_trail_decay

    # --- RADIAL REFLECTIONS (outside the well) ---
    if reflect_enable and reflect_n > 0:
        ang = np.arctan2(yy - cy, xx - cx)  # [-pi, pi]

        def angle_wrap(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        refl = np.zeros((H, W), dtype=np.float32)

        for _ in range(int(reflect_n)):
            theta0 = rng.uniform(-np.pi, np.pi)
            theta0 += rng.normal(0, reflect_wobble)
            r_off  = rng.uniform(*reflect_offset_range)
            alpha  = rng.uniform(*reflect_alpha_range)
            radial_term = np.exp(-((rr - (R + r_off))**2) / (2 * reflect_radial_sigma**2))
            angular_term = np.exp(-(angle_wrap(ang - theta0)**2) / (2 * reflect_theta_sigma**2))
            base = radial_term * angular_term

            comb = base.copy()
            th_sig = reflect_theta_sigma
            for h in range(1, int(reflect_harmonics) + 1):
                decay = reflect_harmonic_decay**h
                comb += decay * np.exp(-(angle_wrap(ang - (theta0 + h*th_sig*2.0))**2) / (2 * th_sig**2)) * radial_term
                comb += decay * np.exp(-(angle_wrap(ang - (theta0 - h*th_sig*2.0))**2) / (2 * th_sig**2)) * radial_term

            refl += alpha * comb

        outside = rr > R
        refl = np.clip(refl / (refl.max() + 1e-8), 0, 1)
        img[outside, 0] += (refl[outside] * bg_color[0] * 0.9).astype(np.float32)
        img[outside, 1] += (refl[outside] * bg_color[1] * 0.9).astype(np.float32)
        img[outside, 2] += (refl[outside] * bg_color[2] * 0.9).astype(np.float32)

    # ---------- debris (small, dim) — irregular blobs, INSIDE the well only ----------
    inside_idx = np.flatnonzero(inside.ravel())
    if inside_idx.size > 0:
        n_dirt = int(inside.sum() * float(dirt_density))
        for _ in range(n_dirt):
            idx = int(rng.choice(inside_idx))
            ry, rx = divmod(idx, W)

            rad_d = int(rng.integers(dirt_size[0], dirt_size[1] + 1))
            patch_r = max(3, int(rad_d * rng.uniform(1.0, 1.6)))
            y0, y1 = ry - patch_r, ry + patch_r + 1
            x0, x1 = rx - patch_r, rx + patch_r + 1
            if y1 <= 0 or x1 <= 0 or y0 >= H or x0 >= W:
                continue

            y0c, y1c = max(0, y0), min(H, y1)
            x0c, x1c = max(0, x0), min(W, x1)
            sy0, sy1 = y0c - y0, y1c - y0
            sx0, sx1 = x0c - x0, x1c - x0

            mask_in = inside[y0c:y1c, x0c:x1c]
            if mask_in.sum() == 0:
                continue
            if mask_in.mean() < 0.25 and rng.random() < 0.7:
                continue

            ps_h = (y1 - y0)
            ps_w = (x1 - x0)
            noise = rng.normal(0.0, 1.0, size=(ps_h, ps_w)).astype(np.float32)
            base_sig = rng.uniform(0.8, 1.6)
            field = blur(noise, base_sig)

            thr = np.percentile(field, rng.uniform(72.0, 90.0))
            blob = (field > thr)

            yy_l, xx_l = np.mgrid[y0:y1, x0:x1]
            dy = yy_l - ry
            dx = xx_l - rx
            r2 = (dy*dy + dx*dx).astype(np.float32)
            mask_center = r2 <= (rad_d * rad_d * rng.uniform(0.9, 1.4))
            blob = np.logical_and(blob, mask_center)

            r1 = int(rng.integers(0, 2))   # 0 or 1
            r2c = int(rng.integers(0, 2))  # 0 or 1
            if r1 > 0:
                blob = morphology.binary_opening(blob, footprint=morphology.disk(r1))
            if r2c > 0:
                blob = morphology.binary_closing(blob, footprint=morphology.disk(r2c))

            if not blob.any():
                continue

            sig_edge = rng.uniform(*dirt_sigma)
            alpha_soft = blur(blob.astype(np.float32), sig_edge)
            alpha_soft = alpha_soft / (alpha_soft.max() + 1e-8)

            a = rng.uniform(*dirt_alpha)
            alpha_map = (a * alpha_soft).astype(np.float32)

            h = float(rng.uniform(0.0, 1.0))  # 0=orange, 1=green
            base_mix = ((1.0 - h) * base_orange + h * base_green).astype(np.float32)
            bright = float(0.85 + 0.3 * rng.random())
            col = (bright * base_mix).clip(0.0, 1.0).astype(np.float32)

            alpha_local = alpha_map[sy0:sy1, sx0:sx1] * mask_in.astype(np.float32)
            if alpha_local.max() <= 0:
                continue

            sl = (slice(y0c, y1c), slice(x0c, x1c))
            img[sl + (slice(None),)] += alpha_local[..., None] * col[None, None, :]

    # optional global blur
    if blur_sigma_global and blur_sigma_global > 0:
        img = blur(img, float(blur_sigma_global))

    img = np.clip(img, 0, 1).astype(np.float32)

    # ---------- camera noise ----------
    counts = (img * photon_level).astype(np.float32)
    noised = rng.poisson(counts).astype(np.float32) / max(1.0, photon_level)
    noised += rng.normal(0.0, read_noise, size=noised.shape).astype(np.float32)
    noised = np.clip(noised, 0, 1).astype(np.float32)

    # ---------- targets ----------
    if in_focus_sigma_thresh is None:
        in_focus_sigma_thresh = 1.15 * max(sigma_in)

    inst_all = inst.astype(np.int32)

    # decide which instances are "in focus"
    thr = float(in_focus_sigma_thresh)
    keep_ids = np.flatnonzero(final_sigmas <= thr) + 1  # ids are 1..n_cells

    inst_in = np.where(np.isin(inst_all, keep_ids), inst_all, 0).astype(np.int32)
    cell_mask = (inst_in > 0)

    boundary = segmentation.find_boundaries(inst_in, mode="thick")
    if boundary_width and boundary_width >= 1:
        boundary = morphology.binary_dilation(boundary, morphology.disk(int(boundary_width)))
    boundary = np.logical_and(boundary, cell_mask)

    targets = {
        "instance_labels": inst_all,
        "cell_mask": cell_mask.astype(np.float32),
        "boundary": boundary.astype(np.float32),
    } if return_targets else {}
    if min_cell_sep_px is None:
        min_cell_sep_px = 0.9 * cell_diameter

    all_params = capture_params(simulate_image, locals())

    meta = {
        "centers": centers,
        "labels": labels,
        "frac_positive": frac_positive,
        "n_cells": n_cells,
        "well_center": (float(cy), float(cx)),
        "radius_px": float(R),
        "params": all_params
    }

    return noised, meta, targets

def _apply_s_curve(img: np.ndarray, strength: float) -> np.ndarray:
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


def _lift_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    w = (1.0 - img) ** 2
    out = img + amount * 0.35 * w
    return np.clip(out, 0.0, 1.0)


def _compress_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    thr = 0.72
    out = img.copy()
    mask = out > thr
    if np.any(mask):
        x = out[mask] - thr
        out[mask] = thr + (1.0 - np.exp(-x / (amount + 1e-6))) * (1.0 - thr)
    return np.clip(out, 0.0, 1.0)

def _apply_channel_median_match(
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

def _compress_highlights_piecewise(
    img: np.ndarray,
    threshold: float = 0.75,
    strength: float = 0.15,
) -> np.ndarray:
    """
    Mildly compress values above `threshold`.

    strength:
        0.0 -> no compression
        larger -> stronger compression
    """
    out = np.clip(img.astype(np.float32), 0.0, 1.0).copy()
    m = out > threshold
    if not np.any(m):
        return out

    t = (out[m] - threshold) / max(1e-6, 1.0 - threshold)
    # exponent > 1 compresses the upper tail
    out[m] = threshold + (1.0 - threshold) * np.power(t, 1.0 + strength)
    return np.clip(out, 0.0, 1.0)


def _weak_quantize(
    img: np.ndarray,
    rng: RNG,
    levels: int = 128,
    blend: float = 0.15,
    dither_scale: float = 0.25,
) -> np.ndarray:
    """
    Very light quantization with tiny dither.

    This is only meant to make histograms less smooth, not to create visible banding.
    """
    if levels <= 1 or blend <= 0:
        return np.clip(img.astype(np.float32), 0.0, 1.0)

    x = np.clip(img.astype(np.float32), 0.0, 1.0)
    step = 1.0 / float(levels - 1)

    noise = rng.normal(0.0, dither_scale * step, size=x.shape).astype(np.float32)
    x_noisy = np.clip(x + noise, 0.0, 1.0)

    q = np.round(x_noisy * (levels - 1)) / float(levels - 1)
    out = (1.0 - blend) * x + blend * q
    return np.clip(out, 0.0, 1.0)


def _apply_microscope_blue_cleanup(
    img: np.ndarray,
    rng: RNG,
) -> np.ndarray:
    """
    Real microscope images have much less blue bleed-through,
    especially near zero and in the low/mid range.
    """
    out = np.clip(img.astype(np.float32), 0.0, 1.0).copy()
    b = out[..., 2]

    # stronger suppression in low and mid intensities
    low = np.clip((0.40 - b) / 0.40, 0.0, 1.0)
    mid = np.clip((0.70 - b) / 0.70, 0.0, 1.0)

    b = b * (1.0 - 0.42 * np.power(low, 1.2) - 0.10 * np.power(mid, 2.0))
    b = np.clip(b, 0.0, 1.0)

    # snap a fraction of very-low blue values exactly to zero
    near0 = np.clip((0.06 - b) / 0.06, 0.0, 1.0)
    p_zero = 0.55 * np.power(near0, 1.5)
    mask = rng.random(b.shape) < p_zero
    b = np.where(mask, 0.0, b)

    out[..., 2] = np.clip(b, 0.0, 1.0)
    return out


def _apply_iphone_blue_toe(
    img: np.ndarray,
) -> np.ndarray:
    """
    Small blue suppression in the low/mid range.
    Helps reduce the overly smooth blue tail without changing the whole image too much.
    """
    out = np.clip(img.astype(np.float32), 0.0, 1.0).copy()
    b = out[..., 2]

    toe = np.clip((0.45 - b) / 0.45, 0.0, 1.0)
    b = b * (1.0 - 0.10 * toe)

    out[..., 2] = np.clip(b, 0.0, 1.0)
    return out

def _compress_channel_highlights(x: np.ndarray, threshold: float = 0.75, strength: float = 0.15) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    out = x.copy()
    mask = out > threshold
    if np.any(mask):
        t = (out[mask] - threshold) / max(1e-6, 1.0 - threshold)
        t = t / (1.0 + strength * 4.0 * t)
        out[mask] = threshold + t * (1.0 - threshold)
    return np.clip(out, 0.0, 1.0)


def _lift_channel_midtones(
    x: np.ndarray,
    center: float = 0.4,
    width: float = 0.18,
    amount: float = 0.05,
) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    w = np.exp(-0.5 * ((x - center) / max(width, 1e-6)) ** 2).astype(np.float32)
    out = x + amount * w
    return np.clip(out, 0.0, 1.0)


def _remap_channel_curve(
    x: np.ndarray,
    x_knots,
    y_knots,
) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    xp = np.asarray(x_knots, dtype=np.float32)
    fp = np.asarray(y_knots, dtype=np.float32)
    return np.clip(np.interp(x, xp, fp).astype(np.float32), 0.0, 1.0)


def _apply_device_histogram_polish(
    img: np.ndarray,
    style_name: str,
    rng: RNG,
) -> np.ndarray:
    """
    Final small device-specific histogram polish.

    Purpose:
      - reduce overshot tails
      - make histograms less smooth
      - add a bit of device-like irregularity
    """
    out = np.clip(img.astype(np.float32), 0.0, 1.0).copy()

    if style_name == "iphone":
        out = _compress_highlights_piecewise(out, threshold=0.72, strength=0.18)
        out = _apply_iphone_blue_toe(out)
        out = _weak_quantize(out, rng=rng, levels=96, blend=0.22, dither_scale=0.35)

    elif style_name == "googlepixel":
        # 1) global high-end control
        out = _compress_highlights_piecewise(out, threshold=0.78, strength=0.14)
    
        # 2) extra channel-wise tail compression
        out[..., 0] = _compress_channel_highlights(out[..., 0], threshold=0.74, strength=0.18)  # R
        out[..., 1] = _compress_channel_highlights(out[..., 1], threshold=0.76, strength=0.16)  # G
    
        # 3) slight red midtone lift
        out[..., 0] = _lift_channel_midtones(out[..., 0], center=0.42, width=0.18, amount=0.06)
    
        # 4) blue low/mid reshape
        out[..., 2] = _remap_channel_curve(
            out[..., 2],
            x_knots=[0.0, 0.06, 0.16, 0.32, 0.55, 1.0],
            y_knots=[0.0, 0.05, 0.13, 0.27, 0.50, 1.0],
        )

        # 5) mild histogram roughening
        out = _weak_quantize(out, rng=rng, levels=112, blend=0.16, dither_scale=0.30)

    elif style_name == "microscope":
        out = _compress_highlights_piecewise(out, threshold=0.82, strength=0.08)
        out = _apply_microscope_blue_cleanup(out, rng=rng)
        out = _weak_quantize(out, rng=rng, levels=128, blend=0.10, dither_scale=0.20)

    return np.clip(out, 0.0, 1.0)

def apply_camera_style(
    img: np.ndarray,
    rng: RNG,
    style_cfg: CameraStyleConfig,
    style_registry: Dict[str, CameraStyleParams],
    quantile_band_cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply a sampled camera style, then softly push the result into the
    corresponding real-device histogram band.

    Returns
    -------
    img_out : np.ndarray
        RGB float32 image in [0,1]

    """
    assert img.ndim == 3 and img.shape[2] == 3

    img = np.clip(img.astype(np.float32).copy(), 0.0, 1.0)

    style_name = style_cfg.sample_style(rng)
    if style_name not in style_registry:
        raise KeyError(f"Style '{style_name}' not found in style_registry")

    params = style_registry[style_name]

    if style_name == "simulated_raw":
        return img

    H, W, _ = img.shape

    # 1) exposure
    exposure = rng.uniform(*params.exposure_range)
    img = np.clip(img * exposure, 0.0, 1.0)

    # 2) global contrast / brightness
    c = rng.uniform(*params.c_range)
    b = rng.uniform(*params.b_range)
    img = np.clip(img * c + b, 0.0, 1.0)

    # 3) white balance
    wb = rng.uniform(params.wb_range[0], params.wb_range[1], size=3).astype(np.float32)
    wb = wb / (wb.mean() + 1e-8)
    img = np.clip(img * wb[None, None, :], 0.0, 1.0)

    # 4) explicit color-axis shifts
    gm = rng.uniform(*params.green_magenta_shift_range)
    by = rng.uniform(*params.blue_yellow_shift_range)

    color_shift = np.array(
        [
            1.0 - 0.35 * gm - 0.50 * by,
            1.0 + 1.00 * gm,
            1.0 - 0.35 * gm + 0.50 * by,
        ],
        dtype=np.float32,
    )
    color_shift = color_shift / (color_shift.mean() + 1e-8)
    img = np.clip(img * color_shift[None, None, :], 0.0, 1.0)

    # 5) saturation
    sat = rng.uniform(*params.saturation_range)
    gray = img.mean(axis=2, keepdims=True)
    img = np.clip(gray + sat * (img - gray), 0.0, 1.0)

    # 6) R/G mixing
    a = rng.uniform(*params.mix_range)
    M = np.array([
        [1.0 - a, a,       0.0],
        [a,       1.0 - a, 0.0],
        [0.0,     0.0,     1.0],
    ], dtype=np.float32)
    img = np.clip(img @ M.T, 0.0, 1.0)

    # 7) uneven illumination
    illum_amp = rng.uniform(*params.illum_amp_range)
    if illum_amp > 0:
        scale = 64
        h_small = max(1, H // scale)
        w_small = max(1, W // scale)
        field_small = rng.normal(0.0, 1.0, size=(h_small, w_small)).astype(np.float32)

        field = cv2.resize(field_small, (W, H), interpolation=cv2.INTER_CUBIC)
        field = field - field.mean()
        field = field / (field.std() + 1e-6)
        field = 1.0 + illum_amp * field
        field = np.clip(field, 1.0 - 2.0 * illum_amp, 1.0 + 2.0 * illum_amp)
        img = np.clip(img * field[..., None], 0.0, 1.0)

    # 8) vignette
    vignette_amp = rng.uniform(*params.vignette_amp_range)
    if vignette_amp > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = rr / (0.72 * max(H, W))
        vignette = 1.0 - vignette_amp * (r_norm ** 2)
        vignette = np.clip(vignette, 1.0 - vignette_amp, 1.0)
        img = np.clip(img * vignette[..., None], 0.0, 1.0)

    # 9) S-curve / midtone contrast
    s = rng.uniform(*params.midtone_contrast_range)
    img = _apply_s_curve(img, s)

    # 10) shadow lift
    shadow_lift = rng.uniform(*params.shadow_lift_range)
    img = _lift_shadows(img, shadow_lift)

    # 11) highlight compression
    highlight_rolloff = rng.uniform(*params.highlight_rolloff_range)
    img = _compress_highlights(img, highlight_rolloff)

    # 12) gamma
    gamma = rng.uniform(*params.gamma_range)
    img = np.clip(img, 1e-6, 1.0) ** gamma

    # 13) clipping event
    if rng.random() < params.clip_prob:
        gain = rng.uniform(1.03, 1.25)
        img = np.clip(img * gain, 0.0, 1.0)

    # 14) resampling artifacts
    if rng.random() < params.resize_prob:
        resize_scale = rng.uniform(*params.resize_scale_range)
        h2 = max(8, int(round(H * resize_scale)))
        w2 = max(8, int(round(W * resize_scale)))
        tmp = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        img = cv2.resize(tmp, (W, H), interpolation=cv2.INTER_LINEAR)
        img = np.clip(img, 0.0, 1.0)

    # 15) blur + sharpen
    sigma = rng.uniform(*params.blur_sigma_range)
    sharpen_strength = rng.uniform(*params.sharpen_strength_range)

    if sigma > 0.0 or sharpen_strength > 0.0:
        tmp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        tmp_bgr = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

        if sigma > 0.0:
            ksize = max(3, int(2 * round(sigma) + 1))
            blur_bgr = cv2.GaussianBlur(tmp_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        else:
            blur_bgr = tmp_bgr

        sharp_bgr = cv2.addWeighted(
            tmp_bgr, 1.0 + sharpen_strength,
            blur_bgr, -sharpen_strength,
            0
        )
        sharp_rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB)
        img = np.clip(sharp_rgb.astype(np.float32) / 255.0, 0.0, 1.0)

    # 16) per-channel noise
    noise_std_base = rng.uniform(*params.noise_std_base_range)
    if noise_std_base > 0:
        noise_std = np.array([
            noise_std_base * rng.uniform(0.8, 1.2),
            noise_std_base * rng.uniform(0.8, 1.2),
            noise_std_base * rng.uniform(1.0, 1.4),
        ], dtype=np.float32)
        noise = rng.normal(0.0, 1.0, size=img.shape).astype(np.float32)
        img = np.clip(img + noise * noise_std[None, None, :], 0.0, 1.0)

    # 17) JPEG
    if rng.random() < params.jpeg_prob:
        tmp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        tmp_bgr = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
        quality = int(rng.integers(params.jpeg_quality_range[0], params.jpeg_quality_range[1] + 1))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, enc = cv2.imencode(".jpg", tmp_bgr, encode_param)
        if ok:
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.clip(img, 0.0, 1.0)

    # 18) soft histogram band match
    if (
        params.use_histogram_match
        and quantile_band_cache is not None
        and style_name in quantile_band_cache.get("devices", {})
    ):
        hist_strength = rng.uniform(*params.histogram_match_strength_range)
        if hist_strength > 0:
            img = apply_device_quantile_band_match(
                img=img,
                target_device=style_name,
                quantile_band_cache=quantile_band_cache,
                strength=float(hist_strength),
                preserve_input_layout=True,
            )

    # 19) optional per-channel median correction
    if (
        params.use_histogram_match
        and quantile_band_cache is not None
        and style_name in quantile_band_cache.get("devices", {})
    ):
        median_strength = getattr(params, "median_match_strength", 0.0)
        if median_strength > 0:
            if style_name == "microscope":
                channel_strength = np.array([0.35, 0.35, 1.00], dtype=np.float32)
            else:
                channel_strength = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            img = _apply_channel_median_match(
                img=img,
                target_device=style_name,
                quantile_band_cache=quantile_band_cache,
                strength=float(median_strength),
                per_channel_strength=channel_strength,
            )

    # 20) final device-specific histogram polish
    img = _apply_device_histogram_polish(
        img=img,
        style_name=style_name,
        rng=rng,
    )

    return img.astype(np.float32)
