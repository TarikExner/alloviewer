import numpy as np
from skimage import morphology
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import cv2

from .utils import (
    assign_cluster_ids,
    capture_params,
    place_clustered_centers,
    sample_calibrated_cell_diameters,
    make_wormy_dirt_patch
)
from ..utils import (
    make_ellipse_mask,
    make_soft_boundary_from_instances,
    render_mask_derived_cell,
)


_BOUNDARY_INNER_WIDTH = 1
_BOUNDARY_OUTER_WIDTH = 2
_BOUNDARY_SIGMA = 1.0

_CELL_RENDER_HALO_WEIGHT = 0.0
_CELL_RENDER_HALO_SIGMA_FACTOR = 1.0
_CELL_RENDER_MIN_SIGMA = 0.10



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

    # --- multiplicative background texture inside the well ---
    background_texture_enable=True,
    background_texture_sigma_fine=0.7,
    background_texture_sigma_coarse=4.0,
    background_texture_fine_weight=0.85,
    background_texture_coarse_weight=0.15,
    background_texture_strength=0.05,
    background_texture_clip=(0.1, 1.6),
    background_texture_downsample = 4,
    background_texture_fullres_fine_strength = 0.005,

    # --- cells (sharp ones inside the well) ---
    n_cells=150,

    # Image-size-calibrated core-diameter model. Each row is:
    #   (image short side in px, minimum core diameter, maximum core diameter)
    # Values outside the measured image-size interval are clamped.
    cell_diameter_bounds_by_short_side=(
        (1620.0, 5.0, 8.0),
        (3024.0, 11.0, 14.0),
    ),
    cell_diameter_center_margin_frac=0.20,
    cell_diameter_sigma_frac=0.18,
    cell_diameter_min_sigma_px=0.25,

    # Legacy diameter model, used only when
    # cell_diameter_bounds_by_short_side is None.
    cell_diameter=20,
    cell_diameter_reference_short_side=1620.0,
    cell_diameter_size_exponent=0.75,
    cell_diameter_scale_clip=(0.85, 2.0),

    # In calibrated mode, large cells are sampled from the upper part of the
    # valid diameter interval. They never exceed the image-specific limit.
    large_cell_frac=0.0,
    large_cell_diameter_factor=1.5,  # legacy mode only

    # --- cells: shape + brightness ---
    cell_ellipse_enable=True,
    cell_axis_jitter=0.20,          # +/-20% axis ratio
    cell_random_rotation=True,      # random rotation angle
    cell_intensity_range=(0.70, 1.05),

    frac_positive=0.5,
    color_jitter=0.07,
    sigma_in=(0.5, 1.0),
    sigma_out=(1.6, 3.2),    # used if focus_frac_in<1
    focus_frac_in=1.0,       # default: draw sharp; ghosts carry the blur
    in_focus_sigma_thresh=None,
    boundary_width=1,        # deprecated; kept for backward compatibility, ignored

    # crowd cells near the *outer* wall, but keep a filled center
    rim_bias=0.85,
    rim_band=0.12,
    edge_clamp=0.65,

    # --- clustering ---
    cluster_enable=True,
    clustered_cell_frac=0.35,
    cluster_size_range=(2, 8),
    cluster_contact_factor_range=(0.95, 1.08),
    cluster_core_min_sep_factor=0.80,
    cluster_chain_probability=0.70,
    cluster_angle_jitter=0.65,

    # Explicit cluster geometry. Small clusters are usually lengthy; larger
    # clusters are increasingly likely to use compact packed placement.
    cluster_packed_probability=0.55,
    cluster_packed_size_bias_range=(3, 15),
    cluster_packed_contact_factor_range=(0.88, 1.02),
    cluster_packed_candidate_count=8,
    cluster_packed_contact_bonus=1.50,

    # Packed clusters may seed beside an earlier packed cluster. Separate
    # cluster IDs can then form one larger packed region without core overlap.
    cluster_packed_region_join_probability=0.30,
    cluster_packed_region_contact_factor_range=(0.95, 1.12),

    cluster_seed_tries=120,
    cluster_member_tries=24,
    cluster_pack_min_sep_factor=0.84,

    # --- collision / packing control ---
    min_cell_sep_px=None,   # if None -> pair-specific radius-based separation
    rim_min_sep_px=4,
    pack_iters=20,
    pack_strength=0.45,
    wall_margin_px=2.0,

    # --- sidedness (pile-up on one side near the rim) ---
    side_bias_enable=False,
    side_bias_theta=0.0,
    side_bias_strength=0.75,
    side_bias_kappa=5.0,
    side_bias_inner_frac=0.55,

    # --- visual wall (soft rim) ---
    wall_blur_sigma=12.0,
    ring_artifacts=0,
    ring_sigma_range=(6.0, 18.0),
    ring_alpha_range=(0.03, 0.12),

    # --- "ghost cells" OUTSIDE the well (big, elongated, not in masks) ---
    ghost_enable=True,
    ghost_density=0.50,
    ghost_offset_px=10.0,
    ghost_offset_jitter=6.0,
    ghost_sigma=(2.5, 6.0),
    ghost_dilate=1.0,
    ghost_intensity=(0.8, 1.4),

    ghost_stretch=3.0,
    ghost_trail=3,
    ghost_trail_decay=0.6,

    # --- debris INSIDE the well (small + dim) ---
    dirt_density=0.0007,
    dirt_size=(2, 4),
    dirt_sigma=(1.2, 2.0),
    dirt_alpha=(0.01, 0.04),

    # --- radial reflections on the wall (outside the well) ---
    reflect_enable=True,
    reflect_n=6,
    reflect_theta_sigma=0.10,
    reflect_radial_sigma=8.0,
    reflect_offset_range=(6.0, 24.0),
    reflect_alpha_range=(0.05, 0.20),
    reflect_wobble=0.35,
    reflect_harmonics=2,
    reflect_harmonic_decay=0.55,

    seed=None,
    return_targets=True,
    return_aux_targets=False
):
    """
    Returns:
      image: (H, W, 3) float32 in [0,1]
      meta: dict with centers, labels, well_center, radius_px, final_sigmas
      targets, if return_targets:
        instance_labels: int32 (H,W), all true inside-well cell instances
        cell_mask: float32 (H,W), all true inside-well cells
        boundary: float32 (H,W), soft inside/outside boundary ring around instance_labels

        optional debug/audit entries:
          instance_labels_sigma_filtered: int32 (H,W), sigma-thresholded subset
          cell_mask_sigma_filtered: float32 (H,W), sigma-thresholded subset
          final_sigmas: float32 (n_cells,), sampled render sigmas
          focus_keep_ids: int32 array, labels kept by sigma threshold

    Notes:
      - Visible cells are rendered from the final instance mask, not from an
        independent Gaussian spot. This keeps the rendered cell body and label
        geometry coupled.
      - Core diameters are sampled from image-size-dependent measured bounds.
        The old power-law scaling remains available only as a fallback.
      - Clusters explicitly use lengthy or packed placement. Packed clusters
        score a small local candidate set and may share a larger packed region.
        Global overlap resolution still uses a KD-tree.
      - The base render halo can be disabled; camera/acquisition halo may be added later.
      - Boundary target settings are internal constants, not user parameters.
      - focus_frac_in still affects sampled render sigmas, but the standard
        training targets include all true inside-well cells. The sigma-filtered
        outputs are debug/audit only.
    """
    rng = np.random.default_rng(seed)
    H = int(H)
    W = int(W)
    n_cells = int(n_cells)

    # ---------- fast blur helper (SciPy) ----------
    def blur(a, sigma):
        if sigma is None or sigma <= 0:
            return a
        if a.ndim == 2:
            return ndi.gaussian_filter(a, sigma=float(sigma), mode="reflect")
        if a.ndim == 3:
            return ndi.gaussian_filter(a, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
        return a

    # ---------- helpers ----------
    def jitter_color(base_rgb):
        j = rng.normal(1.0, color_jitter, size=3).astype(np.float32)
        c = (base_rgb * j).clip(0, 1)
        scale = np.linalg.norm(base_rgb) / max(1e-6, np.linalg.norm(c))
        return (c * scale).clip(0, 1)

    base_orange = np.array([1.00, 0.62, 0.08], dtype=np.float32)
    base_green = np.array([0.05, 0.95, 0.35], dtype=np.float32)
    bg_color = (0.75 * base_orange + 0.25 * base_green).astype(np.float32)

    # ---------- grids / well ----------
    yy, xx = np.mgrid[0:H, 0:W]
    yy = yy.astype(np.float32, copy=False)
    xx = xx.astype(np.float32, copy=False)

    cy = np.float32(0.5 * H + rng.normal(0, well_center_jitter * min(H, W)))
    cx = np.float32(0.5 * W + rng.normal(0, well_center_jitter * min(H, W)))
    R = np.float32(well_radius_frac * min(H, W))

    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dy * dy + dx * dx).astype(np.float32)

    inside = rr <= R
    r_norm = np.clip(rr / (R + np.float32(1e-8)), 0, 1).astype(np.float32)

    illum = np.power(r_norm, np.float32(radial_gamma)).astype(np.float32)

    bg_inside = (background_level + edge_boost * illum).astype(np.float32)
    img = (bg_inside[..., None] * bg_color[None, None, :]).astype(np.float32)

    # vignette
    ry = ((yy - np.float32(0.5 * H)) / np.float32(0.5 * H)).astype(np.float32)
    rx = ((xx - np.float32(0.5 * W)) / np.float32(0.5 * W)).astype(np.float32)
    v = np.sqrt(rx * rx + ry * ry).astype(np.float32)

    vignette = (
        np.float32(1.0)
        - np.float32(vignette_strength) * (v * v)
    ).astype(np.float32)

    vignette = np.clip(vignette, np.float32(0.15), np.float32(1.0)).astype(np.float32)
    img *= vignette[..., None]

    # soft wall glow (background only)
    rim = np.exp(
        -((rr - R) * (rr - R)) / np.float32(2.0 * 0.8 * 0.8)
    ).astype(np.float32)
    rim = blur(rim, wall_blur_sigma)
    rim = rim / (rim.max() + 1e-8)
    img += (0.06 * rim)[..., None] * bg_color[None, None, :]

    if ring_artifacts > 0:
        rings = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(ring_artifacts)):
            r_shift = rng.uniform(-0.04, 0.06) * R
            sig = rng.uniform(*ring_sigma_range)
            a = rng.uniform(*ring_alpha_range)
            r_center = np.float32(R + np.float32(r_shift))
            ring = np.exp(
                -((rr - r_center) * (rr - r_center)) / np.float32(2.0)
            ).astype(np.float32)
            ring = blur(ring, sig)
            ring /= ring.max() + 1e-8
            rings += a * ring
        img += rings[..., None] * bg_color[None, None, :]

    # multiplicative background texture inside the well
    if background_texture_enable and background_texture_strength > 0:
        ds = int(max(1, background_texture_downsample))

        if ds > 1:
            h_tex = max(8, int(np.ceil(H / ds)))
            w_tex = max(8, int(np.ceil(W / ds)))

            tex1 = rng.normal(0, 1.0, size=(h_tex, w_tex)).astype(np.float32)
            tex2 = rng.normal(0, 1.0, size=(h_tex, w_tex)).astype(np.float32)

            # Convert full-res sigma to low-res sigma.
            sig_fine = max(0.0, float(background_texture_sigma_fine) / float(ds))
            sig_coarse = max(0.0, float(background_texture_sigma_coarse) / float(ds))

            tex1 = blur(tex1, sig_fine).astype(np.float32, copy=False)
            tex2 = blur(tex2, sig_coarse).astype(np.float32, copy=False)

            fine_w = np.float32(background_texture_fine_weight)
            coarse_w = np.float32(background_texture_coarse_weight)
            denom_w = np.float32(abs(float(fine_w)) + abs(float(coarse_w)))

            if denom_w <= np.float32(1e-8):
                fine_w = np.float32(1.0)
                coarse_w = np.float32(0.0)
                denom_w = np.float32(1.0)

            tex_small = ((fine_w * tex1 + coarse_w * tex2) / denom_w).astype(np.float32)

            tex = cv2.resize(
                tex_small,
                (W, H),
                interpolation=cv2.INTER_CUBIC,
            ).astype(np.float32)

            # Optional tiny full-res fine irregularity.
            # This is cheap compared with full-res Gaussian blur.
            if background_texture_fullres_fine_strength > 0:
                fine = rng.normal(0, 1.0, size=(H, W)).astype(np.float32)
                tex = tex + np.float32(background_texture_fullres_fine_strength) * fine

        else:
            tex1 = rng.normal(0, 1.0, size=(H, W)).astype(np.float32)
            tex2 = rng.normal(0, 1.0, size=(H, W)).astype(np.float32)

            tex1 = blur(tex1, background_texture_sigma_fine).astype(np.float32, copy=False)
            tex2 = blur(tex2, background_texture_sigma_coarse).astype(np.float32, copy=False)

            fine_w = np.float32(background_texture_fine_weight)
            coarse_w = np.float32(background_texture_coarse_weight)
            denom_w = np.float32(abs(float(fine_w)) + abs(float(coarse_w)))

            if denom_w <= np.float32(1e-8):
                fine_w = np.float32(1.0)
                coarse_w = np.float32(0.0)
                denom_w = np.float32(1.0)

            tex = ((fine_w * tex1 + coarse_w * tex2) / denom_w).astype(np.float32)

        tex = (tex - np.float32(tex.mean())) / (np.float32(tex.std()) + np.float32(1e-6))

        lo, hi = background_texture_clip
        tex = np.clip(
            np.float32(1.0) + np.float32(background_texture_strength) * tex,
            np.float32(lo),
            np.float32(hi),
        ).astype(np.float32)

        img[inside] *= tex[inside, None]

    # ---------- sample cell labels and calibrated core diameters ----------
    n_pos = int(round(frac_positive * n_cells))
    labels = np.array([1] * n_pos + [0] * (n_cells - n_pos), dtype=np.int32)
    rng.shuffle(labels)

    short_side = np.float32(min(H, W))
    large_cell_frac = float(np.clip(large_cell_frac, 0.0, 1.0))

    if cell_diameter_bounds_by_short_side is not None:
        diameters, base_cell_diameter, diameter_bounds, is_large = (
            sample_calibrated_cell_diameters(
                rng=rng,
                n_cells=n_cells,
                short_side=float(short_side),
                anchors=cell_diameter_bounds_by_short_side,
                center_margin_frac=cell_diameter_center_margin_frac,
                cell_sigma_frac=cell_diameter_sigma_frac,
                min_sigma_px=cell_diameter_min_sigma_px,
                large_cell_frac=large_cell_frac,
            )
        )
        cell_diameter_mode = "calibrated_bounds"
        cell_diameter_size_scale = None
    else:
        # Backward-compatible power-law scaling. Final values are not bounded
        # unless calibrated anchors are supplied.
        ref_short_side = np.float32(
            max(1.0, float(cell_diameter_reference_short_side))
        )
        size_exponent = np.float32(cell_diameter_size_exponent)
        cell_diameter_size_scale = np.float32(
            (short_side / ref_short_side) ** size_exponent
        )

        if cell_diameter_scale_clip is not None:
            lo, hi = cell_diameter_scale_clip
            cell_diameter_size_scale = np.float32(
                np.clip(cell_diameter_size_scale, np.float32(lo), np.float32(hi))
            )

        base_cell_diameter = np.float32(cell_diameter) * cell_diameter_size_scale
        is_large = rng.random(n_cells) < large_cell_frac
        diameters = np.full(n_cells, base_cell_diameter, dtype=np.float32)
        diameters[is_large] *= float(large_cell_diameter_factor)

        jitter = rng.normal(1.0, 0.10, size=n_cells).astype(np.float32)
        jitter = np.clip(jitter, 0.8, 1.2)
        diameters *= jitter

        diameter_bounds = (float(diameters.min()), float(diameters.max()))
        cell_diameter_mode = "legacy_power_law"

    radii = np.maximum(2, np.round(diameters / 2.0).astype(np.int32))

    # ---------- per-cell ellipse params ----------
    if cell_ellipse_enable:
        axis_ratio = rng.uniform(
            1.0 - cell_axis_jitter,
            1.0 + cell_axis_jitter,
            size=n_cells,
        ).astype(np.float32)
        axis_ratio = np.clip(
            axis_ratio,
            1.0 - cell_axis_jitter,
            1.0 + cell_axis_jitter,
        )
        if cell_random_rotation:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=n_cells).astype(np.float32)
        else:
            theta = np.zeros(n_cells, dtype=np.float32)
    else:
        axis_ratio = np.ones(n_cells, dtype=np.float32)
        theta = np.zeros(n_cells, dtype=np.float32)

    # Conservative radius used only for placement and collision checks.
    r_px = radii.astype(np.float32) * np.sqrt(
        np.maximum(axis_ratio, 1.0 / axis_ratio)
    ).astype(np.float32)

    def sample_radius():
        a = (1.0 - rim_band) * R
        if rng.random() < rim_bias:
            r = np.sqrt(a * a + rng.random() * (R * R - a * a))
            if edge_clamp > 0:
                r = (1.0 - edge_clamp) * r + edge_clamp * R
                r -= rng.uniform(0.0, 0.02 * rim_band * R)
        else:
            r = a * np.sqrt(rng.random())
        return float(np.clip(r, 0.05 * R, 0.985 * R))

    def sample_theta(r):
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
        return rng.uniform(-np.pi, np.pi)

    # ---------- cluster assignment and initial placement ----------
    if cluster_enable and n_cells > 1 and clustered_cell_frac > 0:
        cluster_ids = assign_cluster_ids(
            rng=rng,
            n_cells=n_cells,
            clustered_fraction=clustered_cell_frac,
            cluster_size_range=cluster_size_range,
        )
    else:
        cluster_ids = np.full(n_cells, -1, dtype=np.int32)

    (
        cf,
        cluster_ids,
        cluster_edges,
        cluster_modes,
        cluster_region_ids,
    ) = place_clustered_centers(
        rng=rng,
        cluster_ids=cluster_ids,
        r_px=r_px,
        H=H,
        W=W,
        cy=float(cy),
        cx=float(cx),
        R=float(R),
        wall_margin_px=float(wall_margin_px),
        rim_band=float(rim_band),
        rim_min_sep_px=float(rim_min_sep_px),
        sample_radius=sample_radius,
        sample_theta=sample_theta,
        contact_factor_range=cluster_contact_factor_range,
        core_min_sep_factor=cluster_core_min_sep_factor,
        chain_probability=cluster_chain_probability,
        angle_jitter=cluster_angle_jitter,
        packed_probability=cluster_packed_probability,
        packed_size_bias_range=cluster_packed_size_bias_range,
        packed_contact_factor_range=cluster_packed_contact_factor_range,
        packed_candidate_count=cluster_packed_candidate_count,
        packed_contact_bonus=cluster_packed_contact_bonus,
        packed_region_join_probability=cluster_packed_region_join_probability,
        packed_region_contact_factor_range=(
            cluster_packed_region_contact_factor_range
        ),
        seed_tries=cluster_seed_tries,
        member_tries=cluster_member_tries,
    )

    # ---------- resolve overlaps: cluster-aware KD-tree packing ----------
    eps = np.float32(1e-6)
    N = int(cf.shape[0])
    regular_pack_min_sep_factor = np.float32(0.90)
    cluster_pack_min_sep_factor = np.float32(
        np.clip(cluster_pack_min_sep_factor, 0.0, 0.90)
    )

    if N > 1 and int(pack_iters) > 0:
        if min_cell_sep_px is None:
            query_factor = max(
                float(regular_pack_min_sep_factor),
                float(cluster_pack_min_sep_factor),
            )
            query_radius = np.float32(
                query_factor * 2.0 * float(np.max(r_px))
            )
        else:
            min_sep_scalar = np.float32(min_cell_sep_px)
            query_radius = np.float32(
                max(
                    float(min_sep_scalar),
                    2.0 * float(cluster_pack_min_sep_factor) * float(np.max(r_px)),
                )
            )

        query_radius = np.float32(max(float(query_radius), 1.0))

        for _ in range(int(pack_iters)):
            tree = cKDTree(cf)
            pairs = tree.query_pairs(r=float(query_radius), output_type="ndarray")

            if pairs.size == 0:
                break

            i = pairs[:, 0].astype(np.int32, copy=False)
            j = pairs[:, 1].astype(np.int32, copy=False)

            v = cf[i] - cf[j]
            d = np.sqrt((v * v).sum(axis=1)).astype(np.float32)
            pair_radius = (r_px[i] + r_px[j]).astype(np.float32)
            same_cluster = (
                (cluster_ids[i] >= 0)
                & (cluster_ids[i] == cluster_ids[j])
            )

            if min_cell_sep_px is None:
                separation_factor = np.where(
                    same_cluster,
                    cluster_pack_min_sep_factor,
                    regular_pack_min_sep_factor,
                ).astype(np.float32)
                min_sep = separation_factor * pair_radius
            else:
                min_sep = np.full_like(d, min_sep_scalar, dtype=np.float32)
                if np.any(same_cluster):
                    min_sep[same_cluster] = np.minimum(
                        min_sep[same_cluster],
                        cluster_pack_min_sep_factor * pair_radius[same_cluster],
                    )

            overlap_mask = d < min_sep
            if not np.any(overlap_mask):
                break

            i = i[overlap_mask]
            j = j[overlap_mask]
            v = v[overlap_mask]
            d = d[overlap_mask]
            min_sep = min_sep[overlap_mask]

            u = v / (d[:, None] + eps)
            overlap = (min_sep - d).astype(np.float32)
            step = np.float32(pack_strength * 0.5) * overlap[:, None] * u

            disp = np.zeros_like(cf, dtype=np.float32)
            np.add.at(disp, i, step)
            np.add.at(disp, j, -step)
            cf += disp

            cf[:, 0] = np.clip(cf[:, 0], r_px + 1, H - r_px - 2)
            cf[:, 1] = np.clip(cf[:, 1], r_px + 1, W - r_px - 2)

            vy = (cf[:, 0] - cy).astype(np.float32, copy=False)
            vx = (cf[:, 1] - cx).astype(np.float32, copy=False)
            rr_c = np.sqrt(vy * vy + vx * vx).astype(np.float32) + eps

            max_r_center = (
                R - np.float32(wall_margin_px) - r_px
            ).astype(np.float32)
            max_r_center = np.maximum(max_r_center, np.float32(0.0))

            too_far = rr_c > max_r_center
            if np.any(too_far):
                scale = (
                    max_r_center[too_far] / rr_c[too_far]
                ).astype(np.float32)
                cf[too_far, 0] = cy + vy[too_far] * scale
                cf[too_far, 1] = cx + vx[too_far] * scale

    centers = [(int(round(y)), int(round(x))) for y, x in cf]
    radial_centers = np.sqrt(
        (cf[:, 0] - cy) ** 2 + (cf[:, 1] - cx) ** 2
    )
    is_rim = (radial_centers >= (1.0 - rim_band) * R).tolist()

    # ---------- instance map ----------
    # Build final labels first. Rendering later uses this final map, so visible
    # cells and labels share geometry even in overlap cases.
    inst = np.zeros((H, W), dtype=np.int32)
    realized_equivalent_diameters = np.zeros(n_cells, dtype=np.float32)

    for k_id, (y, x) in enumerate(centers, start=1):
        r = int(radii[k_id - 1])
        m, r_box = make_ellipse_mask(
            r,
            float(axis_ratio[k_id - 1]),
            float(theta[k_id - 1]),
        )
        realized_equivalent_diameters[k_id - 1] = np.float32(
            2.0 * np.sqrt(float(m.sum()) / np.pi)
        )

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

    # ---------- render cells from final instance masks ----------
    final_sigmas = np.zeros(n_cells, dtype=np.float32)
    final_sigma_fractions = np.zeros(n_cells, dtype=np.float32)

    for zero_i, (y, x) in enumerate(centers):
        k_id = zero_i + 1

        base_col = (
            base_orange
            if labels[zero_i] == 1
            else base_green
        )
        col = jitter_color(base_col)

        d0 = float(diameters[zero_i])

        # sigma_in and sigma_out are fractions of cell-core diameter.
        if rng.random() < focus_frac_in:
            sigma_fraction = float(rng.uniform(*sigma_in))
        else:
            sigma_fraction = float(rng.uniform(*sigma_out))

        sig = max(
            _CELL_RENDER_MIN_SIGMA,
            d0 * sigma_fraction,
        )

        final_sigma_fractions[zero_i] = sigma_fraction
        final_sigmas[zero_i] = sig

        render_radius = int(
            np.ceil(
                max(
                    2,
                    0.5 * d0
                    + 3.0 * sig * _CELL_RENDER_HALO_SIGMA_FACTOR,
                )
            )
        )

        y0, y1 = y - render_radius, y + render_radius + 1
        x0, x1 = x - render_radius, x + render_radius + 1

        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)

        if y1c <= y0c or x1c <= x0c:
            continue

        sl = (
            slice(y0c, y1c),
            slice(x0c, x1c),
        )

        core_mask = inst[sl] == k_id

        if not np.any(core_mask):
            continue

        render = render_mask_derived_cell(
            core_mask,
            sigma=sig,
            halo_weight=_CELL_RENDER_HALO_WEIGHT,
            halo_sigma_factor=_CELL_RENDER_HALO_SIGMA_FACTOR,
            min_sigma=_CELL_RENDER_MIN_SIGMA,
        )

        amp = float(rng.uniform(*cell_intensity_range))

        img[sl + (slice(None),)] += (
            amp
            * render[..., None]
            * col[None, None, :]
        )

    # ---------- elongated ghosts OUTSIDE the wall (not in masks) ----------
    def draw_elliptical_gaussian(dst, gy, gx, sig_minor, stretch, angle_rad, amp, col, ref_diameter):
        sig_x = sig_minor * stretch
        sig_y = sig_minor
        radius = int(np.ceil(ghost_dilate * (0.5 * float(ref_diameter) + 3 * max(sig_x, sig_y))))
        yy_l, xx_l = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        ca, sa = np.cos(angle_rad), np.sin(angle_rad)
        xr = ca * xx_l + sa * yy_l
        yr = -sa * xx_l + ca * yy_l
        g = np.exp(-0.5 * ((xr / sig_x) ** 2 + (yr / sig_y) ** 2)).astype(np.float32)
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

                step = max(2.0, 0.6 * float(diameters[i]))
                amp = amp0 * ghost_trail_decay
                for t in range(1, int(ghost_trail)):
                    gy = int(round(base_y + t * step * np.sin(ang)))
                    gx = int(round(base_x + t * step * np.cos(ang)))
                    sig_t = sig0 * (1.0 + 0.4 * t)
                    draw_elliptical_gaussian(img, gy, gx, sig_t, ghost_stretch, ang, amp, col, diameters[i])
                    amp *= ghost_trail_decay

    # --- RADIAL REFLECTIONS (outside the well) ---
    if reflect_enable and reflect_n > 0:
        ang = np.arctan2(yy - cy, xx - cx).astype(np.float32)

        def angle_wrap(a):
            return ((a + np.float32(np.pi)) % np.float32(2.0 * np.pi) - np.float32(np.pi)).astype(np.float32)

        refl = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(reflect_n)):

            theta0 = np.float32(rng.uniform(-np.pi, np.pi))
            theta0 = np.float32(theta0 + rng.normal(0, reflect_wobble))

            r_off = np.float32(rng.uniform(*reflect_offset_range))
            alpha = np.float32(rng.uniform(*reflect_alpha_range))

            rad_sig = np.float32(reflect_radial_sigma)
            theta_sig = np.float32(reflect_theta_sigma)
            r_center = np.float32(R + r_off)

            radial_term = np.exp(
                -((rr - r_center) * (rr - r_center)) / np.float32(2.0 * rad_sig * rad_sig)
            ).astype(np.float32)

            dtheta = angle_wrap(ang - theta0)
            angular_term = np.exp(
                -(dtheta * dtheta) / np.float32(2.0 * theta_sig * theta_sig)
            ).astype(np.float32)

            base = radial_term * angular_term
            comb = base.copy()
            th_sig = np.float32(reflect_theta_sigma)
            harm_decay = np.float32(reflect_harmonic_decay)

            for h in range(1, int(reflect_harmonics) + 1):
                decay = np.float32(harm_decay ** h)
                offset = np.float32(h) * th_sig * np.float32(2.0)

                dtheta_p = angle_wrap(ang - (theta0 + offset))
                dtheta_m = angle_wrap(ang - (theta0 - offset))

                term_p = np.exp(
                    -(dtheta_p * dtheta_p) / np.float32(2.0 * th_sig * th_sig)
                ).astype(np.float32)

                term_m = np.exp(
                    -(dtheta_m * dtheta_m) / np.float32(2.0 * th_sig * th_sig)
                ).astype(np.float32)

                comb += (decay * term_p * radial_term).astype(np.float32)
                comb += (decay * term_m * radial_term).astype(np.float32)

            refl += alpha * comb

        outside = rr > R
        refl = np.clip(refl / (refl.max() + 1e-8), 0, 1)
        img[outside, 0] += (refl[outside] * bg_color[0] * 0.9).astype(np.float32)
        img[outside, 1] += (refl[outside] * bg_color[1] * 0.9).astype(np.float32)
        img[outside, 2] += (refl[outside] * bg_color[2] * 0.9).astype(np.float32)

    # ---------- debris (small, dim, worm-like) inside the well only ----------
    inside_idx = np.flatnonzero(inside.ravel())
    if inside_idx.size > 0:
        n_dirt = int(inside.sum() * float(dirt_density))

        for _ in range(n_dirt):
            idx = int(rng.choice(inside_idx))
            ry_d, rx_d = divmod(idx, W)

            base_size = int(rng.integers(dirt_size[0], dirt_size[1] + 1))

            alpha_map, col = make_wormy_dirt_patch(
                rng=rng,
                base_size=base_size,
                dirt_sigma=dirt_sigma,
                dirt_alpha=dirt_alpha,
                base_orange=base_orange,
                base_green=base_green,
            )

            ph, pw = alpha_map.shape
            patch_r_y = ph // 2
            patch_r_x = pw // 2

            y0, y1 = ry_d - patch_r_y, ry_d + patch_r_y + 1
            x0, x1 = rx_d - patch_r_x, rx_d + patch_r_x + 1

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

            alpha_local = alpha_map[sy0:sy1, sx0:sx1] * mask_in.astype(np.float32)
            if alpha_local.max() <= 0:
                continue

            sl = (slice(y0c, y1c), slice(x0c, x1c))
            img[sl + (slice(None),)] += alpha_local[..., None] * col[None, None, :]

    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # ---------- targets ----------
    inst_all = inst.astype(np.int32)
    cell_mask_all = (inst_all > 0).astype(np.float32)

    # Keep sigma-filter metadata cheap. Do not build full-res filtered masks unless requested.
    if in_focus_sigma_thresh is None:
        in_focus_sigma_thresh = 1.15 * max(sigma_in)

    thr = float(in_focus_sigma_thresh)
    keep_ids = (np.flatnonzero(final_sigmas <= thr) + 1).astype(np.int32)

    targets = {}

    if return_targets:
        # Minimal targets needed by SimCellsDataset.
        # Tile-level heads are rebuilt later in BaseCellsTilesDataset._finalize_tiles().
        targets = {
            "instance_labels": inst_all,
            "cell_mask": cell_mask_all,
        }

        if return_aux_targets:
            boundary = make_soft_boundary_from_instances(
                inst_all,
                ring_width=_BOUNDARY_INNER_WIDTH,
                soft_band=_BOUNDARY_OUTER_WIDTH,
                sigma=_BOUNDARY_SIGMA,
            ).astype(np.float32)

            inst_sigma_filtered = np.where(
                np.isin(inst_all, keep_ids),
                inst_all,
                0,
            ).astype(np.int32)

            targets.update(
                {
                    "boundary": boundary,
                    "instance_labels_sigma_filtered": inst_sigma_filtered,
                    "cell_mask_sigma_filtered": (inst_sigma_filtered > 0).astype(np.float32),
                    "final_sigmas": final_sigmas.astype(np.float32),
                    "focus_keep_ids": keep_ids.astype(np.int32),
                }
            )

    if min_cell_sep_px is None:
        min_cell_sep_px = 0.9 * base_cell_diameter

    all_params = capture_params(simulate_image, locals())

    meta = {
        "centers": centers,
        "labels": labels,
        "frac_positive": frac_positive,
        "n_cells": n_cells,
        "well_center": (float(cy), float(cx)),
        "radius_px": float(R),
        "base_cell_diameter": float(base_cell_diameter),
        "cell_diameter_mode": cell_diameter_mode,
        "cell_diameter_bounds": tuple(float(v) for v in diameter_bounds),
        "requested_diameters": diameters.astype(np.float32),
        "realized_equivalent_diameters": realized_equivalent_diameters,
        "is_large": is_large.astype(bool),
        "cell_diameter_size_scale": (
            None
            if cell_diameter_size_scale is None
            else float(cell_diameter_size_scale)
        ),
        "cell_diameter_reference_short_side": float(cell_diameter_reference_short_side),
        "cell_diameter_size_exponent": float(cell_diameter_size_exponent),
        "cell_diameter_scale_clip": cell_diameter_scale_clip,
        "cluster_ids": cluster_ids.astype(np.int32),
        "cluster_edges": cluster_edges,
        "cluster_modes": cluster_modes.astype(np.int8),
        "cluster_mode_codes": {
            -1: "isolated",
            0: "lengthy",
            1: "packed",
        },
        "cluster_region_ids": cluster_region_ids.astype(np.int32),
        "n_clusters": int(np.unique(cluster_ids[cluster_ids >= 0]).size),
        "n_clustered_cells": int(np.sum(cluster_ids >= 0)),
        "n_lengthy_clusters": int(
            np.unique(cluster_ids[cluster_modes == 0]).size
        ),
        "n_packed_clusters": int(
            np.unique(cluster_ids[cluster_modes == 1]).size
        ),
        "n_packed_regions": int(
            np.unique(cluster_region_ids[cluster_region_ids >= 0]).size
        ),
        "final_sigmas": final_sigmas,
        "target_keep_ids": keep_ids.astype(np.int32),
        "params": all_params,
    }

    return img, meta, targets
