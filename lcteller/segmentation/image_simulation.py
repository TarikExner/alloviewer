import numpy as np
from skimage import segmentation, morphology

def simulate_image(
    N=512,
    n_cells=150,
    cell_diameter=20,
    frac_positive=0.5,
    background_level=0.02,
    color_jitter=0.07,
    blur_sigma=1.2,
    photon_level=2500,
    seed=None,
    return_targets=True,
    boundary_width=2,
):
    """
    Returns:
      image: (N, N, 3) float32 in [0,1]
      meta: dict with centers, labels(1=orange,0=green), radius, params
      targets (if return_targets):
        {
          "instance_labels": int32 (N,N) 0=bg, 1..K cells
          "cell_mask": float32 (N,N) in {0,1}
          "boundary": float32 (N,N) in {0,1}  (inside cell union)
        }
    """
    rng = np.random.default_rng(seed)
    H = W = int(N)
    rad = max(2, int(round(cell_diameter / 2)))

    base_orange = np.array([1.00, 0.62, 0.08], dtype=np.float32)
    base_green  = np.array([0.05, 0.95, 0.35], dtype=np.float32)

    img = np.full((H, W, 3), background_level, dtype=np.float32)

    # Pure disk kernel
    yx = np.mgrid[-rad:rad+1, -rad:rad+1]
    rr = np.sqrt(yx[0]**2 + yx[1]**2)
    disk = (rr <= rad).astype(np.float32)

    # Labels (phenotype)
    n_pos = int(round(frac_positive * n_cells))
    labels = np.array([1]*n_pos + [0]*(n_cells - n_pos))
    rng.shuffle(labels)

    # Place centers (light overlap avoidance)
    centers = []
    max_tries = 50 * n_cells
    tries = 0
    while len(centers) < n_cells and tries < max_tries:
        tries += 1
        cy = rng.integers(rad+1, H - rad - 1)
        cx = rng.integers(rad+1, W - rad - 1)
        ok = True
        for (py, px) in centers[-30:]:
            if (py - cy)**2 + (px - cx)**2 < (1.2*rad)**2:
                ok = False
                break
        if ok:
            centers.append((int(cy), int(cx)))
    while len(centers) < n_cells:
        cy = rng.integers(rad+1, H - rad - 1)
        cx = rng.integers(rad+1, W - rad - 1)
        centers.append((int(cy), int(cx)))

    # jitter color
    def jitter_color(base_rgb):
        jitter = rng.normal(1.0, color_jitter, size=3).astype(np.float32)
        c = (base_rgb * jitter).clip(0, 1)
        scale = np.linalg.norm(base_rgb) / max(1e-6, np.linalg.norm(c))
        return (c * scale).clip(0, 1)

    # --- Ground-truth instance labels (nearest-center within disks) ---
    inst = np.zeros((H, W), dtype=np.int32)
    bestd = np.full((H, W), np.inf, dtype=np.float32)
    r = int(rad)
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    rr_patch = np.sqrt(yy**2 + xx**2)
    inside = rr_patch <= r + 1e-6

    for k, (cy, cx) in enumerate(centers, start=1):
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        sy0, sy1 = y0c - y0, y1c - y0
        sx0, sx1 = x0c - x0, x1c - x0

        rr_sub = rr_patch[sy0:sy1, sx0:sx1]
        inside_sub = inside[sy0:sy1, sx0:sx1]
        pd = bestd[y0c:y1c, x0c:x1c]
        better = np.logical_and(inside_sub, rr_sub < pd)
        pd[better] = rr_sub[better]
        inst[y0c:y1c, x0c:x1c][better] = k

    cell_mask = (inst > 0)

    # Boundary from instances
    boundary = segmentation.find_boundaries(inst, mode="thick")
    if boundary_width and boundary_width > 1:
        boundary = morphology.binary_dilation(boundary, morphology.disk(int(boundary_width)))
    boundary = np.logical_and(boundary, cell_mask)

    # --- Render image (flat disks), then blur+noise ---
    for k, (cy, cx) in enumerate(centers):
        col = jitter_color(base_orange if labels[k] == 1 else base_green)
        amp = float(rng.uniform(0.95, 1.05))
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        patch = amp * disk[..., None] * col[None, None, :]
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        sy0, sy1 = y0c - y0, y1c - y0
        sx0, sx1 = x0c - x0, x1c - x0
        img[y0c:y1c, x0c:x1c, :] += patch[sy0:sy1, sx0:sx1, :]

    if blur_sigma and blur_sigma > 0:
        def gaussian_kernel1d(sigma, radius=None):
            if radius is None:
                radius = int(np.ceil(3*sigma))
            x = np.arange(-radius, radius+1, dtype=np.float32)
            k = np.exp(-(x**2)/(2*sigma**2))
            k /= k.sum()
            return k
        def sep_conv(a, sigma):
            k = gaussian_kernel1d(sigma)
            pad = len(k)//2
            tmp = np.pad(a, ((0,0),(pad,pad),(0,0)), mode='reflect')
            out = np.empty_like(a)
            for i in range(a.shape[1]):
                out[:, i, :] = np.tensordot(tmp[:, i:i+len(k), :], k, axes=([1],[0]))
            tmp2 = np.pad(out, ((pad,pad),(0,0),(0,0)), mode='reflect')
            out2 = np.empty_like(a)
            for j in range(a.shape[0]):
                out2[j, :, :] = np.tensordot(tmp2[j:j+len(k), :, :], k, axes=([0],[0]))
            return out2
        img = sep_conv(img, float(blur_sigma))

    counts = (img.clip(0, 1) * photon_level).astype(np.float32)
    noised = rng.poisson(counts).astype(np.float32) / max(1.0, photon_level)
    noised += rng.normal(0.0, 0.003, size=noised.shape).astype(np.float32)
    noised = np.clip(noised, 0, 1).astype(np.float32)

    meta = {
        "centers": centers,
        "labels": labels,  # 1=orange, 0=green
        "radius": rad,
        "params": dict(
            N=N, n_cells=n_cells, cell_diameter=cell_diameter,
            frac_positive=frac_positive, seed=seed
        ),
    }

    if return_targets:
        targets = {
            "instance_labels": inst.astype(np.int32),
            "cell_mask": cell_mask.astype(np.float32),
            "boundary": boundary.astype(np.float32),
        }
    else:
        targets = {}

    return noised, meta, targets
