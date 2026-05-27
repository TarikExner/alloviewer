import json
import h5py
import numpy as np
import matplotlib.pyplot as plt


def vis_tile_and_heads(
    h5_path: str,
    image_idx: int = 0,
    tile_idx: int = 0,
    head_names=("cell", "boundary", "center", "energy"),
    figsize=(18, 4),
    image_percentile_clip=(1, 99),
    cmap="magma",
):
    """
    Visualize one image tile together with its 4 UNet target heads.

    Parameters
    ----------
    h5_path:
        Path to the .h5 file.

    image_idx:
        Index along the image/sample axis.

    tile_idx:
        Index along the tile axis.

    head_names:
        Names for the 4 target channels in /tgts.

    figsize:
        Figure size passed to matplotlib.

    image_percentile_clip:
        Percentile range used to display the RGB image robustly.
        Set to None to only clip image values to [0, 1].

    cmap:
        Colormap for target heads.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        for key in ("imgs", "tgts"):
            if key not in f:
                raise KeyError(f"Missing dataset '{key}' in HDF5 file.")

        imgs = f["imgs"]
        tgts = f["tgts"]

        if imgs.ndim != 5:
            raise ValueError(f"Expected /imgs shape [N,T,C,H,W], got {imgs.shape}")

        if tgts.ndim != 5:
            raise ValueError(f"Expected /tgts shape [N,T,C,H,W], got {tgts.shape}")

        n_images, t_max, c_img, h, w = imgs.shape
        _, _, c_tgt, _, _ = tgts.shape

        if image_idx < 0 or image_idx >= n_images:
            raise IndexError(f"image_idx={image_idx} out of range. Valid: 0..{n_images - 1}")

        # Determine real tile count from meta when available.
        t_real = t_max
        tile_info = None

        if "meta" in f:
            meta_raw = f["meta"][image_idx]
            if isinstance(meta_raw, bytes):
                meta = json.loads(meta_raw.decode("utf-8"))
            else:
                meta = json.loads(meta_raw)

            tiles_meta = meta.get("tiles", [])
            if tiles_meta:
                t_real = len(tiles_meta)
                if 0 <= tile_idx < t_real:
                    tile_info = tiles_meta[tile_idx]
        else:
            meta = None

        if tile_idx < 0 or tile_idx >= t_real:
            raise IndexError(
                f"tile_idx={tile_idx} out of range for image_idx={image_idx}. "
                f"Valid: 0..{t_real - 1}"
            )

        if c_tgt < 4:
            raise ValueError(f"Expected at least 4 target heads in /tgts, got {c_tgt}")

        img = np.asarray(imgs[image_idx, tile_idx], dtype=np.float32)      # [3,H,W]
        heads = np.asarray(tgts[image_idx, tile_idx, :4], dtype=np.float32) # [4,H,W]

    # Convert image from [C,H,W] to [H,W,C]
    if c_img == 1:
        img_disp = img[0]
    elif c_img >= 3:
        img_disp = np.moveaxis(img[:3], 0, -1)
    else:
        raise ValueError(f"Unsupported image channel count: {c_img}")

    # Robust display scaling for image.
    if image_percentile_clip is not None:
        lo, hi = np.percentile(img_disp[np.isfinite(img_disp)], image_percentile_clip)
        if hi > lo:
            img_disp = (img_disp - lo) / (hi - lo)

    img_disp = np.clip(img_disp, 0.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)

    axes[0].imshow(img_disp, cmap="gray" if c_img == 1 else None)
    axes[0].set_title(f"Image\nidx={image_idx}, tile={tile_idx}")
    axes[0].axis("off")

    for j in range(4):
        ax = axes[j + 1]
        im = ax.imshow(heads[j], cmap=cmap)
        ax.set_title(head_names[j] if j < len(head_names) else f"head {j}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if tile_info is not None:
        tile_xy = tile_info.get("tile_xy", None)
        tile_hw = tile_info.get("tile_hw", None)
        fig.suptitle(
            f"H5 tile visualization | image_idx={image_idx}, tile_idx={tile_idx}, "
            f"tile_xy={tile_xy}, tile_hw={tile_hw}",
            y=1.08,
        )
    else:
        fig.suptitle(
            f"H5 tile visualization | image_idx={image_idx}, tile_idx={tile_idx}",
            y=1.08,
        )

    return fig, axes
