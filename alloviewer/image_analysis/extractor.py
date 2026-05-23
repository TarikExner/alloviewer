import numpy as np
from typing import List, Dict, Any


class RGBExtractor:
    """Extract mean RGB intensities for labeled instances.

    The extractor computes per-instance mean red, green, and blue channel
    values from an RGB image and a matching integer instance map. Label ``0`` is
    treated as background and ignored.
    """

    def __call__(self, img: np.ndarray, inst: np.ndarray) -> List[Dict[str, Any]]:
        """Compute mean RGB values per instance.

        Parameters
        ----------
        img : numpy.ndarray
            RGB image with either shape ``(H, W, 3)`` or ``(3, H, W)``.
        inst : numpy.ndarray
            Integer instance label map with shape ``(H, W)``. Label ``0`` is
            treated as background.

        Returns
        -------
        list of dict
            One dictionary per instance label with the following keys:

            ``"roi_id"``
                Instance label ID.
            ``"mean_r"``
                Mean red-channel intensity.
            ``"mean_g"``
                Mean green-channel intensity.
            ``"mean_b"``
                Mean blue-channel intensity.
            ``"area"``
                Number of pixels assigned to the instance.

        Raises
        ------
        ValueError
            If ``img`` is not a 3D RGB array, if ``inst`` is not a 2D array, or
            if the image and instance map spatial shapes do not match.

        Notes
        -----
        The output order follows ascending instance label IDs.
        """
        # --- normalize image to HWC [H, W, 3] ---
        if img.ndim != 3:
            raise ValueError(f"Expected img with ndim=3, got shape {img.shape}")

        if img.shape[-1] == 3:  # HWC
            img_hwc = img
        elif img.shape[0] == 3:  # CHW
            img_hwc = np.moveaxis(img, 0, -1)  # [3, H, W] -> [H, W, 3]
        else:
            raise ValueError(f"Expected 3 channels, got shape {img.shape}")

        if inst.ndim != 2:
            raise ValueError(f"Expected inst with shape [H, W], got {inst.shape}")

        if img_hwc.shape[:2] != inst.shape:
            raise ValueError(
                f"Image and instance map shapes do not match: "
                f"img {img_hwc.shape[:2]} vs inst {inst.shape}"
            )

        # ensure contiguous for fast flattening
        img_hwc = np.ascontiguousarray(img_hwc)
        inst = np.ascontiguousarray(inst)

        # flatten
        H, W, _ = img_hwc.shape
        inst_flat = inst.ravel()

        # ignore background (0)
        mask = inst_flat > 0
        if not np.any(mask):
            return []

        labels = inst_flat[mask].astype(np.int64)

        # flatten image to [N, 3] and mask
        pixels = img_hwc.reshape(-1, 3)[mask].astype(np.float64)

        # per-label sums with bincount
        sum_r = np.bincount(labels, weights=pixels[:, 0])
        sum_g = np.bincount(labels, weights=pixels[:, 1])
        sum_b = np.bincount(labels, weights=pixels[:, 2])
        counts = np.bincount(labels)

        # labels that actually exist (counts > 0)
        roi_ids = np.nonzero(counts)[0]
        # just in case, drop label 0 if it sneaks in
        roi_ids = roi_ids[roi_ids > 0]

        res: List[Dict[str, Any]] = []
        for rid in roi_ids:
            area = int(counts[rid])
            mean_r = float(sum_r[rid] / area)
            mean_g = float(sum_g[rid] / area)
            mean_b = float(sum_b[rid] / area)
            res.append({
                "roi_id": int(rid),
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "area": area,
            })

        return res
