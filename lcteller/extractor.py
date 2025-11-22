import numpy as np
from typing import List, Dict, Any


class RGBExtractor:
    def __call__(self, img: np.ndarray, inst: np.ndarray) -> List[Dict[str, Any]]:
        """
        Compute mean RGB per instance.

        img:
          - HWC: [H, W, 3]
          - CHW: [3, H, W]
        inst:
          - [H, W], integer instance labels (0 = background)
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

        res: List[Dict[str, Any]] = []
        max_id = int(inst.max())
        if max_id <= 0:
            return res

        for rid in range(1, max_id + 1):
            m = (inst == rid)
            if not m.any():
                continue
            vals = img_hwc[m]  # [N, 3]
            mean = vals.mean(axis=0)
            res.append({
                "roi_id": int(rid),
                "mean_r": float(mean[0]),
                "mean_g": float(mean[1]),
                "mean_b": float(mean[2]),
                "area": int(m.sum()),
            })

        return res

