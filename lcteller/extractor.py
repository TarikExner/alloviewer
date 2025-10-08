import numpy as np
from typing import List, Dict, Any

class RGBExtractor:
    def __call__(self, img: np.ndarray, inst: np.ndarray) -> List[Dict[str, Any]]:
        res: List[Dict[str, Any]] = []
        max_id = int(inst.max())
        if max_id <= 0:
            return res
        for rid in range(1, max_id + 1):
            m = (inst == rid)
            if not m.any():
                continue
            vals = img[m]  # Nx3
            mean = vals.mean(axis=0)
            res.append({
                "roi_id": rid,
                "mean_r": float(mean[0]),
                "mean_g": float(mean[1]),
                "mean_b": float(mean[2]),
                "area": int(m.sum()),
            })
        return res



