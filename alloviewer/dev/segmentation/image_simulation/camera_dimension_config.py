from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict, Sequence

from .types import RNG, NumOrRange
from .utils import (
    choose_ratio,
    round_to_multiple,
    sample_number
)


from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np

@dataclass
class CameraDimension:
    """
    Sample image dimensions by target megapixels.

    Megapixels are sampled log-uniformly by default, so the sampler covers
    the full range but does not over-sample very large, slow images.

    The aspect ratio is sampled independently, then H/W are derived from:
        area = H * W
        ratio = W / H
    """
    name: str = "Camera"

    # target image area in megapixels
    megapixels: Tuple[float, float] = (2.0, 14.0)

    # sampling mode: "log_uniform" or "uniform"
    megapixel_sampling: str = "log_uniform"

    # optional fixed dimensions; if both are set, megapixel sampling is ignored
    W: Optional[int] = None
    H: Optional[int] = None

    # aspect ratios as (width, height)
    aspect_ratios: Sequence[Tuple[int, int]] = (
        (16, 9),
        (16, 10),
        (3, 2),
        (4, 3),
    )
    portrait_prob: float = 0.5
    size_multiple: int = 32

    # keep both sides at least this large
    min_dim: int = 512

    def sample(self, rng: RNG) -> Dict[str, Any]:
        # fixed dimensions override sampling
        if self.W is not None and self.H is not None:
            H = round_to_multiple(float(self.H), self.size_multiple)
            W = round_to_multiple(float(self.W), self.size_multiple)
            return {
                "H": int(H),
                "W": int(W),
            }

        mp_lo, mp_hi = self.megapixels
        mp_lo = max(float(mp_lo), 1e-6)
        mp_hi = max(float(mp_hi), mp_lo)

        if self.megapixel_sampling == "log_uniform":
            mp = float(np.exp(rng.uniform(np.log(mp_lo), np.log(mp_hi))))
        elif self.megapixel_sampling == "uniform":
            mp = float(rng.uniform(mp_lo, mp_hi))
        else:
            raise ValueError(
                f"Unknown megapixel_sampling={self.megapixel_sampling!r}. "
                "Use 'log_uniform' or 'uniform'."
            )

        area = mp * 1_000_000.0

        ratio = choose_ratio(
            rng,
            self.aspect_ratios,
            self.portrait_prob,
        )
        ratio = float(ratio)  # W / H

        # area = W * H
        # W = ratio * H
        # area = ratio * H^2
        H = float(np.sqrt(area / ratio))
        W = float(ratio * H)

        # enforce minimum side length while preserving aspect ratio
        short_side = min(H, W)
        if short_side < float(self.min_dim):
            scale = float(self.min_dim) / max(1.0, short_side)
            H *= scale
            W *= scale

        # round to allowed multiple
        H = round_to_multiple(H, self.size_multiple)
        W = round_to_multiple(W, self.size_multiple)

        # enforce again after rounding, using ceil-style behavior
        if H < self.min_dim:
            H = int(np.ceil(self.min_dim / self.size_multiple) * self.size_multiple)
        if W < self.min_dim:
            W = int(np.ceil(self.min_dim / self.size_multiple) * self.size_multiple)

        return {
            "H": int(H),
            "W": int(W),
        }

###############
### PRESETS ###
###############

def default_camera() -> CameraDimension:
    return CameraDimension(
        name = "default_camera"
    )

def train_camera() -> CameraDimension:
    return default_camera()

def test_camera() -> CameraDimension:
    return CameraDimension(
        name = "test_cam",
        W = 2160,
        H = 1620,

        aspect_ratios = ((4,3), (4,3)),
        portrait_prob = 0,
    )

def img_export_camera() -> CameraDimension:
    return CameraDimension(
        name = "img_export_cam",
        W = 2160,
        H = 1620,
    )
