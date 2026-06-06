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

    This gives a much flatter area distribution than sampling width directly.
    """
    name: str = "Camera"

    # target image area in megapixels
    megapixels: Tuple[float, float] = (2.0, 14.0)

    # optional fixed H/W override; normally leave both None
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

    # safety lower bound
    min_dim: int = 512

    def sample(self, rng: RNG) -> Dict[str, Any]:
        if self.W is not None and self.H is not None:
            H = int(self.H)
            W = int(self.W)
            return {"H": H, "W": W}

        mp_lo, mp_hi = self.megapixels
        area = float(rng.uniform(float(mp_lo), float(mp_hi))) * 1_000_000.0

        ratio = choose_ratio(rng, self.aspect_ratios, self.portrait_prob)
        ratio = float(ratio)  # W / H

        # area = W * H
        # W = ratio * H
        # area = ratio * H^2
        H = np.sqrt(area / ratio)
        W = ratio * H

        # enforce minimum dimension while preserving aspect ratio
        short_side = min(H, W)
        if short_side < self.min_dim:
            scale = float(self.min_dim) / max(1.0, short_side)
            H *= scale
            W *= scale

        H = round_to_multiple(H, self.size_multiple)
        W = round_to_multiple(W, self.size_multiple)

        # enforce again after rounding
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
