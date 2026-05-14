from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict, Sequence

from .types import RNG, NumOrRange
from .utils import (
    choose_ratio,
    round_to_multiple,
    sample_number
)


@dataclass
class CameraDimension:
    """
    Sample image dimensions.

    Width is sampled first. If H is not fixed, height is computed from
    a sampled aspect ratio.
    """
    name: str = "Camera"

    # width / height
    W: NumOrRange = (512, 2500)
    H: Optional[int] = None

    # aspect ratios as (width, height)
    aspect_ratios: Sequence[Tuple[int, int]] = ((16, 9), (16, 10), (3, 2), (4, 3))
    portrait_prob: float = 0.5
    size_multiple: int = 32

    def sample(self, rng: RNG) -> Dict[str, Any]:
        W = sample_number(rng, self.W, integer=True)

        if self.H is None:
            ratio = choose_ratio(rng, self.aspect_ratios, self.portrait_prob)
            H = round_to_multiple(W / ratio, self.size_multiple)
        else:
            H = self.H

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
