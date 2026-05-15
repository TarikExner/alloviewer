import numpy as np
from typing import Union, Tuple

RNG = np.random.Generator
NumOrRange = Union[int, float, Tuple[float, float], Tuple[int, int]]
ChannelRange = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

