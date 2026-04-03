from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MADScaler:
    median_: np.ndarray
    mad_: np.ndarray
    eps: float = 1e-6
    clip: float = 8.0

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-6, clip: float = 8.0) -> "MADScaler":
        X = np.asarray(X, dtype=np.float32)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        mad = np.maximum(mad, float(eps))
        return cls(
            median_=med.astype(np.float32, copy=False),
            mad_=mad.astype(np.float32, copy=False),
            eps=float(eps),
            clip=float(clip),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Z = (X - self.median_) / self.mad_
        if self.clip and self.clip > 0:
            Z = np.clip(Z, -float(self.clip), float(self.clip))
        return Z.astype(np.float32, copy=False)
