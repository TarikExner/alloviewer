from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MADScaler:
    """
    Store per-feature medians and MADs for clipped robust scaling.

    Attributes
    ----------
    median_ : numpy.ndarray
        Per-feature medians fitted on the training events.
    mad_ : numpy.ndarray
        Per-feature median absolute deviations, floored at eps.
    eps : float
        Small positive floor preventing division by zero. Default is ``1e-06``.
    clip : float
        Symmetric clipping bound; a nonpositive value disables clipping. Default is ``8.0``.
    """
    median_: np.ndarray
    mad_: np.ndarray
    eps: float = 1e-6
    clip: float = 8.0

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-6, clip: float = 8.0) -> "MADScaler":
        """
        Estimate column medians and MADs, flooring each MAD at eps.

        Parameters
        ----------
        X : numpy.ndarray
            Numeric feature matrix with rows for events and columns for features.
        eps : float
            Small positive floor preventing division by zero. Default is ``1e-06``.
        clip : float
            Symmetric clipping bound; a nonpositive value disables clipping. Default is ``8.0``.

        Returns
        -------
        MADScaler
            Column medians and MADs, flooring each MAD at eps.
        """
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
        """
        Center and scale features using fitted medians and MADs, with optional clipping.

        Parameters
        ----------
        X : numpy.ndarray
            Numeric feature matrix with rows for events and columns for features.

        Returns
        -------
        numpy.ndarray
            Computed value of ``Z.astype(np.float32, copy=False)``.
        """
        X = np.asarray(X, dtype=np.float32)
        Z = (X - self.median_) / self.mad_
        if self.clip and self.clip > 0:
            Z = np.clip(Z, -float(self.clip), float(self.clip))
        return Z.astype(np.float32, copy=False)
