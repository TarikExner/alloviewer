from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import hdbscan
import numpy as np
from flowsom.models import FlowSOMEstimator
from hdbscan.prediction import approximate_predict, approximate_predict_scores
from parc import PARC

from .config import ClustererConfig, FlowSOMConfig, HDBSCANConfig, PARCConfig


FlowSOMLabelKind = Literal["cluster", "metacluster"]


@dataclass
class GenericPrediction:
    """Generic prediction output returned by clusterer adapters.

    Parameters
    ----------
    labels : numpy.ndarray
        Predicted cluster labels.
    prob : numpy.ndarray or None, optional
        Prediction confidence or assignment probability, when available.
    outlier_score : numpy.ndarray or None, optional
        Outlier scores, when available.
    """

    labels: np.ndarray
    prob: Optional[np.ndarray] = None
    outlier_score: Optional[np.ndarray] = None


class BaseClusterer(Protocol):
    """Protocol for clustering backends.

    Classes implementing this protocol can be used by the gating workflow as
    interchangeable clustering backends.
    """

    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """Fit the clusterer.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        BaseClusterer
            Fitted clusterer instance.
        """
        ...

    def predict(self, X: np.ndarray) -> GenericPrediction:
        """Predict cluster labels.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        GenericPrediction
            Predicted labels and optional confidence values.
        """
        ...

    @property
    def labels_(self) -> np.ndarray:
        """Return training labels.

        Returns
        -------
        numpy.ndarray
            Cluster labels assigned during fitting.
        """
        ...


class FlowSOMClusterer:
    """Adapter for :class:`flowsom.models.FlowSOMEstimator`.

    Parameters
    ----------
    cfg : FlowSOMConfig
        FlowSOM clusterer configuration.

    Raises
    ------
    ValueError
        If ``cfg.label_kind`` is not ``"cluster"`` or ``"metacluster"``.
    """

    def __init__(self, cfg: FlowSOMConfig) -> None:
        if cfg.label_kind not in {"cluster", "metacluster"}:
            raise ValueError("cfg.label_kind must be 'cluster' or 'metacluster'")

        self.cfg = cfg
        self.label_kind = cfg.label_kind

        estimator_kwargs = dict(cfg.estimator_kwargs)
        estimator_kwargs.setdefault("n_clusters", cfg.n_clusters)
        estimator_kwargs.setdefault("seed", cfg.seed)

        self.model = FlowSOMEstimator(**estimator_kwargs)

        self._is_fitted = False
        self._X_fit: Optional[np.ndarray] = None
        self._labels_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FlowSOMClusterer":
        """Fit FlowSOM on a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        FlowSOMClusterer
            Fitted clusterer.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        self.model.fit_predict(X)
        self._X_fit = X
        self._labels_fit = self._get_training_labels()
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> GenericPrediction:
        """Predict FlowSOM labels for a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        GenericPrediction
            Predicted labels. Probability and outlier score are not provided.

        Raises
        ------
        RuntimeError
            If the clusterer has not been fitted.
        ValueError
            If ``X`` is not two-dimensional.
        """
        if not self._is_fitted:
            raise RuntimeError("FlowSOMClusterer must be fitted before calling predict().")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        y = self.model.predict(X)
        self._labels_fit = y

        return GenericPrediction(
            labels=np.asarray(y, dtype=int),
            prob=None,
            outlier_score=None,
        )

    @property
    def labels_(self) -> np.ndarray:
        """Return labels assigned during fitting.

        Returns
        -------
        numpy.ndarray
            Training labels.

        Raises
        ------
        AttributeError
            If the clusterer has not been fitted.
        """
        if not self._is_fitted or self._labels_fit is None:
            raise AttributeError("labels_ is only available after fit().")

        return self._labels_fit

    def _get_training_labels(self) -> np.ndarray:
        """Return FlowSOM training labels for the configured label kind.

        Returns
        -------
        numpy.ndarray
            Cluster or metacluster labels.

        Raises
        ------
        RuntimeError
            If ``label_kind`` is unsupported.
        """
        if self.label_kind == "cluster":
            return np.asarray(self.model.cluster_labels, dtype=int)

        if self.label_kind == "metacluster":
            return np.asarray(self.model.metacluster_labels, dtype=int)

        raise RuntimeError(f"Unsupported label kind: {self.label_kind}")


class PARCClusterer(PARC):
    """Adapter exposing PARC through the common clusterer interface.

    Parameters
    ----------
    cfg : PARCConfig
        PARC clusterer configuration.
    """

    def __init__(self, cfg: PARCConfig) -> None:
        self.cfg = cfg
        self.allow_oos_prediction = cfg.allow_oos_prediction
        self.prediction_k = cfg.prediction_k

        super().__init__(
            data=np.empty((0, 0), dtype=np.float32),
            true_label=None,
            dist_std_local=cfg.dist_std_local,
            jac_std_global=cfg.jac_std_global,
            keep_all_local_dist=cfg.keep_all_local_dist,
            too_big_factor=cfg.too_big_factor,
            small_pop=cfg.small_pop,
            jac_weighted_edges=cfg.jac_weighted_edges,
            knn=cfg.knn,
            n_iter_leiden=cfg.n_iter_leiden,
            random_seed=42,
            num_threads=cfg.num_threads,
            distance=cfg.distance,
            time_smallpop=cfg.time_smallpop,
            partition_type=cfg.partition_type,
            resolution_parameter=cfg.resolution_parameter,
            knn_struct=None,
            neighbor_graph=None,
            hnsw_param_ef_construction=cfg.hnsw_param_ef_construction,
        )

        self._is_fitted = False
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PARCClusterer":
        """Fit PARC on a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        PARCClusterer
            Fitted clusterer.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        self.data = X
        self.true_label = None
        self.labels = None
        self.knn_struct = None
        self.neighbor_graph = None

        self.run_PARC()

        self._X_fit = X
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> GenericPrediction:
        """Predict PARC labels for a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        GenericPrediction
            Predicted labels with optional kNN-transfer confidence.

        Raises
        ------
        RuntimeError
            If the clusterer has not been fitted or no kNN structure is
            available for out-of-sample prediction.
        ValueError
            If ``X`` is not two-dimensional.
        NotImplementedError
            If out-of-sample prediction is requested while disabled.
        """
        if not self._is_fitted or self._X_fit is None:
            raise RuntimeError("PARCClusterer must be fitted before calling predict().")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        if X is self._X_fit or np.array_equal(X, self._X_fit):
            return GenericPrediction(
                labels=self.labels_,
                prob=None,
                outlier_score=None,
            )

        if not self.allow_oos_prediction:
            raise NotImplementedError(
                "PARC has no native out-of-sample predict(). "
                "Set allow_oos_prediction=True to use kNN label transfer."
            )

        if self.knn_struct is None:
            raise RuntimeError(
                "No fitted kNN structure is available. "
                "This should not happen after a successful fit."
            )

        k = self.prediction_k if self.prediction_k is not None else self.knn
        k = max(1, min(int(k), self._X_fit.shape[0]))

        neighbor_idx, _ = self.knn_struct.knn_query(X, k=k)
        train_labels = self.labels_

        pred_labels = np.empty(X.shape[0], dtype=int)
        pred_prob = np.empty(X.shape[0], dtype=np.float32)

        for i, nn_idx in enumerate(neighbor_idx):
            nn_labels = train_labels[nn_idx]
            values, counts = np.unique(nn_labels, return_counts=True)
            best = np.argmax(counts)
            pred_labels[i] = int(values[best])
            pred_prob[i] = np.float32(counts[best] / counts.sum())

        return GenericPrediction(
            labels=pred_labels,
            prob=pred_prob,
            outlier_score=1.0 - pred_prob,
        )

    @property
    def labels_(self) -> np.ndarray:
        """Return labels assigned during fitting.

        Returns
        -------
        numpy.ndarray
            Training labels.

        Raises
        ------
        AttributeError
            If the clusterer has not been fitted.
        """
        if not self._is_fitted or self.labels is None:
            raise AttributeError("labels_ is only available after fit().")

        return np.asarray(self.labels, dtype=int)


class HDBSCANClusterer:
    """Adapter for HDBSCAN clustering.

    Parameters
    ----------
    cfg : HDBSCANConfig
        HDBSCAN clusterer configuration.
    """

    def __init__(self, cfg: HDBSCANConfig) -> None:
        self.cfg = cfg
        self.min_cluster_size = cfg.min_cluster_size
        self.min_samples = cfg.min_samples
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=int(cfg.min_cluster_size),
            min_samples=(None if cfg.min_samples is None else int(cfg.min_samples)),
            prediction_data=bool(cfg.prediction_data),
            cluster_selection_method=cfg.cluster_selection_method,
        )
        self.max_noise = cfg.max_noise

    def modify_model(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> None:
        """Rebuild the underlying HDBSCAN model.

        Parameters
        ----------
        min_cluster_size : int or None, optional
            Minimum cluster size. If ``None``, the current value is reused.
        min_samples : int or None, optional
            Minimum samples value. If ``None``, the current value is reused.
        """
        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size
        if min_samples is None:
            min_samples = self.min_samples

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=(None if min_samples is None else int(min_samples)),
            prediction_data=bool(self.cfg.prediction_data),
            cluster_selection_method=self.cfg.cluster_selection_method,
        )

    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        """Fit HDBSCAN on a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        HDBSCANClusterer
            Fitted clusterer.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        self.model.fit(X)
        y_train = self.labels_
        n_noise = int(np.sum(y_train == -1))
        frac_noise = n_noise / y_train.size
        i = 0

        while frac_noise > self.max_noise:
            if i == 5:
                print("MAX ITERATIONS REACHED WITHOUT REACHING TOLERANCE LEVEL!!!")
                self.max_noise *= 1.5
                self.modify_model(min_cluster_size=self.cfg.min_cluster_size)

            min_cluster_size = self.min_cluster_size
            self.modify_model(min_cluster_size=int(min_cluster_size * 1.5))
            self.model.fit(X)

            y_train = self.labels_
            n_train = y_train.size
            n_noise = int(np.sum(y_train == -1))
            frac_noise = n_noise / n_train
            i += 1

        return self

    def predict(self, X: np.ndarray) -> GenericPrediction:
        """Predict HDBSCAN labels for a feature matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional feature matrix.

        Returns
        -------
        GenericPrediction
            Predicted labels, prediction strengths, and outlier scores.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X.shape}")

        y, strength = approximate_predict(self.model, X)
        out_score = approximate_predict_scores(self.model, X)

        return GenericPrediction(
            labels=np.asarray(y, dtype=int),
            prob=np.asarray(strength, dtype=np.float32),
            outlier_score=np.asarray(out_score, dtype=np.float32),
        )

    @property
    def labels_(self) -> np.ndarray:
        """Return labels assigned during fitting.

        Returns
        -------
        numpy.ndarray
            Training labels.
        """
        return np.asarray(self.model.labels_, dtype=int)


def HDBSCANFactory(cfg: HDBSCANConfig) -> BaseClusterer:
    """Create an HDBSCAN clusterer.

    Parameters
    ----------
    cfg : HDBSCANConfig
        HDBSCAN configuration.

    Returns
    -------
    BaseClusterer
        HDBSCAN clusterer adapter.
    """
    return HDBSCANClusterer(cfg)


def FlowSOMFactory(cfg: FlowSOMConfig) -> BaseClusterer:
    """Create a FlowSOM clusterer.

    Parameters
    ----------
    cfg : FlowSOMConfig
        FlowSOM configuration.

    Returns
    -------
    BaseClusterer
        FlowSOM clusterer adapter.
    """
    return FlowSOMClusterer(cfg)


def PARCFactory(cfg: PARCConfig) -> BaseClusterer:
    """Create a PARC clusterer.

    Parameters
    ----------
    cfg : PARCConfig
        PARC configuration.

    Returns
    -------
    BaseClusterer
        PARC clusterer adapter.
    """
    return PARCClusterer(cfg)


def default_clusterer_factory(cfg: ClustererConfig) -> BaseClusterer:
    """Create the default clusterer for a clusterer configuration.

    Parameters
    ----------
    cfg : ClustererConfig
        Clusterer configuration.

    Returns
    -------
    BaseClusterer
        Clusterer adapter matching the configuration type.

    Raises
    ------
    TypeError
        If the configuration type is unsupported.
    """
    if isinstance(cfg, HDBSCANConfig):
        return HDBSCANFactory(cfg)

    if isinstance(cfg, FlowSOMConfig):
        return FlowSOMFactory(cfg)

    if isinstance(cfg, PARCConfig):
        return PARCFactory(cfg)

    raise TypeError(f"Unsupported clusterer config type: {type(cfg)!r}")
