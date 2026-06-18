from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass
class ClusteringResult:
    labels_train: np.ndarray
    labels_test: np.ndarray
    centers: np.ndarray
    center_indices_train: np.ndarray | None
    native_cost: float
    runtime_seconds: float


def assign_to_nearest_centers(x: np.ndarray, centers: np.ndarray, metric: str) -> np.ndarray:
    distances = pairwise_distances(x, centers, metric=metric)
    return np.argmin(distances, axis=1).astype(int)


def fit_kmeans_reference(
    x_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    seed: int,
) -> ClusteringResult:
    started = time.perf_counter()
    model = KMeans(n_clusters=k, random_state=int(seed), n_init=30)
    model.fit(x_train)
    return ClusteringResult(
        labels_train=np.asarray(model.labels_, dtype=int),
        labels_test=np.asarray(model.predict(x_test), dtype=int),
        centers=np.asarray(model.cluster_centers_, dtype=float),
        center_indices_train=None,
        native_cost=float(model.inertia_),
        runtime_seconds=time.perf_counter() - started,
    )


def fit_pam_reference(
    x_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
    seed: int,
    metric: str = "manhattan",
    max_iter: int = 300,
) -> ClusteringResult:
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise ImportError(
            "PAM requires scikit-learn-extra. Install the pinned environment."
        ) from exc

    started = time.perf_counter()
    model = KMedoids(
        n_clusters=k,
        metric=metric,
        method="pam",
        init="k-medoids++",
        max_iter=max_iter,
        random_state=int(seed),
    )
    model.fit(x_train)
    medoid_indices = np.asarray(model.medoid_indices_, dtype=int)
    medoids = np.asarray(x_train[medoid_indices], dtype=float)
    labels_train = assign_to_nearest_centers(x_train, medoids, metric=metric)
    labels_test = assign_to_nearest_centers(x_test, medoids, metric=metric)
    return ClusteringResult(
        labels_train=labels_train,
        labels_test=labels_test,
        centers=medoids,
        center_indices_train=medoid_indices,
        native_cost=float(model.inertia_),
        runtime_seconds=time.perf_counter() - started,
    )
