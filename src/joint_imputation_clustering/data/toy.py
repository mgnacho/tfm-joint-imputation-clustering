from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from joint_imputation_clustering.utils.seeding import stable_seed


def ensure_positive_definite(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue <= 1e-8:
        covariance = covariance + np.eye(covariance.shape[0]) * (
            1e-8 - minimum_eigenvalue + 1e-4
        )
    return covariance


def noisy_geometry(dimension: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if dimension == 2:
        means = [
            np.array([0.0, 0.0]),
            np.array([3.0, 0.5]),
            np.array([1.3, 2.6]),
        ]
        covariances = [
            np.array([[1.45, 0.72], [0.72, 0.62]]),
            np.array([[0.72, -0.48], [-0.48, 1.35]]),
            np.array([[1.05, 0.48], [0.48, 0.78]]),
        ]
    elif dimension == 3:
        means = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 0.5, 0.8]),
            np.array([1.3, 2.6, 2.2]),
        ]
        covariances = [
            np.array([[1.40, 0.65, 0.25], [0.65, 0.70, 0.20], [0.25, 0.20, 0.55]]),
            np.array([[0.80, -0.42, 0.15], [-0.42, 1.35, 0.38], [0.15, 0.38, 0.80]]),
            np.array([[1.00, 0.25, -0.38], [0.25, 0.70, 0.30], [-0.38, 0.30, 1.15]]),
        ]
    else:
        raise ValueError("The confirmatory toy supports only dimensions 2 and 3")
    return means, [ensure_positive_definite(cov) for cov in covariances]


def generate_noisy_toy_data(
    n_total: int,
    dimension: int,
    k: int,
    seed: int,
    outlier_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k != 3:
        raise ValueError("The fixed toy geometry is defined for k=3")
    rng = np.random.default_rng(seed)
    means, covariances = noisy_geometry(dimension)

    base, remainder = divmod(n_total, k)
    cluster_sizes = [base + (1 if cluster < remainder else 0) for cluster in range(k)]

    matrices: list[np.ndarray] = []
    labels: list[int] = []
    for cluster, cluster_size in enumerate(cluster_sizes):
        matrices.append(
            rng.multivariate_normal(means[cluster], covariances[cluster], size=cluster_size)
        )
        labels.extend([cluster] * cluster_size)

    x = np.vstack(matrices)
    y = np.asarray(labels, dtype=int)
    is_outlier = np.zeros(n_total, dtype=bool)

    n_outliers = max(1, int(round(outlier_rate * n_total)))
    outlier_indices = rng.choice(n_total, size=n_outliers, replace=False)
    directions = rng.normal(size=(n_outliers, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms <= 1e-12, 1.0, norms)
    magnitudes = rng.uniform(3.0, 5.0, size=(n_outliers, 1))
    x[outlier_indices] = x[outlier_indices] + directions * magnitudes
    is_outlier[outlier_indices] = True

    permutation = rng.permutation(n_total)
    return x[permutation], y[permutation], is_outlier[permutation]


def stratified_train_test_indices(
    y: np.ndarray,
    n_total: int,
    dimension: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    split_seed = stable_seed(n_total, dimension, "split_train_test", base=21_000)
    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=split_seed,
        stratify=y,
        shuffle=True,
    )
    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)
