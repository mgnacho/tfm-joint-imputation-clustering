from __future__ import annotations

import time
from typing import Any

import numpy as np

from joint_imputation_clustering.clustering.references import (
    fit_kmeans_reference,
    fit_pam_reference,
)
from joint_imputation_clustering.metrics.evaluation import evaluate_partition


def evaluate_baseline(
    *,
    method_name: str,
    cluster_algorithm: str,
    x_train_imputed: np.ndarray,
    x_test_imputed: np.ndarray,
    x_test_complete: np.ndarray,
    mask_test: np.ndarray,
    k: int,
    seed: int,
    labels_ref_kmeans_test: np.ndarray,
    labels_ref_pam_test: np.ndarray,
    labels_ref_l1_test: np.ndarray,
    y_test_true: np.ndarray | None,
    train_scales: dict[str, np.ndarray],
    imputation_runtime: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if cluster_algorithm == "kmeans":
        clustering = fit_kmeans_reference(x_train_imputed, x_test_imputed, k, seed)
    elif cluster_algorithm == "pam":
        clustering = fit_pam_reference(x_train_imputed, x_test_imputed, k, seed)
    else:
        raise ValueError("cluster_algorithm must be 'kmeans' or 'pam'")
    clustering_runtime = time.perf_counter() - started

    metrics = evaluate_partition(
        x_complete=x_test_complete,
        x_imputed=x_test_imputed,
        labels=clustering.labels_test,
        missing_mask=mask_test,
        labels_ref_kmeans=labels_ref_kmeans_test,
        labels_ref_pam=labels_ref_pam_test,
        labels_ref_l1=labels_ref_l1_test,
        y_true=y_test_true,
        centers=clustering.centers,
        train_scales=train_scales,
    )
    return {
        "method": method_name,
        "cluster_algo": cluster_algorithm,
        "labels_train": clustering.labels_train,
        "labels_test": clustering.labels_test,
        "centers_values": clustering.centers,
        "center_indices_train": clustering.center_indices_train,
        "native_clustering_cost": clustering.native_cost,
        "imputation_runtime": float(imputation_runtime),
        "clustering_runtime": float(clustering_runtime),
        "total_runtime": float(imputation_runtime + clustering_runtime),
        **metrics,
    }
