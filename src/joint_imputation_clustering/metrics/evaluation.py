from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def safe_silhouette_scores(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    output = {"euclidean": np.nan, "manhattan": np.nan}
    if len(unique) < 2 or len(unique) >= len(labels):
        return output
    for metric in output:
        try:
            output[metric] = float(silhouette_score(x, labels, metric=metric))
        except (ValueError, FloatingPointError):
            output[metric] = np.nan
    return output


def feature_scales(x_train_complete: np.ndarray) -> dict[str, np.ndarray]:
    feature_range = np.ptp(x_train_complete, axis=0)
    feature_std = np.std(x_train_complete, axis=0, ddof=0)
    feature_range = np.where(feature_range <= 1e-12, np.nan, feature_range)
    feature_std = np.where(feature_std <= 1e-12, np.nan, feature_std)
    return {"range": feature_range, "std": feature_std}


def imputation_metrics(
    x_true: np.ndarray,
    x_imputed: np.ndarray,
    missing_mask: np.ndarray,
    scales: dict[str, np.ndarray],
) -> dict[str, float | int]:
    missing_mask = np.asarray(missing_mask, dtype=bool)
    if missing_mask.sum() == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "nrmse_range": np.nan,
            "nrmse_std": np.nan,
            "n_missing_eval": 0,
        }

    errors = x_true - x_imputed
    selected_errors = errors[missing_mask]
    rmse = float(np.sqrt(np.mean(selected_errors**2)))
    mae = float(np.mean(np.abs(selected_errors)))

    per_feature_range: list[float] = []
    per_feature_std: list[float] = []
    for feature in range(x_true.shape[1]):
        feature_mask = missing_mask[:, feature]
        if not feature_mask.any():
            continue
        feature_rmse = float(np.sqrt(np.mean(errors[feature_mask, feature] ** 2)))
        if np.isfinite(scales["range"][feature]):
            per_feature_range.append(feature_rmse / float(scales["range"][feature]))
        if np.isfinite(scales["std"][feature]):
            per_feature_std.append(feature_rmse / float(scales["std"][feature]))

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse_range": float(np.mean(per_feature_range)) if per_feature_range else np.nan,
        "nrmse_std": float(np.mean(per_feature_std)) if per_feature_std else np.nan,
        "n_missing_eval": int(missing_mask.sum()),
    }


def common_l1_compactness(x_complete: np.ndarray, labels: np.ndarray) -> float:
    """Mean absolute deviation from cluster-wise medians in the common complete space."""
    labels = np.asarray(labels, dtype=int)
    total_absolute_deviation = 0.0
    for cluster in np.unique(labels):
        points = x_complete[labels == cluster]
        center = np.median(points, axis=0)
        total_absolute_deviation += float(np.sum(np.abs(points - center)))
    return total_absolute_deviation / float(x_complete.size)


def evaluate_partition(
    x_complete: np.ndarray,
    x_imputed: np.ndarray,
    labels: np.ndarray,
    missing_mask: np.ndarray,
    labels_ref_kmeans: np.ndarray,
    labels_ref_pam: np.ndarray,
    labels_ref_l1: np.ndarray,
    y_true: np.ndarray,
    centers: np.ndarray,
    train_scales: dict[str, np.ndarray],
) -> dict[str, float | int]:
    common_silhouette = safe_silhouette_scores(x_complete, labels)
    own_silhouette = safe_silhouette_scores(x_imputed, labels)
    imputation = imputation_metrics(x_complete, x_imputed, missing_mask, train_scales)
    return {
        "ari_ref_kmeans": float(adjusted_rand_score(labels_ref_kmeans, labels)),
        "ari_ref_pam": float(adjusted_rand_score(labels_ref_pam, labels)),
        "ari_ref_l1model": float(adjusted_rand_score(labels_ref_l1, labels)),
        "ari_true": float(adjusted_rand_score(y_true, labels)),
        "silhouette_common_euclidean": common_silhouette["euclidean"],
        "silhouette_common_manhattan": common_silhouette["manhattan"],
        "silhouette_own_euclidean": own_silhouette["euclidean"],
        "silhouette_own_manhattan": own_silhouette["manhattan"],
        "common_l1_compactness": common_l1_compactness(x_complete, labels),
        **imputation,
    }
