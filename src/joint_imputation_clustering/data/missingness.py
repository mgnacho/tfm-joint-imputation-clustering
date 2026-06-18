from __future__ import annotations

import numpy as np


def induce_mcar_fixed_rate(
    x: np.ndarray,
    rate: float,
    seed: int,
    max_attempts: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove exactly round(rate*n*p) cells while preserving usable rows and columns."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n_rows, n_features = x.shape
    n_missing = int(round(rate * n_rows * n_features))
    if n_missing < n_features:
        raise ValueError("Too few missing cells to guarantee at least one per feature")

    for _ in range(max_attempts):
        flat_indices = rng.choice(n_rows * n_features, size=n_missing, replace=False)
        mask = np.zeros(n_rows * n_features, dtype=bool)
        mask[flat_indices] = True
        mask = mask.reshape(n_rows, n_features)

        row_counts = mask.sum(axis=1)
        column_counts = mask.sum(axis=0)
        valid_rows = np.all(row_counts < n_features)
        valid_columns = np.all((column_counts >= 1) & (column_counts < n_rows))
        if valid_rows and valid_columns:
            x_missing = np.asarray(x, dtype=float).copy()
            x_missing[mask] = np.nan
            return x_missing, mask

    raise RuntimeError("Unable to construct a valid exact-rate MCAR mask")


def compute_train_bounds(
    x_train_missing: np.ndarray,
    margin_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = x_train_missing.shape[1]
    lower = np.zeros(n_features, dtype=float)
    upper = np.zeros(n_features, dtype=float)

    for feature in range(n_features):
        observed = x_train_missing[~np.isnan(x_train_missing[:, feature]), feature]
        if observed.size == 0:
            raise ValueError(f"Feature {feature} has no observed TRAIN values")
        observed_min = float(np.min(observed))
        observed_max = float(np.max(observed))
        span = observed_max - observed_min
        margin = max(1.0, abs(observed_min) * 0.10) if span < 1e-8 else margin_fraction * span
        lower[feature] = observed_min - margin
        upper[feature] = observed_max + margin
    return lower, upper


def clip_imputed_values(
    x_missing: np.ndarray,
    x_imputed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    result = np.asarray(x_imputed, dtype=float).copy()
    missing_mask = np.isnan(x_missing)
    for feature in range(result.shape[1]):
        rows_missing = missing_mask[:, feature]
        result[rows_missing, feature] = np.clip(
            result[rows_missing, feature], lower[feature], upper[feature]
        )
        result[~rows_missing, feature] = x_missing[~rows_missing, feature]
    if np.isnan(result).any():
        raise ValueError("Imputed matrix still contains missing values")
    return result
