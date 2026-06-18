from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

from joint_imputation_clustering.data.missingness import clip_imputed_values
from joint_imputation_clustering.utils.seeding import stable_seed


@dataclass
class CandidateTensors:
    names: list[str]
    train: np.ndarray
    test: np.ndarray
    audit: pd.DataFrame
    runtimes: dict[str, float]


def safe_mode_rounded(values: np.ndarray, decimals: int) -> float:
    observed = np.asarray(values, dtype=float)
    observed = observed[~np.isnan(observed)]
    if observed.size == 0:
        raise ValueError("No observed values for rounded mode")
    modes = pd.Series(np.round(observed, decimals)).mode()
    return float(modes.iloc[0])


def empirical_random_values_from_train(
    train_column: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    observed = np.asarray(train_column, dtype=float)
    observed = observed[~np.isnan(observed)]
    if observed.size == 0:
        raise ValueError("No observed values for empirical sampling")
    rng = np.random.default_rng(seed)
    return rng.choice(observed, size=size, replace=True).astype(float)


def fill_column_strategy(x_missing: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.asarray(x_missing, dtype=float).copy()
    for feature in range(result.shape[1]):
        missing = np.isnan(result[:, feature])
        result[missing, feature] = float(values[feature])
    return result


def pmm_like_impute_train_test(
    x_train_missing: np.ndarray,
    x_test_missing: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_donors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_result = x_train_missing.copy()
    test_result = x_test_missing.copy()
    n_features = x_train_missing.shape[1]

    medians = np.nanmedian(x_train_missing, axis=0)
    if np.isnan(medians).any():
        raise ValueError("PMM-like found a feature without observed TRAIN values")
    train_base = np.where(np.isnan(x_train_missing), medians, x_train_missing)
    test_base = np.where(np.isnan(x_test_missing), medians, x_test_missing)

    for feature in range(n_features):
        observed_train = ~np.isnan(x_train_missing[:, feature])
        missing_train = np.isnan(x_train_missing[:, feature])
        missing_test = np.isnan(x_test_missing[:, feature])

        if observed_train.sum() < max(5, n_features + 1):
            train_result[missing_train, feature] = medians[feature]
            test_result[missing_test, feature] = medians[feature]
            continue

        predictors = [index for index in range(n_features) if index != feature]
        regression = LinearRegression()
        regression.fit(
            train_base[observed_train][:, predictors],
            x_train_missing[observed_train, feature],
        )
        predicted_observed = regression.predict(train_base[observed_train][:, predictors])
        observed_values = x_train_missing[observed_train, feature]

        def donor_impute(rows: np.ndarray, stream_name: str) -> np.ndarray:
            if rows.shape[0] == 0:
                return np.empty(0, dtype=float)
            rng = np.random.default_rng(stable_seed(seed, feature, stream_name))
            predicted_missing = regression.predict(rows[:, predictors])
            output: list[float] = []
            for predicted in predicted_missing:
                distances = np.abs(predicted_observed - predicted)
                donor_indices = np.argsort(distances)[: min(n_donors, len(distances))]
                output.append(float(observed_values[rng.choice(donor_indices)]))
            return np.asarray(output, dtype=float)

        train_result[missing_train, feature] = donor_impute(
            train_base[missing_train], "pmm_train"
        )
        test_result[missing_test, feature] = donor_impute(test_base[missing_test], "pmm_test")

    return (
        clip_imputed_values(x_train_missing, train_result, lower, upper),
        clip_imputed_values(x_test_missing, test_result, lower, upper),
    )


def _run_candidate(
    name: str,
    builder: Callable[[], tuple[np.ndarray, np.ndarray]],
    x_train_missing: np.ndarray,
    x_test_missing: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    started = time.perf_counter()
    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_matrix, test_matrix = builder()
            captured = [str(item.message) for item in caught]
        train_matrix = clip_imputed_values(x_train_missing, train_matrix, lower, upper)
        test_matrix = clip_imputed_values(x_test_missing, test_matrix, lower, upper)
        success = True
        error_type = ""
        error_message = ""
    except Exception as exc:  # Candidate is excluded rather than silently relabelled.
        train_matrix = None
        test_matrix = None
        success = False
        error_type = type(exc).__name__
        error_message = str(exc)
    runtime = time.perf_counter() - started
    audit = {
        "candidate": name,
        "success": success,
        "excluded": not success,
        "runtime_seconds": runtime,
        "warning_count": len(captured),
        "warnings": " | ".join(captured),
        "error_type": error_type,
        "error_message": error_message,
    }
    return train_matrix, test_matrix, audit


def build_candidate_tensors_train_test(
    x_train_missing: np.ndarray,
    x_test_missing: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    requested_names: list[str],
    seed: int,
    mode_decimals: int = 2,
    knn_neighbors: int = 5,
    pmm_donors: int = 5,
) -> CandidateTensors:
    n_train, n_features = x_train_missing.shape
    train_means = np.nanmean(x_train_missing, axis=0)
    train_medians = np.nanmedian(x_train_missing, axis=0)
    train_modes = np.array(
        [safe_mode_rounded(x_train_missing[:, feature], mode_decimals) for feature in range(n_features)]
    )
    if np.isnan(train_means).any() or np.isnan(train_medians).any():
        raise ValueError("At least one TRAIN feature has no observed values")

    builders: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {}
    builders["mean"] = lambda: (
        fill_column_strategy(x_train_missing, train_means),
        fill_column_strategy(x_test_missing, train_means),
    )
    builders["median"] = lambda: (
        fill_column_strategy(x_train_missing, train_medians),
        fill_column_strategy(x_test_missing, train_medians),
    )
    builders["mode_rounded"] = lambda: (
        fill_column_strategy(x_train_missing, train_modes),
        fill_column_strategy(x_test_missing, train_modes),
    )

    def random_empirical() -> tuple[np.ndarray, np.ndarray]:
        train_result = x_train_missing.copy()
        test_result = x_test_missing.copy()
        for feature in range(n_features):
            train_missing = np.isnan(train_result[:, feature])
            test_missing = np.isnan(test_result[:, feature])
            train_result[train_missing, feature] = empirical_random_values_from_train(
                x_train_missing[:, feature],
                int(train_missing.sum()),
                stable_seed(seed, feature, "empirical_train"),
            )
            test_result[test_missing, feature] = empirical_random_values_from_train(
                x_train_missing[:, feature],
                int(test_missing.sum()),
                stable_seed(seed, feature, "empirical_test"),
            )
        return train_result, test_result

    builders["random_empirical"] = random_empirical

    def knn_builder() -> tuple[np.ndarray, np.ndarray]:
        model = KNNImputer(
            n_neighbors=min(knn_neighbors, max(1, n_train - 1)),
            weights="distance",
        )
        return model.fit_transform(x_train_missing), model.transform(x_test_missing)

    builders["knn"] = knn_builder

    def iterative_bayes_builder() -> tuple[np.ndarray, np.ndarray]:
        model = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            sample_posterior=False,
            random_state=stable_seed(seed, "iterative_bayes"),
            min_value=lower,
            max_value=upper,
            initial_strategy="median",
        )
        return model.fit_transform(x_train_missing), model.transform(x_test_missing)

    builders["iterative_bayes"] = iterative_bayes_builder

    def iterative_rf_builder() -> tuple[np.ndarray, np.ndarray]:
        model = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=30,
                random_state=stable_seed(seed, "iterative_rf_estimator"),
                n_jobs=-1,
                min_samples_leaf=2,
                max_depth=6,
            ),
            max_iter=4,
            sample_posterior=False,
            random_state=stable_seed(seed, "iterative_rf"),
            min_value=lower,
            max_value=upper,
            initial_strategy="median",
        )
        return model.fit_transform(x_train_missing), model.transform(x_test_missing)

    builders["iterative_rf"] = iterative_rf_builder
    builders["pmm_like"] = lambda: pmm_like_impute_train_test(
        x_train_missing,
        x_test_missing,
        lower,
        upper,
        n_donors=pmm_donors,
        seed=stable_seed(seed, "pmm_like"),
    )

    successful_names: list[str] = []
    train_matrices: list[np.ndarray] = []
    test_matrices: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    runtimes: dict[str, float] = {}

    for name in requested_names:
        if name not in builders:
            raise ValueError(f"Unknown imputation candidate: {name}")
        train_matrix, test_matrix, audit = _run_candidate(
            name,
            builders[name],
            x_train_missing,
            x_test_missing,
            lower,
            upper,
        )
        audit_rows.append(audit)
        runtimes[name] = float(audit["runtime_seconds"])
        if bool(audit["success"]):
            assert train_matrix is not None and test_matrix is not None
            successful_names.append(name)
            train_matrices.append(train_matrix)
            test_matrices.append(test_matrix)

    if not successful_names:
        raise RuntimeError("All imputation candidates failed")

    return CandidateTensors(
        names=successful_names,
        train=np.stack(train_matrices, axis=0),
        test=np.stack(test_matrices, axis=0),
        audit=pd.DataFrame(audit_rows),
        runtimes=runtimes,
    )
