from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

WHOLESALE_UCI_ID = 292
WHOLESALE_FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]
WHOLESALE_EXTERNAL = ["Channel", "Region"]
WHOLESALE_EXPECTED_COLUMNS = WHOLESALE_EXTERNAL + WHOLESALE_FEATURES


@dataclass(frozen=True)
class WholesaleDataset:
    frame: pd.DataFrame
    variables: pd.DataFrame
    metadata: dict[str, Any]
    data_sha256: str


@dataclass(frozen=True)
class WholesaleSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray
    stratification_name: str
    split_sha256: str


@dataclass(frozen=True)
class WholesalePreprocessing:
    scaler: RobustScaler
    x_train: np.ndarray
    x_test: np.ndarray
    feature_names: list[str]
    fit_scope: str


def _to_plain_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_object(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain_object(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stable_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame[columns].to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    header = json.dumps(columns, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((header + "\n" + payload).encode("utf-8")).hexdigest()


def _combine_uci_features_and_targets(dataset: Any) -> pd.DataFrame:
    features = getattr(getattr(dataset, "data", None), "features", None)
    targets = getattr(getattr(dataset, "data", None), "targets", None)
    if features is None:
        raise ValueError("UCI dataset does not expose data.features")

    frame = pd.DataFrame(features).copy()
    if targets is not None:
        target_frame = pd.DataFrame(targets).copy()
        missing_target_columns = [column for column in target_frame if column not in frame]
        if missing_target_columns:
            frame = pd.concat([frame, target_frame[missing_target_columns]], axis=1)

    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def validate_wholesale_frame(
    frame: pd.DataFrame,
    *,
    expected_instances: int = 440,
) -> pd.DataFrame:
    missing_columns = sorted(set(WHOLESALE_EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing expected UCI columns: {missing_columns}")

    validated = frame[WHOLESALE_EXPECTED_COLUMNS].copy()
    for column in WHOLESALE_EXPECTED_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="raise")

    if len(validated) != int(expected_instances):
        raise ValueError(
            f"Expected {expected_instances} Wholesale customers, received {len(validated)}"
        )
    if validated.isna().any().any():
        raise ValueError(
            "The original Wholesale Customers data unexpectedly contains missing values"
        )
    if (validated[WHOLESALE_FEATURES] < 0).any().any():
        raise ValueError("Spending variables contain negative values")

    channel_values = set(validated["Channel"].astype(int).unique().tolist())
    region_values = set(validated["Region"].astype(int).unique().tolist())
    if not channel_values.issubset({1, 2}):
        raise ValueError(f"Unexpected Channel codes: {sorted(channel_values)}")
    if not region_values.issubset({1, 2, 3}):
        raise ValueError(f"Unexpected Region codes: {sorted(region_values)}")

    validated.insert(0, "customer_id", np.arange(1, len(validated) + 1, dtype=int))
    return validated


def wholesale_frame_from_uci_dataset(
    dataset: Any,
    *,
    expected_instances: int = 440,
) -> WholesaleDataset:
    frame = validate_wholesale_frame(
        _combine_uci_features_and_targets(dataset),
        expected_instances=expected_instances,
    )
    variables = pd.DataFrame(getattr(dataset, "variables", pd.DataFrame())).copy()
    metadata = _to_plain_object(getattr(dataset, "metadata", {}))
    data_hash = _stable_frame_hash(frame, ["customer_id"] + WHOLESALE_EXPECTED_COLUMNS)
    return WholesaleDataset(
        frame=frame,
        variables=variables,
        metadata=metadata,
        data_sha256=data_hash,
    )


def fetch_wholesale_customers(
    *,
    uci_id: int = WHOLESALE_UCI_ID,
    expected_instances: int = 440,
    fetcher: Callable[..., Any] | None = None,
) -> WholesaleDataset:
    if fetcher is None:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "The Wholesale case requires ucimlrepo==0.0.7. "
                "Install the pinned project dependencies."
            ) from exc
        fetcher = fetch_ucirepo

    dataset = fetcher(id=int(uci_id))
    return wholesale_frame_from_uci_dataset(
        dataset,
        expected_instances=expected_instances,
    )


def make_stratification_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        raise ValueError("At least one stratification column is required")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Unknown stratification columns: {missing}")
    return frame[columns].astype(str).agg("_".join, axis=1)


def _split_hash(train_indices: np.ndarray, test_indices: np.ndarray) -> str:
    payload = {
        "train": [int(value) for value in train_indices],
        "test": [int(value) for value in test_indices],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stratified_wholesale_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.30,
    seed: int = 42,
    stratify_columns: list[str] | None = None,
) -> WholesaleSplit:
    stratify_columns = stratify_columns or ["Channel", "Region"]
    strata = make_stratification_key(frame, stratify_columns)
    counts = strata.value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Channel x Region stratification is not possible because a stratum "
            "has fewer than two rows"
        )

    all_indices = np.arange(len(frame), dtype=int)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=strata,
        shuffle=True,
    )
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    if set(train_indices).intersection(test_indices):
        raise RuntimeError("TRAIN and TEST indices overlap")
    if len(train_indices) + len(test_indices) != len(frame):
        raise RuntimeError("TRAIN and TEST do not cover all observations")

    return WholesaleSplit(
        train=frame.iloc[train_indices].copy(),
        test=frame.iloc[test_indices].copy(),
        train_indices=train_indices,
        test_indices=test_indices,
        stratification_name="_x_".join(stratify_columns),
        split_sha256=_split_hash(train_indices, test_indices),
    )


def stratified_subsample(
    frame: pd.DataFrame,
    *,
    n_rows: int | None,
    seed: int,
    stratify_columns: list[str] | None = None,
) -> pd.DataFrame:
    if n_rows is None or int(n_rows) >= len(frame):
        return frame.copy()
    n_rows = int(n_rows)
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2")

    stratify_columns = stratify_columns or ["Channel", "Region"]
    candidate_strata = [
        stratify_columns,
        ["Channel"],
        [],
    ]
    all_indices = np.arange(len(frame), dtype=int)
    last_error: Exception | None = None

    for columns in candidate_strata:
        stratify = make_stratification_key(frame, columns) if columns else None
        try:
            selected, _ = train_test_split(
                all_indices,
                train_size=n_rows,
                random_state=int(seed),
                stratify=stratify,
                shuffle=True,
            )
            return frame.iloc[np.asarray(selected, dtype=int)].copy()
        except ValueError as exc:
            last_error = exc

    raise ValueError(f"Unable to construct smoke subsample: {last_error}")


def fit_wholesale_robust_scaler(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    quantile_range: tuple[float, float] = (25.0, 75.0),
) -> WholesalePreprocessing:
    feature_names = feature_names or WHOLESALE_FEATURES
    missing = [column for column in feature_names if column not in train or column not in test]
    if missing:
        raise ValueError(f"Missing preprocessing columns: {missing}")

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=tuple(float(value) for value in quantile_range),
        unit_variance=False,
    )
    x_train = scaler.fit_transform(train[feature_names].to_numpy(dtype=float))
    x_test = scaler.transform(test[feature_names].to_numpy(dtype=float))

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Robust scaling produced non-finite values")

    return WholesalePreprocessing(
        scaler=scaler,
        x_train=np.asarray(x_train, dtype=float),
        x_test=np.asarray(x_test, dtype=float),
        feature_names=list(feature_names),
        fit_scope="TRAIN complete before artificial missingness",
    )


def scaler_parameters_frame(preprocessing: WholesalePreprocessing) -> pd.DataFrame:
    scaler = preprocessing.scaler
    return pd.DataFrame(
        {
            "feature": preprocessing.feature_names,
            "center_median": np.asarray(scaler.center_, dtype=float),
            "scale_iqr": np.asarray(scaler.scale_, dtype=float),
            "fit_scope": preprocessing.fit_scope,
        }
    )
