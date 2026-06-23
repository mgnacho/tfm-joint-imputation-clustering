from __future__ import annotations

from typing import Any


def proposed_runtime_record(
    *,
    candidate_generation_runtime: float,
    train_total_runtime: float,
    test_total_runtime: float,
    data_loading_runtime: float = 0.0,
    preprocessing_runtime: float = 0.0,
    missingness_runtime: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    pipeline_runtime = (
        float(candidate_generation_runtime)
        + float(train_total_runtime)
        + float(test_total_runtime)
    )
    return {
        "model_type": "proposed",
        "data_loading_runtime": float(data_loading_runtime),
        "preprocessing_runtime": float(preprocessing_runtime),
        "missingness_runtime": float(missingness_runtime),
        "candidate_generation_runtime": float(candidate_generation_runtime),
        "imputation_runtime": float(candidate_generation_runtime),
        "clustering_runtime": 0.0,
        "train_total_runtime": float(train_total_runtime),
        "test_total_runtime": float(test_total_runtime),
        "pipeline_runtime": pipeline_runtime,
        "end_to_end_runtime": (
            float(data_loading_runtime)
            + float(preprocessing_runtime)
            + float(missingness_runtime)
            + pipeline_runtime
        ),
        **metadata,
    }


def baseline_runtime_record(
    *,
    imputation_runtime: float,
    clustering_runtime: float,
    data_loading_runtime: float = 0.0,
    preprocessing_runtime: float = 0.0,
    missingness_runtime: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    pipeline_runtime = float(imputation_runtime) + float(clustering_runtime)
    return {
        "model_type": "baseline",
        "data_loading_runtime": float(data_loading_runtime),
        "preprocessing_runtime": float(preprocessing_runtime),
        "missingness_runtime": float(missingness_runtime),
        "candidate_generation_runtime": 0.0,
        "imputation_runtime": float(imputation_runtime),
        "clustering_runtime": float(clustering_runtime),
        "train_total_runtime": 0.0,
        "test_total_runtime": 0.0,
        "pipeline_runtime": pipeline_runtime,
        "end_to_end_runtime": (
            float(data_loading_runtime)
            + float(preprocessing_runtime)
            + float(missingness_runtime)
            + pipeline_runtime
        ),
        **metadata,
    }
