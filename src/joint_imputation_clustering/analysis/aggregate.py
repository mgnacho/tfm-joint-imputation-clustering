from __future__ import annotations

import pandas as pd


def aggregate_proposed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["d", "missing_rate_target", "rho", "lambda_center"], as_index=False)
        .agg(
            ari_l1_mean=("ari_ref_l1model", "mean"),
            ari_l1_std=("ari_ref_l1model", "std"),
            ari_l1_min=("ari_ref_l1model", "min"),
            ari_l1_max=("ari_ref_l1model", "max"),
            ari_pam_mean=("ari_ref_pam", "mean"),
            ari_kmeans_mean=("ari_ref_kmeans", "mean"),
            ari_true_mean=("ari_true", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            silhouette_common_manhattan_mean=("silhouette_common_manhattan", "mean"),
            silhouette_common_euclidean_mean=("silhouette_common_euclidean", "mean"),
            common_l1_compactness_mean=("common_l1_compactness", "mean"),
            train_gap_mean=("train_gap", "mean"),
            train_gap_median=("train_gap", "median"),
            certified_rate_2pct=("train_is_certified", "mean"),
            time_limit_rate=("train_is_time_limit", "mean"),
            incomplete_centers_mean=("n_incomplete_centers", "mean"),
            center_l1_displacement_mean=("mean_center_l1_displacement", "mean"),
            n_successful=("scenario_id", "count"),
            n_seeds=("missing_seed", "nunique"),
        )
    )


def aggregate_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["d", "missing_rate_target", "method", "cluster_algo"], as_index=False)
        .agg(
            ari_l1_mean=("ari_ref_l1model", "mean"),
            ari_l1_std=("ari_ref_l1model", "std"),
            ari_pam_mean=("ari_ref_pam", "mean"),
            ari_kmeans_mean=("ari_ref_kmeans", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            silhouette_common_manhattan_mean=("silhouette_common_manhattan", "mean"),
            common_l1_compactness_mean=("common_l1_compactness", "mean"),
            total_runtime_mean=("total_runtime", "mean"),
            n_successful=("scenario_id", "count"),
        )
    )


def gap_filtered_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subsets = [
        ("all_feasible", frame),
        ("gap_le_20pct", frame[frame["train_gap"] <= 0.20]),
        ("gap_le_10pct", frame[frame["train_gap"] <= 0.10]),
        ("gap_le_5pct", frame[frame["train_gap"] <= 0.05]),
        ("certified_2pct", frame[frame["train_is_certified"]]),
    ]
    results: list[pd.DataFrame] = []
    for name, subset in subsets:
        if subset.empty:
            continue
        summary = (
            subset.groupby(["d", "missing_rate_target", "rho", "lambda_center"], as_index=False)
            .agg(
                n=("scenario_id", "count"),
                ari_l1_mean=("ari_ref_l1model", "mean"),
                rmse_mean=("rmse", "mean"),
                silhouette_common_manhattan_mean=("silhouette_common_manhattan", "mean"),
                train_gap_mean=("train_gap", "mean"),
            )
        )
        summary["subset"] = name
        results.append(summary)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def paired_comparisons(proposed: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    if proposed.empty or baselines.empty:
        return pd.DataFrame()
    keys = ["scenario_id", "n_total", "d", "missing_rate_target", "missing_seed"]
    baseline_for_merge = baselines.drop(
        columns=["rho", "lambda_center"],
        errors="ignore",
    )
    merged = proposed.merge(
        baseline_for_merge,
        on=keys,
        suffixes=("_proposed", "_baseline"),
    )
    output = merged[
        keys
        + [
            "rho",
            "lambda_center",
            "method",
            "cluster_algo",
            "ari_ref_l1model_proposed",
            "ari_ref_l1model_baseline",
            "rmse_proposed",
            "rmse_baseline",
            "silhouette_common_manhattan_proposed",
            "silhouette_common_manhattan_baseline",
        ]
    ].copy()
    output["delta_ari_l1"] = (
        output["ari_ref_l1model_proposed"] - output["ari_ref_l1model_baseline"]
    )
    output["delta_rmse"] = output["rmse_proposed"] - output["rmse_baseline"]
    output["delta_silhouette_manhattan"] = (
        output["silhouette_common_manhattan_proposed"]
        - output["silhouette_common_manhattan_baseline"]
    )
    return output


def oracle_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    keys = ["scenario_id", "n_total", "d", "missing_rate_target", "missing_seed"]
    rows: list[dict[str, object]] = []
    for key_values, group in frame.groupby(keys):
        base = dict(zip(keys, key_values if isinstance(key_values, tuple) else [key_values]))
        best_ari = group.sort_values(
            ["ari_ref_l1model", "ari_ref_pam", "silhouette_common_manhattan", "rmse"],
            ascending=[False, False, False, True],
        ).iloc[0]
        best_rmse = group.sort_values(["rmse", "ari_ref_l1model"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                **base,
                "criterion": "ari_oracle_test",
                "method": best_ari["method"],
                "cluster_algo": best_ari["cluster_algo"],
                "ari_ref_l1model": best_ari["ari_ref_l1model"],
                "rmse": best_ari["rmse"],
            }
        )
        rows.append(
            {
                **base,
                "criterion": "rmse_oracle_test",
                "method": best_rmse["method"],
                "cluster_algo": best_rmse["cluster_algo"],
                "ari_ref_l1model": best_rmse["ari_ref_l1model"],
                "rmse": best_rmse["rmse"],
            }
        )
    return pd.DataFrame(rows)
