from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from joint_imputation_clustering.analysis.aggregate import (
    aggregate_baselines,
    aggregate_proposed,
    gap_filtered_summary,
    oracle_baselines,
    paired_comparisons,
)
from joint_imputation_clustering.clustering.references import (
    fit_kmeans_reference,
    fit_pam_reference,
)
from joint_imputation_clustering.config import ExperimentConfig
from joint_imputation_clustering.data.missingness import (
    compute_train_bounds,
    induce_mcar_fixed_rate,
)
from joint_imputation_clustering.data.toy import (
    generate_noisy_toy_data,
    stratified_train_test_indices,
)
from joint_imputation_clustering.experiment.baselines import evaluate_baseline
from joint_imputation_clustering.imputation.candidates import (
    build_candidate_tensors_train_test,
)
from joint_imputation_clustering.metrics.evaluation import (
    evaluate_partition,
    feature_scales,
    safe_silhouette_scores,
)
from joint_imputation_clustering.models.complete_l1 import (
    solve_complete_l1_pmedian_reference,
)
from joint_imputation_clustering.models.joint_test import solve_joint_test_model
from joint_imputation_clustering.models.joint_train import solve_joint_train_model
from joint_imputation_clustering.utils.io import (
    atomic_write_csv,
    atomic_write_dataframe,
    configure_logging,
    copy_config,
    create_run_directory,
    write_manifest,
)
from joint_imputation_clustering.utils.seeding import stable_seed
from joint_imputation_clustering.visualization.plots import (
    create_summary_plots,
    plot_objective_components,
    plot_reference_comparison,
)

LOGGER = logging.getLogger(__name__)

ERROR_COLUMNS = [
    "scenario_id",
    "n_total",
    "d",
    "missing_rate_target",
    "missing_seed",
    "rho",
    "lambda_center",
    "stage",
    "error_type",
    "error_message",
]


class ResultStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "tables" / "raw"
        self.rows: dict[str, list[dict[str, Any]]] = {
            "proposed_test_results": [],
            "baseline_test_results": [],
            "reference_audit": [],
            "solver_diagnostics": [],
            "objective_components": [],
            "center_diagnostics": [],
            "method_selection": [],
            "candidate_audit": [],
            "missing_masks_long": [],
            "labels_long": [],
            "centers_long": [],
            "dataset_complete_long": [],
            "split_membership": [],
            "dataset_scale": [],
            "errors": [],
        }

    def save_raw(self) -> None:
        for name, rows in self.rows.items():
            columns = ERROR_COLUMNS if name == "errors" else None
            atomic_write_csv(rows, self.raw_dir / f"{name}.csv", columns=columns)

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.rows[name])


def _append_solver_row(
    store: ResultStore,
    *,
    record_type: str,
    stage: str,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    diagnostic_keys = [
        "status",
        "status_name",
        "is_certified_within_tolerance",
        "is_time_limit",
        "sol_count",
        "objective",
        "obj_bound",
        "gap",
        "absolute_gap",
        "gurobi_runtime",
        "wall_runtime",
        "node_count",
        "work",
        "num_vars",
        "num_binary_vars",
        "num_constraints",
        "max_coefficient",
        "min_coefficient",
        "max_bound",
        "max_rhs",
    ]
    store.rows["solver_diagnostics"].append(
        {
            "record_type": record_type,
            "stage": stage,
            **metadata,
            **{key: diagnostics.get(key, np.nan) for key in diagnostic_keys},
        }
    )


def _save_labels(
    store: ResultStore,
    *,
    model_name: str,
    record_type: str,
    split: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    for local_index, global_index in enumerate(indices):
        store.rows["labels_long"].append(
            {
                "record_type": record_type,
                "model_name": model_name,
                "split": split,
                "global_index": int(global_index),
                "local_index": int(local_index),
                "y_true": int(y_true[local_index]),
                "label": int(labels[local_index]),
                **metadata,
            }
        )


def _save_centers(
    store: ResultStore,
    *,
    model_name: str,
    record_type: str,
    centers: np.ndarray,
    metadata: dict[str, Any],
    local_indices: np.ndarray | None = None,
    global_indices: np.ndarray | None = None,
) -> None:
    for center_order, center in enumerate(centers):
        row: dict[str, Any] = {
            "record_type": record_type,
            "model_name": model_name,
            "center_order": center_order,
            "train_local_index": (
                int(local_indices[center_order]) if local_indices is not None else np.nan
            ),
            "dataset_global_index": (
                int(global_indices[center_order]) if global_indices is not None else np.nan
            ),
            **metadata,
        }
        for feature, value in enumerate(center):
            row[f"x{feature + 1}"] = float(value)
        store.rows["centers_long"].append(row)


def _reference_audit_rows(
    *,
    n_total: int,
    dimension: int,
    data_seed: int,
    split: str,
    y_true: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_pam: np.ndarray,
    labels_l1: np.ndarray,
    x_complete: np.ndarray,
    l1_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    kmeans_silhouette = safe_silhouette_scores(x_complete, labels_kmeans)
    pam_silhouette = safe_silhouette_scores(x_complete, labels_pam)
    l1_silhouette = safe_silhouette_scores(x_complete, labels_l1)
    return {
        "n_total": n_total,
        "d": dimension,
        "data_seed": data_seed,
        "split": split,
        "ari_kmeans_vs_pam": adjusted_rand_score(labels_kmeans, labels_pam),
        "ari_kmeans_vs_l1": adjusted_rand_score(labels_kmeans, labels_l1),
        "ari_pam_vs_l1": adjusted_rand_score(labels_pam, labels_l1),
        "ari_kmeans_true": adjusted_rand_score(y_true, labels_kmeans),
        "ari_pam_true": adjusted_rand_score(y_true, labels_pam),
        "ari_l1_true": adjusted_rand_score(y_true, labels_l1),
        "kmeans_sil_euclidean": kmeans_silhouette["euclidean"],
        "kmeans_sil_manhattan": kmeans_silhouette["manhattan"],
        "pam_sil_euclidean": pam_silhouette["euclidean"],
        "pam_sil_manhattan": pam_silhouette["manhattan"],
        "l1_sil_euclidean": l1_silhouette["euclidean"],
        "l1_sil_manhattan": l1_silhouette["manhattan"],
        "l1_reference_status": l1_diagnostics["status_name"],
        "l1_reference_gap": l1_diagnostics["gap"],
        "l1_reference_objective": l1_diagnostics["objective"],
        "l1_reference_obj_bound": l1_diagnostics["obj_bound"],
        "fallback_used": False,
    }


def _save_aggregates(store: ResultStore) -> None:
    aggregate_dir = store.run_dir / "tables" / "aggregated"
    proposed = store.frame("proposed_test_results")
    baselines = store.frame("baseline_test_results")

    outputs = {
        "proposed_test_agg.csv": aggregate_proposed(proposed),
        "baseline_test_agg.csv": aggregate_baselines(baselines),
        "gap_filtered_summary.csv": gap_filtered_summary(proposed),
        "paired_proposed_vs_baselines.csv": paired_comparisons(proposed, baselines),
        "oracle_baseline_by_scenario.csv": oracle_baselines(baselines),
    }
    for filename, frame in outputs.items():
        atomic_write_dataframe(frame, aggregate_dir / filename)


def _write_summary(store: ResultStore, started: float) -> None:
    proposed = store.frame("proposed_test_results")
    solver = store.frame("solver_diagnostics")
    errors = store.frame("errors")
    summary = {
        "duration_seconds": time.perf_counter() - started,
        "proposed_runs": int(len(proposed)),
        "baseline_runs": int(len(store.rows["baseline_test_results"])),
        "errors": int(len(errors)),
        "train_certified_2pct": int(proposed.get("train_is_certified", pd.Series(dtype=bool)).sum()),
        "train_time_limits": int(proposed.get("train_is_time_limit", pd.Series(dtype=bool)).sum()),
        "solver_records": int(len(solver)),
    }
    with (store.run_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    LOGGER.info("Run summary: %s", summary)


def run_experiment(
    config: ExperimentConfig,
    repository_root: Path,
    log_level: str = "INFO",
) -> Path:
    started = time.perf_counter()
    output_root = config.output_root
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    run_dir = create_run_directory(output_root, config.experiment_name)
    configure_logging(run_dir / "logs" / "run.log", log_level)
    store = ResultStore(run_dir)
    copy_config(config.source_path, run_dir)
    write_manifest(run_dir, config.raw, repository_root)

    problem = config.section("problem")
    randomness = config.section("randomness")
    hyperparameters = config.section("hyperparameters")
    imputation_config = config.section("imputation")
    solver = config.section("solver")

    k = int(problem["k"])
    n_values = [int(value) for value in problem["n_values"]]
    dimensions = [int(value) for value in problem["d_values"]]
    missing_rates = [float(value) for value in problem["missing_rates"]]
    missing_seeds = [int(value) for value in problem["missing_seeds"]]
    rho_values = [float(value) for value in hyperparameters["rho_values"]]
    lambda_values = [float(value) for value in hyperparameters["lambda_values"]]
    dataset_seeds = {int(key): int(value) for key, value in randomness["dataset_seeds"].items()}

    scenario_id = 0
    experiment_id = 0

    for n_total in n_values:
        for dimension in dimensions:
            data_seed = dataset_seeds[dimension]
            LOGGER.info("Dataset n=%s d=%s seed=%s", n_total, dimension, data_seed)
            x_complete, y_true, is_outlier = generate_noisy_toy_data(
                n_total=n_total,
                dimension=dimension,
                k=k,
                seed=data_seed,
                outlier_rate=float(problem["outlier_rate"]),
            )
            idx_train, idx_test = stratified_train_test_indices(
                y_true,
                n_total=n_total,
                dimension=dimension,
                train_fraction=float(problem["train_fraction"]),
            )
            x_train_complete = x_complete[idx_train]
            x_test_complete = x_complete[idx_test]
            y_train_true = y_true[idx_train]
            y_test_true = y_true[idx_test]
            train_scales = feature_scales(x_train_complete)

            for global_index in range(n_total):
                split = "train" if global_index in set(idx_train.tolist()) else "test"
                row = {
                    "n_total": n_total,
                    "d": dimension,
                    "global_index": global_index,
                    "split": split,
                    "y_true": int(y_true[global_index]),
                    "is_outlier": bool(is_outlier[global_index]),
                    "data_seed": data_seed,
                }
                for feature in range(dimension):
                    row[f"x{feature + 1}"] = float(x_complete[global_index, feature])
                store.rows["dataset_complete_long"].append(row)

            for split, indices in [("train", idx_train), ("test", idx_test)]:
                for local_index, global_index in enumerate(indices):
                    store.rows["split_membership"].append(
                        {
                            "n_total": n_total,
                            "d": dimension,
                            "global_index": int(global_index),
                            "local_index": int(local_index),
                            "split": split,
                            "y_true": int(y_true[global_index]),
                            "is_outlier": bool(is_outlier[global_index]),
                            "data_seed": data_seed,
                        }
                    )

            scale_frame = pd.DataFrame(
                x_complete, columns=[f"x{feature + 1}" for feature in range(dimension)]
            ).agg(["min", "max", "mean", "std", "var"]).T.reset_index(names="variable")
            scale_frame["n_total"] = n_total
            scale_frame["d"] = dimension
            store.rows["dataset_scale"].extend(scale_frame.to_dict("records"))

            reference_seed = stable_seed(n_total, dimension, "references", base=22_000)
            kmeans_reference = fit_kmeans_reference(
                x_train_complete, x_test_complete, k, reference_seed
            )
            pam_reference = fit_pam_reference(
                x_train_complete, x_test_complete, k, reference_seed
            )
            l1_reference = solve_complete_l1_pmedian_reference(
                x_train_complete,
                x_test_complete,
                k,
                time_limit=float(solver["reference_time_limit"]),
                mip_gap=float(solver["reference_mip_gap"]),
                output_flag=int(solver["output_flag"]),
                solver_seed=stable_seed(n_total, dimension, "reference_l1", base=25_000),
                threads=int(solver["threads"]),
            )
            _append_solver_row(
                store,
                record_type="reference",
                stage="train_reference_l1_complete",
                diagnostics=l1_reference,
                metadata={
                    "n_total": n_total,
                    "d": dimension,
                    "missing_rate_target": 0.0,
                    "missing_seed": 0,
                    "rho": 0.0,
                    "lambda_center": 0.0,
                },
            )

            for split, x_split, y_split, labels_km, labels_pam, labels_l1 in [
                (
                    "train",
                    x_train_complete,
                    y_train_true,
                    kmeans_reference.labels_train,
                    pam_reference.labels_train,
                    l1_reference["labels_train"],
                ),
                (
                    "test",
                    x_test_complete,
                    y_test_true,
                    kmeans_reference.labels_test,
                    pam_reference.labels_test,
                    l1_reference["labels_test"],
                ),
            ]:
                store.rows["reference_audit"].append(
                    _reference_audit_rows(
                        n_total=n_total,
                        dimension=dimension,
                        data_seed=data_seed,
                        split=split,
                        y_true=y_split,
                        labels_kmeans=labels_km,
                        labels_pam=labels_pam,
                        labels_l1=labels_l1,
                        x_complete=x_split,
                        l1_diagnostics=l1_reference,
                    )
                )

            reference_metadata = {
                "n_total": n_total,
                "d": dimension,
                "missing_rate_target": 0.0,
                "missing_seed": 0,
                "rho": np.nan,
                "lambda_center": np.nan,
            }
            for model_name, labels_train, labels_test in [
                ("reference_kmeans_complete", kmeans_reference.labels_train, kmeans_reference.labels_test),
                ("reference_pam_complete", pam_reference.labels_train, pam_reference.labels_test),
                ("reference_l1_complete", l1_reference["labels_train"], l1_reference["labels_test"]),
            ]:
                _save_labels(
                    store,
                    model_name=model_name,
                    record_type="reference",
                    split="train",
                    indices=idx_train,
                    y_true=y_train_true,
                    labels=labels_train,
                    metadata=reference_metadata,
                )
                _save_labels(
                    store,
                    model_name=model_name,
                    record_type="reference",
                    split="test",
                    indices=idx_test,
                    y_true=y_test_true,
                    labels=labels_test,
                    metadata=reference_metadata,
                )

            _save_centers(
                store,
                model_name="reference_kmeans_complete",
                record_type="reference",
                centers=kmeans_reference.centers,
                metadata=reference_metadata,
            )
            pam_global = idx_train[pam_reference.center_indices_train]
            _save_centers(
                store,
                model_name="reference_pam_complete",
                record_type="reference",
                centers=pam_reference.centers,
                local_indices=pam_reference.center_indices_train,
                global_indices=pam_global,
                metadata=reference_metadata,
            )
            l1_global = idx_train[l1_reference["centers_idx"]]
            _save_centers(
                store,
                model_name="reference_l1_complete",
                record_type="reference",
                centers=l1_reference["centers_values"],
                local_indices=l1_reference["centers_idx"],
                global_indices=l1_global,
                metadata=reference_metadata,
            )

            if config.create_plots:
                plot_reference_comparison(
                    x_test_complete,
                    kmeans_reference.labels_test,
                    pam_reference.labels_test,
                    l1_reference["labels_test"],
                    kmeans_reference.centers,
                    pam_reference.centers,
                    l1_reference["centers_values"],
                    k,
                    run_dir / "figures" / f"references_n{n_total}_d{dimension}.png",
                )

            for missing_rate in missing_rates:
                for missing_seed in missing_seeds:
                    scenario_id += 1
                    LOGGER.info(
                        "Scenario %s n=%s d=%s missing=%s seed=%s",
                        scenario_id,
                        n_total,
                        dimension,
                        missing_rate,
                        missing_seed,
                    )
                    try:
                        x_train_missing, mask_train = induce_mcar_fixed_rate(
                            x_train_complete,
                            missing_rate,
                            stable_seed(dimension, missing_rate, missing_seed, "mask_train", base=30_000),
                        )
                        x_test_missing, mask_test = induce_mcar_fixed_rate(
                            x_test_complete,
                            missing_rate,
                            stable_seed(dimension, missing_rate, missing_seed, "mask_test", base=32_000),
                        )
                        for split, indices, mask in [
                            ("train", idx_train, mask_train),
                            ("test", idx_test, mask_test),
                        ]:
                            for local_index, feature in np.argwhere(mask):
                                store.rows["missing_masks_long"].append(
                                    {
                                        "scenario_id": scenario_id,
                                        "n_total": n_total,
                                        "d": dimension,
                                        "missing_rate_target": missing_rate,
                                        "missing_seed": missing_seed,
                                        "split": split,
                                        "local_index": int(local_index),
                                        "global_index": int(indices[local_index]),
                                        "feature_index": int(feature),
                                        "feature_name": f"x{feature + 1}",
                                    }
                                )

                        lower, upper = compute_train_bounds(
                            x_train_missing,
                            margin_fraction=float(problem["bounds_margin_fraction"]),
                        )
                        candidate_seed = stable_seed(
                            n_total, dimension, missing_rate, "candidates_fixed", base=40_000
                        )
                        candidates = build_candidate_tensors_train_test(
                            x_train_missing,
                            x_test_missing,
                            lower,
                            upper,
                            requested_names=list(imputation_config["candidate_names"]),
                            seed=candidate_seed,
                            mode_decimals=int(imputation_config["mode_decimals"]),
                            knn_neighbors=int(imputation_config["knn_neighbors"]),
                            pmm_donors=int(imputation_config["pmm_donors"]),
                        )
                        audit = candidates.audit.copy()
                        audit["scenario_id"] = scenario_id
                        audit["n_total"] = n_total
                        audit["d"] = dimension
                        audit["missing_rate_target"] = missing_rate
                        audit["missing_seed"] = missing_seed
                        store.rows["candidate_audit"].extend(audit.to_dict("records"))

                        for candidate_index, method_name in enumerate(candidates.names):
                            for cluster_algorithm in ["kmeans", "pam"]:
                                baseline_seed = stable_seed(
                                    n_total,
                                    dimension,
                                    method_name,
                                    cluster_algorithm,
                                    "baseline",
                                    base=50_000,
                                )
                                baseline = evaluate_baseline(
                                    method_name=method_name,
                                    cluster_algorithm=cluster_algorithm,
                                    x_train_imputed=candidates.train[candidate_index],
                                    x_test_imputed=candidates.test[candidate_index],
                                    x_test_complete=x_test_complete,
                                    mask_test=mask_test,
                                    k=k,
                                    seed=baseline_seed,
                                    labels_ref_kmeans_test=kmeans_reference.labels_test,
                                    labels_ref_pam_test=pam_reference.labels_test,
                                    labels_ref_l1_test=l1_reference["labels_test"],
                                    y_test_true=y_test_true,
                                    train_scales=train_scales,
                                    imputation_runtime=candidates.runtimes[method_name],
                                )
                                baseline_row = {
                                    "scenario_id": scenario_id,
                                    "n_total": n_total,
                                    "d": dimension,
                                    "n_train": len(idx_train),
                                    "n_test": len(idx_test),
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "missing_rate_train_real": float(mask_train.mean()),
                                    "missing_rate_test_real": float(mask_test.mean()),
                                    **{
                                        key: value
                                        for key, value in baseline.items()
                                        if key
                                        not in {
                                            "labels_train",
                                            "labels_test",
                                            "centers_values",
                                            "center_indices_train",
                                        }
                                    },
                                }
                                store.rows["baseline_test_results"].append(baseline_row)
                                baseline_metadata = {
                                    "n_total": n_total,
                                    "d": dimension,
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "rho": np.nan,
                                    "lambda_center": np.nan,
                                }
                                _save_labels(
                                    store,
                                    model_name=f"{method_name}_{cluster_algorithm}",
                                    record_type="baseline_test",
                                    split="test",
                                    indices=idx_test,
                                    y_true=y_test_true,
                                    labels=baseline["labels_test"],
                                    metadata=baseline_metadata,
                                )
                                local_indices = baseline["center_indices_train"]
                                global_indices = (
                                    idx_train[local_indices] if local_indices is not None else None
                                )
                                _save_centers(
                                    store,
                                    model_name=f"{method_name}_{cluster_algorithm}",
                                    record_type="baseline_test",
                                    centers=baseline["centers_values"],
                                    local_indices=local_indices,
                                    global_indices=global_indices,
                                    metadata=baseline_metadata,
                                )

                        for rho in rho_values:
                            for lambda_center in lambda_values:
                                experiment_id += 1
                                train_result = solve_joint_train_model(
                                    x_train_missing,
                                    candidates.train,
                                    candidates.names,
                                    lower,
                                    upper,
                                    k,
                                    rho,
                                    lambda_center,
                                    time_limit=float(solver["train_time_limit"]),
                                    mip_gap=float(solver["mip_gap"]),
                                    mip_focus=int(solver["mip_focus_train"]),
                                    output_flag=int(solver["output_flag"]),
                                    solver_seed=stable_seed(
                                        n_total,
                                        dimension,
                                        rho,
                                        lambda_center,
                                        "train_solver",
                                        base=60_000,
                                    ),
                                    threads=int(solver["threads"]),
                                    x_train_complete_reference=x_train_complete,
                                    original_train_indices=idx_train,
                                )
                                test_result = solve_joint_test_model(
                                    x_test_missing,
                                    train_result["centers_values"],
                                    candidates.test,
                                    train_result["chosen_methods"],
                                    lower,
                                    upper,
                                    rho,
                                    time_limit=float(solver["test_time_limit"]),
                                    mip_gap=float(solver["mip_gap"]),
                                    output_flag=int(solver["output_flag"]),
                                    solver_seed=stable_seed(
                                        n_total,
                                        dimension,
                                        rho,
                                        lambda_center,
                                        "test_solver",
                                        base=62_000,
                                    ),
                                    threads=int(solver["threads"]),
                                )
                                metrics = evaluate_partition(
                                    x_complete=x_test_complete,
                                    x_imputed=test_result["X_imputed_test"],
                                    labels=test_result["labels_test"],
                                    missing_mask=mask_test,
                                    labels_ref_kmeans=kmeans_reference.labels_test,
                                    labels_ref_pam=pam_reference.labels_test,
                                    labels_ref_l1=l1_reference["labels_test"],
                                    y_true=y_test_true,
                                    centers=train_result["centers_values"],
                                    train_scales=train_scales,
                                )
                                center_info = train_result["center_missing_info"].copy()
                                center_info["experiment_id"] = experiment_id
                                center_info["scenario_id"] = scenario_id
                                center_info["n_total"] = n_total
                                center_info["d"] = dimension
                                center_info["missing_rate_target"] = missing_rate
                                center_info["missing_seed"] = missing_seed
                                center_info["rho"] = rho
                                center_info["lambda_center"] = lambda_center
                                store.rows["center_diagnostics"].extend(center_info.to_dict("records"))

                                for feature in range(dimension):
                                    store.rows["method_selection"].append(
                                        {
                                            "experiment_id": experiment_id,
                                            "scenario_id": scenario_id,
                                            "n_total": n_total,
                                            "d": dimension,
                                            "missing_rate_target": missing_rate,
                                            "missing_seed": missing_seed,
                                            "rho": rho,
                                            "lambda_center": lambda_center,
                                            "variable": f"x{feature + 1}",
                                            "chosen_method": train_result["chosen_method_names"][feature],
                                        }
                                    )

                                n_incomplete = int(center_info["was_originally_incomplete"].sum())
                                row = {
                                    "experiment_id": experiment_id,
                                    "scenario_id": scenario_id,
                                    "n_total": n_total,
                                    "d": dimension,
                                    "n_train": len(idx_train),
                                    "n_test": len(idx_test),
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "missing_rate_train_real": float(mask_train.mean()),
                                    "missing_rate_test_real": float(mask_test.mean()),
                                    "rho": rho,
                                    "lambda_center": lambda_center,
                                    "objective_fully_normalized": True,
                                    "train_status": train_result["status"],
                                    "train_status_name": train_result["status_name"],
                                    "train_is_certified": train_result[
                                        "is_certified_within_tolerance"
                                    ],
                                    "train_is_time_limit": train_result["is_time_limit"],
                                    "train_sol_count": train_result["sol_count"],
                                    "train_objective": train_result["objective"],
                                    "train_obj_bound": train_result["obj_bound"],
                                    "train_gap": train_result["gap"],
                                    "train_absolute_gap": train_result["absolute_gap"],
                                    "train_runtime": train_result["wall_runtime"],
                                    "test_status": test_result["status"],
                                    "test_status_name": test_result["status_name"],
                                    "test_is_certified": test_result[
                                        "is_certified_within_tolerance"
                                    ],
                                    "test_gap": test_result["gap"],
                                    "test_absolute_gap": test_result["absolute_gap"],
                                    "test_runtime": test_result["wall_runtime"],
                                    "n_incomplete_centers": n_incomplete,
                                    "mean_center_l1_displacement": float(
                                        center_info["l1_displacement"].mean()
                                    ),
                                    "mean_center_l2_displacement": float(
                                        center_info["l2_displacement"].mean()
                                    ),
                                    "train_clustering_normalized": train_result[
                                        "objective_clustering_normalized"
                                    ],
                                    "train_imputation_normalized": train_result[
                                        "objective_imputation_normalized"
                                    ],
                                    "train_center_penalty_normalized": train_result[
                                        "objective_center_penalty_normalized"
                                    ],
                                    "train_objective_reconstructed": train_result[
                                        "objective_reconstructed"
                                    ],
                                    "test_clustering_normalized": test_result[
                                        "objective_clustering_normalized"
                                    ],
                                    "test_imputation_normalized": test_result[
                                        "objective_imputation_normalized"
                                    ],
                                    "test_objective_reconstructed": test_result[
                                        "objective_reconstructed"
                                    ],
                                    **metrics,
                                }
                                store.rows["proposed_test_results"].append(row)

                                objective_row = {
                                    "experiment_id": experiment_id,
                                    "scenario_id": scenario_id,
                                    "n_total": n_total,
                                    "d": dimension,
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "rho": rho,
                                    "lambda_center": lambda_center,
                                }
                                for stage, result in [("train", train_result), ("test", test_result)]:
                                    store.rows["objective_components"].append(
                                        {
                                            **objective_row,
                                            "stage": stage,
                                            "clustering_raw": result[
                                                "objective_clustering_raw"
                                            ],
                                            "clustering_normalized": result[
                                                "objective_clustering_normalized"
                                            ],
                                            "imputation_raw": result[
                                                "objective_imputation_raw"
                                            ],
                                            "imputation_normalized": result[
                                                "objective_imputation_normalized"
                                            ],
                                            "imputation_weighted": result[
                                                "objective_imputation_weighted"
                                            ],
                                            "center_penalty_raw": result.get(
                                                "objective_center_penalty_raw", 0.0
                                            ),
                                            "center_penalty_normalized": result.get(
                                                "objective_center_penalty_normalized", 0.0
                                            ),
                                            "center_penalty_weighted": result.get(
                                                "objective_center_penalty_weighted", 0.0
                                            ),
                                            "objective_reconstructed": result[
                                                "objective_reconstructed"
                                            ],
                                            "objective_solver": result["objective"],
                                        }
                                    )

                                metadata = {
                                    "experiment_id": experiment_id,
                                    "scenario_id": scenario_id,
                                    "n_total": n_total,
                                    "d": dimension,
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "rho": rho,
                                    "lambda_center": lambda_center,
                                }
                                _append_solver_row(
                                    store,
                                    record_type="proposed",
                                    stage="train",
                                    diagnostics=train_result,
                                    metadata=metadata,
                                )
                                _append_solver_row(
                                    store,
                                    record_type="proposed",
                                    stage="test",
                                    diagnostics=test_result,
                                    metadata=metadata,
                                )
                                _save_labels(
                                    store,
                                    model_name="joint_l1_pmedian",
                                    record_type="proposed_test",
                                    split="test",
                                    indices=idx_test,
                                    y_true=y_test_true,
                                    labels=test_result["labels_test"],
                                    metadata=metadata,
                                )
                                global_centers = idx_train[train_result["centers_idx"]]
                                _save_centers(
                                    store,
                                    model_name="joint_l1_pmedian",
                                    record_type="proposed_test",
                                    centers=train_result["centers_values"],
                                    local_indices=train_result["centers_idx"],
                                    global_indices=global_centers,
                                    metadata=metadata,
                                )
                                if config.save_incremental:
                                    store.save_raw()

                        if config.save_incremental:
                            store.save_raw()
                        gc.collect()

                    except Exception as exc:
                        LOGGER.exception("Scenario %s failed", scenario_id)
                        store.rows["errors"].append(
                            {
                                "scenario_id": scenario_id,
                                "n_total": n_total,
                                "d": dimension,
                                "missing_rate_target": missing_rate,
                                "missing_seed": missing_seed,
                                "rho": np.nan,
                                "lambda_center": np.nan,
                                "stage": "scenario",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        store.save_raw()
                        if not config.continue_on_error:
                            raise

    store.save_raw()
    _save_aggregates(store)
    if config.create_plots:
        proposed = store.frame("proposed_test_results")
        baselines = store.frame("baseline_test_results")
        create_summary_plots(proposed, baselines, run_dir / "figures")
        plot_objective_components(proposed, run_dir / "figures")
    _write_summary(store, started)
    return run_dir
