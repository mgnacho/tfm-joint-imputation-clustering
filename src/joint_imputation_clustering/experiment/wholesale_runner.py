from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
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
from joint_imputation_clustering.analysis.marketing import (
    build_marketing_assignments,
    save_marketing_plots,
    summarize_marketing_assignments,
)
from joint_imputation_clustering.analysis.runtime import (
    baseline_runtime_record,
    proposed_runtime_record,
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
from joint_imputation_clustering.data.wholesale import (
    WHOLESALE_FEATURES,
    fetch_wholesale_customers,
    fit_wholesale_robust_scaler,
    scaler_parameters_frame,
    stratified_subsample,
    stratified_wholesale_split,
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
    atomic_write_dataframe,
    configure_logging,
    copy_config,
    create_run_directory,
    write_manifest,
)
from joint_imputation_clustering.utils.seeding import stable_seed

LOGGER = logging.getLogger(__name__)


class WholesaleResultStore:
    TABLES = [
        "proposed_test_results",
        "baseline_test_results",
        "reference_audit",
        "solver_diagnostics",
        "objective_components",
        "center_diagnostics",
        "method_selection",
        "candidate_audit",
        "missing_masks_long",
        "labels_long",
        "centers_long",
        "split_membership",
        "runtime_accounting",
        "marketing_assignments",
        "marketing_cluster_profiles",
        "marketing_association",
        "errors",
    ]

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "tables" / "raw"
        self.rows: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.TABLES
        }

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.rows[name])

    def save_raw(self) -> None:
        for name in self.TABLES:
            atomic_write_dataframe(
                self.frame(name),
                self.raw_dir / f"{name}.csv",
            )


def _append_solver_diagnostics(
    store: WholesaleResultStore,
    *,
    record_type: str,
    stage: str,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
    total_call_runtime: float,
) -> None:
    keys = [
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
            "total_call_runtime": float(total_call_runtime),
            **metadata,
            **{key: diagnostics.get(key, np.nan) for key in keys},
        }
    )


def _save_labels(
    store: WholesaleResultStore,
    *,
    model_name: str,
    record_type: str,
    split: str,
    customer_ids: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    for local_index, (customer_id, label) in enumerate(zip(customer_ids, labels)):
        store.rows["labels_long"].append(
            {
                "record_type": record_type,
                "model_name": model_name,
                "split": split,
                "customer_id": int(customer_id),
                "local_index": int(local_index),
                "label": int(label),
                **metadata,
            }
        )


def _save_centers(
    store: WholesaleResultStore,
    *,
    model_name: str,
    record_type: str,
    centers: np.ndarray,
    metadata: dict[str, Any],
    local_indices: np.ndarray | None = None,
    customer_ids: np.ndarray | None = None,
) -> None:
    for order, center in enumerate(np.asarray(centers, dtype=float)):
        row: dict[str, Any] = {
            "record_type": record_type,
            "model_name": model_name,
            "center_order": int(order),
            "train_local_index": (
                int(local_indices[order]) if local_indices is not None else np.nan
            ),
            "customer_id": (
                int(customer_ids[order]) if customer_ids is not None else np.nan
            ),
            **metadata,
        }
        for feature_name, value in zip(WHOLESALE_FEATURES, center):
            row[feature_name] = float(value)
        store.rows["centers_long"].append(row)


def _append_marketing_outputs(
    store: WholesaleResultStore,
    *,
    raw_frame: pd.DataFrame,
    train_customer_ids: np.ndarray,
    test_customer_ids: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    model_name: str,
    metadata: dict[str, Any],
    create_plot: bool,
    plot_dir: Path,
) -> None:
    assignments = build_marketing_assignments(
        raw_frame,
        train_customer_ids=train_customer_ids,
        test_customer_ids=test_customer_ids,
        labels_train=labels_train,
        labels_test=labels_test,
        model_name=model_name,
        metadata=metadata,
    )
    metadata_columns = list(metadata)
    profiles, association = summarize_marketing_assignments(
        assignments,
        metadata_columns=metadata_columns,
    )
    store.rows["marketing_assignments"].extend(assignments.to_dict("records"))
    store.rows["marketing_cluster_profiles"].extend(profiles.to_dict("records"))
    store.rows["marketing_association"].extend(association.to_dict("records"))

    if create_plot:
        prefix = (
            model_name.replace("/", "_")
            .replace(" ", "_")
            .replace(".", "p")
        )
        save_marketing_plots(
            assignments,
            profiles,
            plot_dir,
            filename_prefix=prefix,
        )


def _reference_audit(
    *,
    split: str,
    x_complete: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_pam: np.ndarray,
    labels_l1: np.ndarray,
    l1_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    kmeans_silhouette = safe_silhouette_scores(x_complete, labels_kmeans)
    pam_silhouette = safe_silhouette_scores(x_complete, labels_pam)
    l1_silhouette = safe_silhouette_scores(x_complete, labels_l1)
    return {
        "split": split,
        "ari_kmeans_vs_pam": adjusted_rand_score(labels_kmeans, labels_pam),
        "ari_kmeans_vs_l1": adjusted_rand_score(labels_kmeans, labels_l1),
        "ari_pam_vs_l1": adjusted_rand_score(labels_pam, labels_l1),
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
    }


def _save_aggregates(store: WholesaleResultStore) -> None:
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


def run_wholesale_experiment(
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
    store = WholesaleResultStore(run_dir)
    copy_config(config.source_path, run_dir)
    write_manifest(run_dir, config.raw, repository_root)

    data_config = config.section("data")
    problem = config.section("problem")
    randomness = config.section("randomness")
    preprocessing_config = config.section("preprocessing")
    hyperparameters = config.section("hyperparameters")
    imputation_config = config.section("imputation")
    solver = config.section("solver")
    marketing = config.optional_section("marketing")

    loading_started = time.perf_counter()
    wholesale = fetch_wholesale_customers(
        uci_id=int(data_config.get("uci_id", 292)),
        expected_instances=int(data_config.get("expected_instances", 440)),
    )
    data_loading_runtime = time.perf_counter() - loading_started

    full_frame = wholesale.frame
    atomic_write_dataframe(
        full_frame,
        run_dir / "tables" / "raw" / "wholesale_source_snapshot.csv",
    )
    atomic_write_dataframe(
        wholesale.variables,
        run_dir / "tables" / "raw" / "wholesale_variables.csv",
    )
    with (run_dir / "wholesale_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "uci_id": int(data_config.get("uci_id", 292)),
                "data_sha256": wholesale.data_sha256,
                "metadata": wholesale.metadata,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    split = stratified_wholesale_split(
        full_frame,
        test_size=1.0 - float(problem["train_fraction"]),
        seed=int(randomness["split_seed"]),
        stratify_columns=list(data_config["stratify_columns"]),
    )
    train_frame = stratified_subsample(
        split.train,
        n_rows=data_config.get("max_train_rows"),
        seed=stable_seed(randomness["split_seed"], "smoke_train"),
        stratify_columns=list(data_config["stratify_columns"]),
    )
    test_frame = stratified_subsample(
        split.test,
        n_rows=data_config.get("max_test_rows"),
        seed=stable_seed(randomness["split_seed"], "smoke_test"),
        stratify_columns=list(data_config["stratify_columns"]),
    )
    used_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    used_train_ids = set(train_frame["customer_id"].astype(int))
    used_test_ids = set(test_frame["customer_id"].astype(int))
    for split_name, frame, used_ids in [
        ("train", split.train, used_train_ids),
        ("test", split.test, used_test_ids),
    ]:
        for local_index, row in enumerate(frame.itertuples(index=False)):
            customer_id = int(row.customer_id)
            store.rows["split_membership"].append(
                {
                    "split": split_name,
                    "local_index_in_full_split": local_index,
                    "customer_id": customer_id,
                    "Channel": int(row.Channel),
                    "Region": int(row.Region),
                    "used_in_run": customer_id in used_ids,
                    "split_seed": int(randomness["split_seed"]),
                    "stratification": split.stratification_name,
                    "full_split_sha256": split.split_sha256,
                    "is_smoke_subsample": bool(
                        data_config.get("max_train_rows") is not None
                        or data_config.get("max_test_rows") is not None
                    ),
                }
            )

    preprocessing_started = time.perf_counter()
    preprocessing = fit_wholesale_robust_scaler(
        train_frame,
        test_frame,
        feature_names=list(data_config["feature_columns"]),
        quantile_range=tuple(preprocessing_config.get("quantile_range", [25.0, 75.0])),
    )
    preprocessing_runtime = time.perf_counter() - preprocessing_started
    x_train_complete = preprocessing.x_train
    x_test_complete = preprocessing.x_test
    train_scales = feature_scales(x_train_complete)
    joblib.dump(preprocessing.scaler, run_dir / "models" / "robust_scaler.joblib")
    atomic_write_dataframe(
        scaler_parameters_frame(preprocessing),
        run_dir / "tables" / "raw" / "preprocessing_parameters.csv",
    )

    k = int(problem["k"])
    n_total = len(used_frame)
    dimension = x_train_complete.shape[1]
    train_ids = train_frame["customer_id"].to_numpy(dtype=int)
    test_ids = test_frame["customer_id"].to_numpy(dtype=int)

    reference_seed = stable_seed(randomness["global_seed"], "wholesale_references")
    kmeans_reference = fit_kmeans_reference(
        x_train_complete,
        x_test_complete,
        k,
        reference_seed,
    )
    pam_reference = fit_pam_reference(
        x_train_complete,
        x_test_complete,
        k,
        reference_seed,
    )
    l1_started = time.perf_counter()
    l1_reference = solve_complete_l1_pmedian_reference(
        x_train_complete,
        x_test_complete,
        k,
        time_limit=float(solver["reference_time_limit"]),
        mip_gap=float(solver["reference_mip_gap"]),
        output_flag=int(solver["output_flag"]),
        solver_seed=stable_seed(randomness["global_seed"], "wholesale_reference_l1"),
        threads=int(solver["threads"]),
    )
    l1_total_runtime = time.perf_counter() - l1_started

    reference_metadata = {
        "n_total": n_total,
        "d": dimension,
        "k": k,
        "missing_rate_target": 0.0,
        "missing_seed": 0,
        "rho": np.nan,
        "lambda_center": np.nan,
    }
    _append_solver_diagnostics(
        store,
        record_type="reference",
        stage="train_reference_l1_complete",
        diagnostics=l1_reference,
        metadata=reference_metadata,
        total_call_runtime=l1_total_runtime,
    )

    for split_name, x_split, labels_km, labels_pam, labels_l1 in [
        (
            "train",
            x_train_complete,
            kmeans_reference.labels_train,
            pam_reference.labels_train,
            l1_reference["labels_train"],
        ),
        (
            "test",
            x_test_complete,
            kmeans_reference.labels_test,
            pam_reference.labels_test,
            l1_reference["labels_test"],
        ),
    ]:
        store.rows["reference_audit"].append(
            {
                **reference_metadata,
                **_reference_audit(
                    split=split_name,
                    x_complete=x_split,
                    labels_kmeans=labels_km,
                    labels_pam=labels_pam,
                    labels_l1=labels_l1,
                    l1_diagnostics=l1_reference,
                ),
            }
        )

    references = [
        (
            "reference_kmeans_complete",
            kmeans_reference.labels_train,
            kmeans_reference.labels_test,
            kmeans_reference.centers,
            None,
        ),
        (
            "reference_pam_complete",
            pam_reference.labels_train,
            pam_reference.labels_test,
            pam_reference.centers,
            pam_reference.center_indices_train,
        ),
        (
            "reference_l1_complete",
            l1_reference["labels_train"],
            l1_reference["labels_test"],
            l1_reference["centers_values"],
            l1_reference["centers_idx"],
        ),
    ]
    for model_name, labels_train, labels_test, centers, local_indices in references:
        _save_labels(
            store,
            model_name=model_name,
            record_type="reference",
            split="train",
            customer_ids=train_ids,
            labels=labels_train,
            metadata=reference_metadata,
        )
        _save_labels(
            store,
            model_name=model_name,
            record_type="reference",
            split="test",
            customer_ids=test_ids,
            labels=labels_test,
            metadata=reference_metadata,
        )
        center_customer_ids = (
            train_ids[local_indices] if local_indices is not None else None
        )
        _save_centers(
            store,
            model_name=model_name,
            record_type="reference",
            centers=centers,
            local_indices=local_indices,
            customer_ids=center_customer_ids,
            metadata=reference_metadata,
        )
        if bool(marketing.get("enabled", True)) and bool(
            marketing.get("include_references", True)
        ):
            _append_marketing_outputs(
                store,
                raw_frame=used_frame,
                train_customer_ids=train_ids,
                test_customer_ids=test_ids,
                labels_train=labels_train,
                labels_test=labels_test,
                model_name=model_name,
                metadata={"record_type": "reference", **reference_metadata},
                create_plot=False,
                plot_dir=run_dir / "figures" / "marketing",
            )

    scenario_id = 0
    experiment_id = 0
    for missing_rate in [float(value) for value in problem["missing_rates"]]:
        for missing_seed in [int(value) for value in problem["missing_seeds"]]:
            scenario_id += 1
            LOGGER.info(
                "Wholesale scenario %s missing=%s seed=%s",
                scenario_id,
                missing_rate,
                missing_seed,
            )
            try:
                missingness_started = time.perf_counter()
                x_train_missing, mask_train = induce_mcar_fixed_rate(
                    x_train_complete,
                    missing_rate,
                    stable_seed(
                        randomness["global_seed"],
                        missing_rate,
                        missing_seed,
                        "wholesale_mask_train",
                    ),
                )
                x_test_missing, mask_test = induce_mcar_fixed_rate(
                    x_test_complete,
                    missing_rate,
                    stable_seed(
                        randomness["global_seed"],
                        missing_rate,
                        missing_seed,
                        "wholesale_mask_test",
                    ),
                )
                missingness_runtime = time.perf_counter() - missingness_started

                for split_name, customer_ids, mask in [
                    ("train", train_ids, mask_train),
                    ("test", test_ids, mask_test),
                ]:
                    for local_index, feature in np.argwhere(mask):
                        store.rows["missing_masks_long"].append(
                            {
                                "scenario_id": scenario_id,
                                "missing_rate_target": missing_rate,
                                "missing_seed": missing_seed,
                                "split": split_name,
                                "local_index": int(local_index),
                                "customer_id": int(customer_ids[local_index]),
                                "feature_index": int(feature),
                                "feature_name": WHOLESALE_FEATURES[feature],
                            }
                        )

                lower, upper = compute_train_bounds(
                    x_train_missing,
                    margin_fraction=float(problem["bounds_margin_fraction"]),
                )
                candidates_started = time.perf_counter()
                candidates = build_candidate_tensors_train_test(
                    x_train_missing,
                    x_test_missing,
                    lower,
                    upper,
                    requested_names=list(imputation_config["candidate_names"]),
                    seed=stable_seed(
                        randomness["global_seed"],
                        missing_rate,
                        "wholesale_candidates_fixed",
                    ),
                    mode_decimals=int(imputation_config["mode_decimals"]),
                    knn_neighbors=int(imputation_config["knn_neighbors"]),
                    pmm_donors=int(imputation_config["pmm_donors"]),
                )
                candidate_generation_runtime = time.perf_counter() - candidates_started
                audit = candidates.audit.copy()
                audit["scenario_id"] = scenario_id
                audit["missing_rate_target"] = missing_rate
                audit["missing_seed"] = missing_seed
                store.rows["candidate_audit"].extend(audit.to_dict("records"))

                for candidate_index, method_name in enumerate(candidates.names):
                    for cluster_algorithm in ["kmeans", "pam"]:
                        baseline = evaluate_baseline(
                            method_name=method_name,
                            cluster_algorithm=cluster_algorithm,
                            x_train_imputed=candidates.train[candidate_index],
                            x_test_imputed=candidates.test[candidate_index],
                            x_test_complete=x_test_complete,
                            mask_test=mask_test,
                            k=k,
                            seed=stable_seed(
                                randomness["global_seed"],
                                missing_rate,
                                missing_seed,
                                method_name,
                                cluster_algorithm,
                                "wholesale_baseline",
                            ),
                            labels_ref_kmeans_test=kmeans_reference.labels_test,
                            labels_ref_pam_test=pam_reference.labels_test,
                            labels_ref_l1_test=l1_reference["labels_test"],
                            y_test_true=None,
                            train_scales=train_scales,
                            imputation_runtime=candidates.runtimes[method_name],
                        )
                        baseline_metadata = {
                            "scenario_id": scenario_id,
                            "n_total": n_total,
                            "d": dimension,
                            "n_train": len(train_frame),
                            "n_test": len(test_frame),
                            "k": k,
                            "missing_rate_target": missing_rate,
                            "missing_seed": missing_seed,
                            "rho": np.nan,
                            "lambda_center": np.nan,
                        }
                        baseline_row = {
                            **baseline_metadata,
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
                        runtime_row = baseline_runtime_record(
                            imputation_runtime=float(baseline["imputation_runtime"]),
                            clustering_runtime=float(baseline["clustering_runtime"]),
                            data_loading_runtime=data_loading_runtime,
                            preprocessing_runtime=preprocessing_runtime,
                            missingness_runtime=missingness_runtime,
                            metadata={
                                **baseline_metadata,
                                "model_name": f"{method_name}_{cluster_algorithm}",
                                "method": method_name,
                                "cluster_algo": cluster_algorithm,
                            },
                        )
                        baseline_row["end_to_end_runtime"] = runtime_row["end_to_end_runtime"]
                        store.rows["baseline_test_results"].append(baseline_row)
                        store.rows["runtime_accounting"].append(runtime_row)

                        model_name = f"{method_name}_{cluster_algorithm}"
                        _save_labels(
                            store,
                            model_name=model_name,
                            record_type="baseline",
                            split="train",
                            customer_ids=train_ids,
                            labels=baseline["labels_train"],
                            metadata=baseline_metadata,
                        )
                        _save_labels(
                            store,
                            model_name=model_name,
                            record_type="baseline",
                            split="test",
                            customer_ids=test_ids,
                            labels=baseline["labels_test"],
                            metadata=baseline_metadata,
                        )
                        local_indices = baseline["center_indices_train"]
                        center_ids = train_ids[local_indices] if local_indices is not None else None
                        _save_centers(
                            store,
                            model_name=model_name,
                            record_type="baseline",
                            centers=baseline["centers_values"],
                            local_indices=local_indices,
                            customer_ids=center_ids,
                            metadata=baseline_metadata,
                        )
                        if bool(marketing.get("enabled", True)) and bool(
                            marketing.get("include_baselines", True)
                        ):
                            _append_marketing_outputs(
                                store,
                                raw_frame=used_frame,
                                train_customer_ids=train_ids,
                                test_customer_ids=test_ids,
                                labels_train=baseline["labels_train"],
                                labels_test=baseline["labels_test"],
                                model_name=model_name,
                                metadata={"record_type": "baseline", **baseline_metadata},
                                create_plot=False,
                                plot_dir=run_dir / "figures" / "marketing",
                            )

                for rho in [float(value) for value in hyperparameters["rho_values"]]:
                    for lambda_center in [
                        float(value) for value in hyperparameters["lambda_values"]
                    ]:
                        experiment_id += 1
                        train_started = time.perf_counter()
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
                                randomness["global_seed"],
                                missing_rate,
                                missing_seed,
                                rho,
                                lambda_center,
                                "wholesale_train_solver",
                            ),
                            threads=int(solver["threads"]),
                            x_train_complete_reference=x_train_complete,
                            original_train_indices=train_ids,
                        )
                        train_total_runtime = time.perf_counter() - train_started

                        test_started = time.perf_counter()
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
                                randomness["global_seed"],
                                missing_rate,
                                missing_seed,
                                rho,
                                lambda_center,
                                "wholesale_test_solver",
                            ),
                            threads=int(solver["threads"]),
                        )
                        test_total_runtime = time.perf_counter() - test_started

                        metrics = evaluate_partition(
                            x_complete=x_test_complete,
                            x_imputed=test_result["X_imputed_test"],
                            labels=test_result["labels_test"],
                            missing_mask=mask_test,
                            labels_ref_kmeans=kmeans_reference.labels_test,
                            labels_ref_pam=pam_reference.labels_test,
                            labels_ref_l1=l1_reference["labels_test"],
                            y_true=None,
                            centers=train_result["centers_values"],
                            train_scales=train_scales,
                        )
                        center_info = train_result["center_missing_info"].copy()
                        center_info["experiment_id"] = experiment_id
                        center_info["scenario_id"] = scenario_id
                        center_info["missing_rate_target"] = missing_rate
                        center_info["missing_seed"] = missing_seed
                        center_info["rho"] = rho
                        center_info["lambda_center"] = lambda_center
                        store.rows["center_diagnostics"].extend(
                            center_info.to_dict("records")
                        )

                        for feature_index, feature_name in enumerate(WHOLESALE_FEATURES):
                            store.rows["method_selection"].append(
                                {
                                    "experiment_id": experiment_id,
                                    "scenario_id": scenario_id,
                                    "missing_rate_target": missing_rate,
                                    "missing_seed": missing_seed,
                                    "rho": rho,
                                    "lambda_center": lambda_center,
                                    "feature_index": feature_index,
                                    "variable": feature_name,
                                    "chosen_method": train_result[
                                        "chosen_method_names"
                                    ][feature_index],
                                }
                            )

                        n_incomplete = int(
                            center_info["was_originally_incomplete"].sum()
                        )
                        metadata = {
                            "experiment_id": experiment_id,
                            "scenario_id": scenario_id,
                            "n_total": n_total,
                            "d": dimension,
                            "n_train": len(train_frame),
                            "n_test": len(test_frame),
                            "k": k,
                            "missing_rate_target": missing_rate,
                            "missing_seed": missing_seed,
                            "rho": rho,
                            "lambda_center": lambda_center,
                        }
                        runtime_row = proposed_runtime_record(
                            candidate_generation_runtime=candidate_generation_runtime,
                            train_total_runtime=train_total_runtime,
                            test_total_runtime=test_total_runtime,
                            data_loading_runtime=data_loading_runtime,
                            preprocessing_runtime=preprocessing_runtime,
                            missingness_runtime=missingness_runtime,
                            metadata={
                                **metadata,
                                "model_name": "joint_l1_pmedian",
                                "train_solver_wall_runtime": train_result["wall_runtime"],
                                "test_solver_wall_runtime": test_result["wall_runtime"],
                            },
                        )
                        store.rows["runtime_accounting"].append(runtime_row)

                        proposed_row = {
                            **metadata,
                            "missing_rate_train_real": float(mask_train.mean()),
                            "missing_rate_test_real": float(mask_test.mean()),
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
                            "train_total_runtime": train_total_runtime,
                            "test_status": test_result["status"],
                            "test_status_name": test_result["status_name"],
                            "test_is_certified": test_result[
                                "is_certified_within_tolerance"
                            ],
                            "test_gap": test_result["gap"],
                            "test_absolute_gap": test_result["absolute_gap"],
                            "test_runtime": test_result["wall_runtime"],
                            "test_total_runtime": test_total_runtime,
                            "candidate_generation_runtime": candidate_generation_runtime,
                            "pipeline_runtime": runtime_row["pipeline_runtime"],
                            "end_to_end_runtime": runtime_row["end_to_end_runtime"],
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
                        store.rows["proposed_test_results"].append(proposed_row)

                        for stage, result in [
                            ("train", train_result),
                            ("test", test_result),
                        ]:
                            store.rows["objective_components"].append(
                                {
                                    **metadata,
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

                        _append_solver_diagnostics(
                            store,
                            record_type="proposed",
                            stage="train",
                            diagnostics=train_result,
                            metadata=metadata,
                            total_call_runtime=train_total_runtime,
                        )
                        _append_solver_diagnostics(
                            store,
                            record_type="proposed",
                            stage="test",
                            diagnostics=test_result,
                            metadata=metadata,
                            total_call_runtime=test_total_runtime,
                        )

                        _save_labels(
                            store,
                            model_name="joint_l1_pmedian",
                            record_type="proposed",
                            split="train",
                            customer_ids=train_ids,
                            labels=train_result["labels_train"],
                            metadata=metadata,
                        )
                        _save_labels(
                            store,
                            model_name="joint_l1_pmedian",
                            record_type="proposed",
                            split="test",
                            customer_ids=test_ids,
                            labels=test_result["labels_test"],
                            metadata=metadata,
                        )
                        center_ids = train_ids[train_result["centers_idx"]]
                        _save_centers(
                            store,
                            model_name="joint_l1_pmedian",
                            record_type="proposed",
                            centers=train_result["centers_values"],
                            local_indices=train_result["centers_idx"],
                            customer_ids=center_ids,
                            metadata=metadata,
                        )
                        if bool(marketing.get("enabled", True)):
                            _append_marketing_outputs(
                                store,
                                raw_frame=used_frame,
                                train_customer_ids=train_ids,
                                test_customer_ids=test_ids,
                                labels_train=train_result["labels_train"],
                                labels_test=test_result["labels_test"],
                                model_name=(
                                    f"joint_l1_pmedian_s{scenario_id}_"
                                    f"rho{rho}_lambda{lambda_center}"
                                ),
                                metadata={"record_type": "proposed", **metadata},
                                create_plot=bool(config.create_plots),
                                plot_dir=run_dir / "figures" / "marketing",
                            )

                        if config.save_incremental:
                            store.save_raw()

                if config.save_incremental:
                    store.save_raw()
                gc.collect()

            except Exception as exc:
                LOGGER.exception("Wholesale scenario %s failed", scenario_id)
                store.rows["errors"].append(
                    {
                        "scenario_id": scenario_id,
                        "missing_rate_target": missing_rate,
                        "missing_seed": missing_seed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                store.save_raw()
                if not config.continue_on_error:
                    raise

    store.save_raw()
    _save_aggregates(store)
    summary = {
        "duration_seconds": time.perf_counter() - started,
        "data_source": "UCI Wholesale Customers",
        "uci_id": int(data_config.get("uci_id", 292)),
        "data_sha256": wholesale.data_sha256,
        "split_sha256": split.split_sha256,
        "n_source": len(full_frame),
        "n_train_used": len(train_frame),
        "n_test_used": len(test_frame),
        "k": k,
        "proposed_runs": len(store.rows["proposed_test_results"]),
        "baseline_runs": len(store.rows["baseline_test_results"]),
        "errors": len(store.rows["errors"]),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    LOGGER.info("Wholesale run summary: %s", summary)
    return run_dir
