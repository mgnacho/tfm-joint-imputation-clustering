from __future__ import annotations

from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from joint_imputation_clustering.models.gurobi_utils import (
    configure_model,
    optimize_with_wall_clock,
    require_solution,
)


def solve_joint_train_model(
    x_train_missing: np.ndarray,
    candidate_tensor_train: np.ndarray,
    candidate_names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    k: int,
    rho: float,
    lambda_center: float,
    *,
    time_limit: float,
    mip_gap: float,
    mip_focus: int,
    output_flag: int,
    solver_seed: int,
    threads: int,
    x_train_complete_reference: np.ndarray,
    original_train_indices: np.ndarray,
    mip_start: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n, n_features = x_train_missing.shape
    n_candidates = candidate_tensor_train.shape[0]
    if candidate_tensor_train.shape[1:] != x_train_missing.shape:
        raise ValueError("candidate_tensor_train has incompatible shape")
    if len(candidate_names) != n_candidates:
        raise ValueError("candidate_names and candidate tensor disagree")
    if x_train_complete_reference.shape != x_train_missing.shape:
        raise ValueError("TRAIN complete reference must match TRAIN missing shape")
    if len(original_train_indices) != n:
        raise ValueError("original_train_indices has incorrect length")

    missing_positions = [tuple(index) for index in np.argwhere(np.isnan(x_train_missing))]
    if not missing_positions:
        raise ValueError("Joint TRAIN model requires at least one missing cell")

    center_missing_count = np.isnan(x_train_missing).sum(axis=1).astype(float)
    center_missing_fraction = center_missing_count / float(n_features)

    model = gp.Model("joint_train_normalized")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    z = model.addVars(n, n, vtype=GRB.BINARY, name="z")
    dvar = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="d")
    t = model.addVars(n_candidates, n_features, vtype=GRB.BINARY, name="t")

    xhat: dict[tuple[int, int], gp.Var] = {}
    u: dict[tuple[int, int], gp.Var] = {}
    for i, feature in missing_positions:
        xhat[i, feature] = model.addVar(
            lb=float(lower[feature]),
            ub=float(upper[feature]),
            vtype=GRB.CONTINUOUS,
            name=f"xhat_{i}_{feature}",
        )
        u[i, feature] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u_{i}_{feature}")

    width = np.asarray(upper - lower, dtype=float)
    w = model.addVars(
        n,
        n,
        n_features,
        lb=0.0,
        ub={(i, j, feature): float(width[feature]) for i in range(n) for j in range(n) for feature in range(n_features)},
        vtype=GRB.CONTINUOUS,
        name="w",
    )
    model.update()

    def x_expression(i: int, feature: int) -> gp.LinExpr | gp.Var | float:
        if (i, feature) in xhat:
            return xhat[i, feature]
        return float(x_train_missing[i, feature])

    clustering_raw_expr = gp.quicksum(dvar[i] for i in range(n))
    imputation_raw_expr = gp.quicksum(u[position] for position in missing_positions)
    center_raw_expr = gp.quicksum(float(center_missing_fraction[j]) * y[j] for j in range(n))

    clustering_normalized_expr = clustering_raw_expr / float(n * n_features)
    imputation_normalized_expr = imputation_raw_expr / float(len(missing_positions))
    center_normalized_expr = center_raw_expr / float(k)

    model.setObjective(
        clustering_normalized_expr
        + float(rho) * imputation_normalized_expr
        + float(lambda_center) * center_normalized_expr,
        GRB.MINIMIZE,
    )

    for i in range(n):
        model.addConstr(gp.quicksum(z[i, j] for j in range(n)) == 1, name=f"assign_{i}")
    for i in range(n):
        for j in range(n):
            model.addConstr(z[i, j] <= y[j], name=f"open_link_{i}_{j}")
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == k, name="num_centers")
    for j in range(n):
        model.addConstr(z[j, j] == y[j], name=f"self_center_{j}")
    for feature in range(n_features):
        model.addConstr(
            gp.quicksum(t[candidate, feature] for candidate in range(n_candidates)) == 1,
            name=f"one_method_{feature}",
        )

    for i in range(n):
        for j in range(n):
            for feature in range(n_features):
                xi = x_expression(i, feature)
                xj = x_expression(j, feature)
                model.addConstr(w[i, j, feature] >= xi - xj)
                model.addConstr(w[i, j, feature] >= -xi + xj)
            model.addGenConstrIndicator(
                z[i, j],
                True,
                dvar[i] >= gp.quicksum(w[i, j, feature] for feature in range(n_features)),
                name=f"distance_if_assigned_{i}_{j}",
            )

    for i, feature in missing_positions:
        reference = gp.quicksum(
            float(candidate_tensor_train[candidate, i, feature]) * t[candidate, feature]
            for candidate in range(n_candidates)
        )
        model.addConstr(u[i, feature] >= xhat[i, feature] - reference)
        model.addConstr(u[i, feature] >= -xhat[i, feature] + reference)

    mip_start_used = mip_start is not None
    mip_start_name: str | None = None
    mip_start_objective = np.nan

    if mip_start is not None:
        x_start = np.asarray(mip_start["x_imputed_train"], dtype=float)
        centers_start = np.asarray(mip_start["center_indices"], dtype=int)
        assigned_center_start = np.asarray(
            mip_start["assigned_center_indices"], dtype=int
        )
        methods_start = np.asarray(
            mip_start["method_indices_by_feature"], dtype=int
        )

        if x_start.shape != x_train_missing.shape:
            raise ValueError("MIP start imputed matrix has incompatible shape")
        if len(centers_start) != k or len(np.unique(centers_start)) != k:
            raise ValueError("MIP start must contain exactly K distinct centers")
        if np.any((centers_start < 0) | (centers_start >= n)):
            raise ValueError("MIP start contains an invalid center index")
        if len(assigned_center_start) != n:
            raise ValueError("MIP start assignments have incorrect length")
        if not set(assigned_center_start.tolist()).issubset(set(centers_start.tolist())):
            raise ValueError("MIP start assigns a row to a non-center")
        if len(methods_start) != n_features:
            raise ValueError("MIP start must provide one method per feature")
        if np.any((methods_start < 0) | (methods_start >= n_candidates)):
            raise ValueError("MIP start contains an invalid method index")

        center_set = set(centers_start.tolist())

        for j in range(n):
            y[j].Start = 1.0 if j in center_set else 0.0

        for i in range(n):
            assigned_center = int(assigned_center_start[i])
            for j in range(n):
                z[i, j].Start = 1.0 if j == assigned_center else 0.0

        for candidate in range(n_candidates):
            for feature in range(n_features):
                t[candidate, feature].Start = (
                    1.0 if candidate == int(methods_start[feature]) else 0.0
                )

        for i, feature in missing_positions:
            xhat[i, feature].Start = float(x_start[i, feature])
            selected_candidate = int(methods_start[feature])
            reference_value = float(
                candidate_tensor_train[selected_candidate, i, feature]
            )
            u[i, feature].Start = abs(
                float(x_start[i, feature]) - reference_value
            )

        for i in range(n):
            for j in range(n):
                for feature in range(n_features):
                    w[i, j, feature].Start = abs(
                        float(x_start[i, feature])
                        - float(x_start[j, feature])
                    )

        for i in range(n):
            assigned_center = int(assigned_center_start[i])
            dvar[i].Start = float(
                np.abs(x_start[i] - x_start[assigned_center]).sum()
            )

        mip_start_name = str(mip_start.get("name", "pam"))
        mip_start_objective = float(mip_start.get("objective", np.nan))
        model.update()

    configure_model(
        model,
        time_limit=time_limit,
        mip_gap=mip_gap,
        mip_focus=mip_focus,
        output_flag=output_flag,
        seed=solver_seed,
        threads=threads,
    )
    wall_runtime, diagnostics = optimize_with_wall_clock(model)
    require_solution(model, "joint TRAIN")

    centers_idx = np.asarray([j for j in range(n) if y[j].X > 0.5], dtype=int)
    center_to_label = {center: label for label, center in enumerate(centers_idx.tolist())}
    labels_train = np.empty(n, dtype=int)
    assigned_center_idx = np.empty(n, dtype=int)
    for i in range(n):
        assigned = max(range(n), key=lambda j: z[i, j].X)
        if z[i, assigned].X < 0.5:
            raise RuntimeError(f"Could not extract TRAIN assignment for row {i}")
        assigned_center_idx[i] = assigned
        labels_train[i] = center_to_label[assigned]

    chosen_methods: dict[int, int] = {}
    chosen_method_names: dict[int, str] = {}
    for feature in range(n_features):
        selected = max(range(n_candidates), key=lambda candidate: t[candidate, feature].X)
        if t[selected, feature].X < 0.5:
            raise RuntimeError(f"Could not extract method for feature {feature}")
        chosen_methods[feature] = selected
        chosen_method_names[feature] = candidate_names[selected]

    x_imputed_train = np.asarray(x_train_missing, dtype=float).copy()
    for position, variable in xhat.items():
        x_imputed_train[position] = variable.X
    centers_values = x_imputed_train[centers_idx]

    center_rows: list[dict[str, Any]] = []
    for order, local_index in enumerate(centers_idx):
        missing_dimensions = np.where(np.isnan(x_train_missing[local_index]))[0]
        true_center = np.asarray(x_train_complete_reference[local_index], dtype=float)
        imputed_center = np.asarray(x_imputed_train[local_index], dtype=float)
        displacement = imputed_center - true_center
        row: dict[str, Any] = {
            "center_order": order,
            "train_local_index": int(local_index),
            "dataset_global_index": int(original_train_indices[local_index]),
            "was_originally_incomplete": bool(len(missing_dimensions) > 0),
            "n_imputed_dimensions": int(len(missing_dimensions)),
            "missing_fraction": float(len(missing_dimensions) / n_features),
            "imputed_dimensions": ",".join(f"x{feature + 1}" for feature in missing_dimensions),
            "l1_displacement": float(np.sum(np.abs(displacement))),
            "l2_displacement": float(np.linalg.norm(displacement)),
            "max_abs_displacement": float(np.max(np.abs(displacement))),
        }
        for feature in range(n_features):
            row[f"true_x{feature + 1}"] = float(true_center[feature])
            row[f"imputed_x{feature + 1}"] = float(imputed_center[feature])
        center_rows.append(row)

    clustering_raw = float(sum(dvar[i].X for i in range(n)))
    imputation_raw = float(sum(u[position].X for position in missing_positions))
    center_raw = float(sum(center_missing_fraction[j] * y[j].X for j in range(n)))
    clustering_normalized = clustering_raw / float(n * n_features)
    imputation_normalized = imputation_raw / float(len(missing_positions))
    center_normalized = center_raw / float(k)
    objective_reconstructed = (
        clustering_normalized
        + float(rho) * imputation_normalized
        + float(lambda_center) * center_normalized
    )

    return {
        **diagnostics,
        "mip_start_used": mip_start_used,
        "mip_start_name": mip_start_name,
        "mip_start_objective": mip_start_objective,
        "centers_idx": centers_idx,
        "centers_values": centers_values,
        "center_missing_info": pd.DataFrame(center_rows),
        "chosen_methods": chosen_methods,
        "chosen_method_names": chosen_method_names,
        "labels_train": labels_train,
        "assigned_center_idx": assigned_center_idx,
        "X_imputed_train": x_imputed_train,
        "objective_clustering_raw": clustering_raw,
        "objective_clustering_normalized": clustering_normalized,
        "objective_imputation_raw": imputation_raw,
        "objective_imputation_normalized": imputation_normalized,
        "objective_imputation_weighted": float(rho) * imputation_normalized,
        "objective_center_penalty_raw": center_raw,
        "objective_center_penalty_normalized": center_normalized,
        "objective_center_penalty_weighted": float(lambda_center) * center_normalized,
        "objective_reconstructed": objective_reconstructed,
        "model": model,
    }
