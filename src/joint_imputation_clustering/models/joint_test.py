from __future__ import annotations

from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from joint_imputation_clustering.models.gurobi_utils import (
    configure_model,
    optimize_with_wall_clock,
    require_solution,
)


def solve_joint_test_model(
    x_test_missing: np.ndarray,
    centers_values: np.ndarray,
    candidate_tensor_test: np.ndarray,
    chosen_methods: dict[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    rho: float,
    *,
    time_limit: float,
    mip_gap: float,
    output_flag: int,
    solver_seed: int,
    threads: int,
) -> dict[str, Any]:
    n, n_features = x_test_missing.shape
    n_centers = centers_values.shape[0]
    if centers_values.ndim != 2 or centers_values.shape[1] != n_features:
        raise ValueError("centers_values has incompatible shape")
    if np.isnan(centers_values).any():
        raise ValueError("centers_values contains missing values")
    if candidate_tensor_test.shape[1:] != x_test_missing.shape:
        raise ValueError("candidate_tensor_test has incompatible shape")
    if set(chosen_methods) != set(range(n_features)):
        raise ValueError("chosen_methods must contain every feature")
    for feature, method_index in chosen_methods.items():
        if not 0 <= int(method_index) < candidate_tensor_test.shape[0]:
            raise ValueError(f"Invalid candidate index for feature {feature}")

    missing_positions = [tuple(index) for index in np.argwhere(np.isnan(x_test_missing))]
    if not missing_positions:
        labels = np.argmin(
            np.sum(np.abs(x_test_missing[:, None, :] - centers_values[None, :, :]), axis=2),
            axis=1,
        )
        clustering_raw = float(
            np.sum(np.abs(x_test_missing - centers_values[labels]))
        )
        clustering_normalized = clustering_raw / float(n * n_features)
        return {
            "status": int(GRB.OPTIMAL),
            "status_name": "NO_MISSING_CLOSED_FORM",
            "is_certified_within_tolerance": True,
            "is_time_limit": False,
            "sol_count": 1,
            "objective": clustering_normalized,
            "obj_bound": clustering_normalized,
            "gap": 0.0,
            "absolute_gap": 0.0,
            "gurobi_runtime": 0.0,
            "wall_runtime": 0.0,
            "node_count": 0.0,
            "work": 0.0,
            "num_vars": 0,
            "num_binary_vars": 0,
            "num_constraints": 0,
            "max_coefficient": np.nan,
            "min_coefficient": np.nan,
            "max_bound": np.nan,
            "max_rhs": np.nan,
            "labels_test": labels.astype(int),
            "assigned_center_order_test": labels.astype(int),
            "X_imputed_test": x_test_missing.copy(),
            "objective_clustering_raw": clustering_raw,
            "objective_clustering_normalized": clustering_normalized,
            "objective_imputation_raw": 0.0,
            "objective_imputation_normalized": 0.0,
            "objective_imputation_weighted": 0.0,
            "objective_reconstructed": clustering_normalized,
            "model": None,
        }

    model = gp.Model("joint_test_normalized")
    z = model.addVars(n, n_centers, vtype=GRB.BINARY, name="z")
    dvar = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

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

    # Observed TEST values may fall outside TRAIN-derived imputation bounds.
    # Therefore the absolute-distance upper bounds must include both the fixed
    # observed TEST values and the fixed TRAIN centers.
    distance_upper = np.zeros(n_features, dtype=float)
    for feature in range(n_features):
        observed = x_test_missing[~np.isnan(x_test_missing[:, feature]), feature]
        possible_min = min(float(lower[feature]), float(np.min(observed))) if observed.size else float(lower[feature])
        possible_max = max(float(upper[feature]), float(np.max(observed))) if observed.size else float(upper[feature])
        center_min = float(np.min(centers_values[:, feature]))
        center_max = float(np.max(centers_values[:, feature]))
        distance_upper[feature] = max(
            abs(possible_min - center_max),
            abs(possible_max - center_min),
        )

    w = model.addVars(
        n,
        n_centers,
        n_features,
        lb=0.0,
        ub={
            (i, center, feature): float(distance_upper[feature])
            for i in range(n)
            for center in range(n_centers)
            for feature in range(n_features)
        },
        vtype=GRB.CONTINUOUS,
        name="w",
    )
    model.update()

    def x_expression(i: int, feature: int) -> gp.LinExpr | gp.Var | float:
        if (i, feature) in xhat:
            return xhat[i, feature]
        return float(x_test_missing[i, feature])

    clustering_raw_expr = gp.quicksum(dvar[i] for i in range(n))
    imputation_raw_expr = gp.quicksum(u[position] for position in missing_positions)
    clustering_normalized_expr = clustering_raw_expr / float(n * n_features)
    imputation_normalized_expr = imputation_raw_expr / float(len(missing_positions))
    model.setObjective(
        clustering_normalized_expr + float(rho) * imputation_normalized_expr,
        GRB.MINIMIZE,
    )

    for i in range(n):
        model.addConstr(
            gp.quicksum(z[i, center] for center in range(n_centers)) == 1,
            name=f"assign_{i}",
        )

    for i in range(n):
        for center in range(n_centers):
            for feature in range(n_features):
                xi = x_expression(i, feature)
                xc = float(centers_values[center, feature])
                model.addConstr(w[i, center, feature] >= xi - xc)
                model.addConstr(w[i, center, feature] >= -xi + xc)
            model.addGenConstrIndicator(
                z[i, center],
                True,
                dvar[i]
                >= gp.quicksum(w[i, center, feature] for feature in range(n_features)),
                name=f"distance_if_assigned_{i}_{center}",
            )

    for i, feature in missing_positions:
        method_index = chosen_methods[feature]
        reference_value = float(candidate_tensor_test[method_index, i, feature])
        model.addConstr(u[i, feature] >= xhat[i, feature] - reference_value)
        model.addConstr(u[i, feature] >= -xhat[i, feature] + reference_value)

    configure_model(
        model,
        time_limit=time_limit,
        mip_gap=mip_gap,
        output_flag=output_flag,
        seed=solver_seed,
        threads=threads,
    )
    wall_runtime, diagnostics = optimize_with_wall_clock(model)
    require_solution(model, "joint TEST")

    labels_test = np.empty(n, dtype=int)
    for i in range(n):
        assigned = max(range(n_centers), key=lambda center: z[i, center].X)
        if z[i, assigned].X < 0.5:
            raise RuntimeError(f"Could not extract TEST assignment for row {i}")
        labels_test[i] = assigned

    x_imputed_test = np.asarray(x_test_missing, dtype=float).copy()
    for position, variable in xhat.items():
        x_imputed_test[position] = variable.X

    clustering_raw = float(sum(dvar[i].X for i in range(n)))
    imputation_raw = float(sum(u[position].X for position in missing_positions))
    clustering_normalized = clustering_raw / float(n * n_features)
    imputation_normalized = imputation_raw / float(len(missing_positions))
    objective_reconstructed = clustering_normalized + float(rho) * imputation_normalized

    return {
        **diagnostics,
        "labels_test": labels_test,
        "assigned_center_order_test": labels_test.copy(),
        "X_imputed_test": x_imputed_test,
        "objective_clustering_raw": clustering_raw,
        "objective_clustering_normalized": clustering_normalized,
        "objective_imputation_raw": imputation_raw,
        "objective_imputation_normalized": imputation_normalized,
        "objective_imputation_weighted": float(rho) * imputation_normalized,
        "objective_reconstructed": objective_reconstructed,
        "model": model,
    }
