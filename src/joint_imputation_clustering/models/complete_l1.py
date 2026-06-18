from __future__ import annotations

from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.metrics import pairwise_distances

from joint_imputation_clustering.clustering.references import assign_to_nearest_centers
from joint_imputation_clustering.models.gurobi_utils import (
    configure_model,
    optimize_with_wall_clock,
    require_solution,
)


def solve_complete_l1_pmedian_reference(
    x_train_complete: np.ndarray,
    x_test_complete: np.ndarray,
    k: int,
    *,
    time_limit: float,
    mip_gap: float,
    output_flag: int,
    solver_seed: int,
    threads: int,
) -> dict[str, Any]:
    n, n_features = x_train_complete.shape
    distances = pairwise_distances(x_train_complete, metric="manhattan")

    model = gp.Model("complete_l1_pmedian_reference")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    z = model.addVars(n, n, vtype=GRB.BINARY, name="z")

    model.setObjective(
        gp.quicksum(float(distances[i, j]) * z[i, j] for i in range(n) for j in range(n))
        / float(n * n_features),
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

    configure_model(
        model,
        time_limit=time_limit,
        mip_gap=mip_gap,
        output_flag=output_flag,
        seed=solver_seed,
        threads=threads,
    )
    wall_runtime, diagnostics = optimize_with_wall_clock(model)
    require_solution(model, "complete L1 reference")

    center_indices = np.asarray([j for j in range(n) if y[j].X > 0.5], dtype=int)
    centers = np.asarray(x_train_complete[center_indices], dtype=float)
    center_to_label = {center: label for label, center in enumerate(center_indices.tolist())}
    labels_train = np.empty(n, dtype=int)
    for i in range(n):
        assigned = max(range(n), key=lambda j: z[i, j].X)
        if z[i, assigned].X < 0.5:
            raise RuntimeError(f"Could not extract TRAIN assignment for row {i}")
        labels_train[i] = center_to_label[assigned]
    labels_test = assign_to_nearest_centers(x_test_complete, centers, metric="manhattan")

    clustering_raw = float(sum(distances[i, center_indices[labels_train[i]]] for i in range(n)))
    clustering_normalized = clustering_raw / float(n * n_features)

    return {
        **diagnostics,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "centers_idx": center_indices,
        "centers_values": centers,
        "objective_clustering_raw": clustering_raw,
        "objective_clustering_normalized": clustering_normalized,
        "objective_reconstructed": clustering_normalized,
        "model": model,
    }
