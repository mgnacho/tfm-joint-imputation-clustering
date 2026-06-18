from __future__ import annotations

import time
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB


STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}
for name in ["WORK_LIMIT", "MEM_LIMIT"]:
    if hasattr(GRB, name):
        STATUS_NAMES[getattr(GRB, name)] = name


def status_name(status: int) -> str:
    return STATUS_NAMES.get(int(status), f"STATUS_{status}")


def safe_attr(model: gp.Model, name: str, default: float = np.nan) -> float:
    try:
        return float(getattr(model, name))
    except (AttributeError, gp.GurobiError, ValueError, TypeError):
        return default


def configure_model(
    model: gp.Model,
    *,
    time_limit: float,
    mip_gap: float,
    output_flag: int,
    seed: int,
    threads: int,
    mip_focus: int | None = None,
) -> None:
    model.Params.OutputFlag = int(output_flag)
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Seed = int(seed)
    model.Params.Threads = int(threads)
    if mip_focus is not None:
        model.Params.MIPFocus = int(mip_focus)


def optimize_with_wall_clock(model: gp.Model) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    model.optimize()
    wall_runtime = time.perf_counter() - started
    return wall_runtime, diagnostics(model, wall_runtime)


def diagnostics(model: gp.Model, wall_runtime: float) -> dict[str, Any]:
    has_solution = int(model.SolCount) > 0
    objective = safe_attr(model, "ObjVal") if has_solution else np.nan
    bound = safe_attr(model, "ObjBound")
    relative_gap = safe_attr(model, "MIPGap") if has_solution else np.nan
    absolute_gap = abs(objective - bound) if has_solution and np.isfinite(bound) else np.nan
    return {
        "status": int(model.Status),
        "status_name": status_name(model.Status),
        "is_certified_within_tolerance": bool(model.Status == GRB.OPTIMAL),
        "is_time_limit": bool(model.Status == GRB.TIME_LIMIT),
        "sol_count": int(model.SolCount),
        "objective": objective,
        "obj_bound": bound,
        "gap": relative_gap,
        "absolute_gap": absolute_gap,
        "gurobi_runtime": safe_attr(model, "Runtime"),
        "wall_runtime": float(wall_runtime),
        "node_count": safe_attr(model, "NodeCount"),
        "work": safe_attr(model, "Work"),
        "num_vars": int(model.NumVars),
        "num_binary_vars": int(model.NumBinVars),
        "num_constraints": int(model.NumConstrs),
        "max_coefficient": safe_attr(model, "MaxCoeff"),
        "min_coefficient": safe_attr(model, "MinCoeff"),
        "max_bound": safe_attr(model, "MaxBound"),
        "max_rhs": safe_attr(model, "MaxRHS"),
    }


def require_solution(model: gp.Model, stage: str) -> None:
    if int(model.SolCount) == 0:
        raise RuntimeError(
            f"{stage} found no feasible solution; status={status_name(model.Status)}"
        )
