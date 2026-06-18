import numpy as np
import pytest


gp = pytest.importorskip("gurobipy")

from joint_imputation_clustering.models.complete_l1 import (  # noqa: E402
    solve_complete_l1_pmedian_reference,
)


def test_complete_l1_reference_small_model():
    x_train = np.array([[0.0, 0.0], [0.2, 0.1], [4.0, 4.0], [4.2, 4.1]])
    x_test = np.array([[0.1, 0.0], [4.1, 4.0]])
    try:
        result = solve_complete_l1_pmedian_reference(
            x_train,
            x_test,
            2,
            time_limit=30,
            mip_gap=0.0,
            output_flag=0,
            solver_seed=1,
            threads=1,
        )
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi license unavailable: {exc}")
    assert len(result["centers_idx"]) == 2
    assert len(result["labels_test"]) == 2
