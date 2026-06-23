import pandas as pd
import pytest

from joint_imputation_clustering.analysis.aggregate import paired_comparisons


def test_paired_comparisons_handles_baseline_hyperparameter_placeholders():
    keys = {
        "scenario_id": 1,
        "n_total": 36,
        "d": 6,
        "missing_rate_target": 0.2,
        "missing_seed": 1,
    }

    proposed = pd.DataFrame(
        [
            {
                **keys,
                "rho": 0.01,
                "lambda_center": 0.03,
                "ari_ref_l1model": 0.50,
                "rmse": 1.00,
                "silhouette_common_manhattan": 0.20,
            }
        ]
    )

    baselines = pd.DataFrame(
        [
            {
                **keys,
                "rho": float("nan"),
                "lambda_center": float("nan"),
                "method": "median",
                "cluster_algo": "pam",
                "ari_ref_l1model": 0.40,
                "rmse": 1.20,
                "silhouette_common_manhattan": 0.10,
            }
        ]
    )

    result = paired_comparisons(proposed, baselines)

    assert result.loc[0, "rho"] == pytest.approx(0.01)
    assert result.loc[0, "lambda_center"] == pytest.approx(0.03)
    assert result.loc[0, "delta_ari_l1"] == pytest.approx(0.10)
    assert result.loc[0, "delta_rmse"] == pytest.approx(-0.20)
