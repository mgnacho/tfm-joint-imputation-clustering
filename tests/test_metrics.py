import numpy as np

from joint_imputation_clustering.metrics.evaluation import (
    feature_scales,
    imputation_metrics,
)


def test_imputation_metrics_only_use_missing_cells():
    true = np.array([[1.0, 2.0], [3.0, 4.0]])
    imputed = np.array([[1.0, 3.0], [10.0, 4.0]])
    mask = np.array([[False, True], [False, False]])
    scales = feature_scales(true)
    metrics = imputation_metrics(true, imputed, mask, scales)
    assert metrics["rmse"] == 1.0
    assert metrics["mae"] == 1.0
    assert metrics["n_missing_eval"] == 1
