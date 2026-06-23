import numpy as np

from joint_imputation_clustering.metrics.evaluation import evaluate_partition


def test_real_case_returns_nan_for_ari_true() -> None:
    x = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate_partition(
        x_complete=x,
        x_imputed=x,
        labels=labels,
        missing_mask=np.zeros_like(x, dtype=bool),
        labels_ref_kmeans=labels,
        labels_ref_pam=labels,
        labels_ref_l1=labels,
        y_true=None,
        centers=np.array([[0.0], [10.0]]),
        train_scales={"range": np.array([10.1]), "std": np.array([5.0])},
    )
    assert np.isnan(metrics["ari_true"])
    assert metrics["ari_ref_l1model"] == 1.0
