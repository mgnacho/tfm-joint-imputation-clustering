import numpy as np
import pandas as pd

from joint_imputation_clustering.data.wholesale import (
    WHOLESALE_FEATURES,
    fit_wholesale_robust_scaler,
)


def _frame(values: np.ndarray, start_id: int) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=WHOLESALE_FEATURES)
    frame.insert(0, "Region", 1)
    frame.insert(0, "Channel", 1)
    frame.insert(0, "customer_id", np.arange(start_id, start_id + len(frame)))
    return frame


def test_robust_scaler_is_fitted_only_on_train() -> None:
    train_values = np.tile(np.arange(1, 11, dtype=float)[:, None], (1, 6))
    test_values = np.full((4, 6), 1_000_000.0)
    train = _frame(train_values, 1)
    test = _frame(test_values, 100)

    preprocessing = fit_wholesale_robust_scaler(train, test)

    expected_median = np.median(train_values, axis=0)
    assert np.allclose(preprocessing.scaler.center_, expected_median)
    assert not np.allclose(preprocessing.scaler.center_, np.median(test_values, axis=0))
    assert preprocessing.x_train.shape == (10, 6)
    assert preprocessing.x_test.shape == (4, 6)
    assert preprocessing.fit_scope.startswith("TRAIN")
