import numpy as np

from joint_imputation_clustering.data.missingness import induce_mcar_fixed_rate


def test_exact_missing_rate_and_valid_rows_columns():
    x = np.arange(120, dtype=float).reshape(60, 2)
    x_missing, mask = induce_mcar_fixed_rate(x, rate=0.20, seed=123)
    assert mask.sum() == round(0.20 * x.size)
    assert np.all(mask.sum(axis=1) < x.shape[1])
    assert np.all(mask.sum(axis=0) >= 1)
    assert np.all(mask.sum(axis=0) < x.shape[0])
    assert np.isnan(x_missing[mask]).all()
    assert np.array_equal(x_missing[~mask], x[~mask])
