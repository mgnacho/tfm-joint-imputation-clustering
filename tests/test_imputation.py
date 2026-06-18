import numpy as np

from joint_imputation_clustering.data.missingness import compute_train_bounds
from joint_imputation_clustering.imputation.candidates import (
    build_candidate_tensors_train_test,
)


def test_candidates_preserve_observed_values():
    train = np.array(
        [[1.0, np.nan], [2.0, 4.0], [np.nan, 5.0], [4.0, 6.0], [5.0, 7.0]]
    )
    test = np.array([[np.nan, 4.5], [3.0, np.nan]])
    lower, upper = compute_train_bounds(train)
    result = build_candidate_tensors_train_test(
        train,
        test,
        lower,
        upper,
        requested_names=["mean", "median", "random_empirical", "knn"],
        seed=42,
    )
    train_observed = ~np.isnan(train)
    test_observed = ~np.isnan(test)
    for matrix in result.train:
        assert np.array_equal(matrix[train_observed], train[train_observed])
        assert not np.isnan(matrix).any()
    for matrix in result.test:
        assert np.array_equal(matrix[test_observed], test[test_observed])
        assert not np.isnan(matrix).any()
