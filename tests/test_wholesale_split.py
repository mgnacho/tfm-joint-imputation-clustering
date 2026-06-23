import numpy as np
import pandas as pd

from joint_imputation_clustering.data.wholesale import (
    WHOLESALE_FEATURES,
    stratified_wholesale_split,
)


def _balanced_frame() -> pd.DataFrame:
    rows = []
    customer_id = 1
    counts = {
        (1, 1): 50,
        (1, 2): 40,
        (1, 3): 208,
        (2, 1): 27,
        (2, 2): 7,
        (2, 3): 108,
    }
    for (channel, region), count in counts.items():
        for _ in range(count):
            row = {
                "customer_id": customer_id,
                "Channel": channel,
                "Region": region,
            }
            row.update({feature: customer_id for feature in WHOLESALE_FEATURES})
            rows.append(row)
            customer_id += 1
    return pd.DataFrame(rows)


def test_split_is_reproducible_disjoint_and_stratified() -> None:
    frame = _balanced_frame()
    first = stratified_wholesale_split(frame, test_size=0.30, seed=42)
    second = stratified_wholesale_split(frame, test_size=0.30, seed=42)

    assert len(first.train) == 308
    assert len(first.test) == 132
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    assert first.split_sha256 == second.split_sha256
    assert set(first.train["customer_id"]).isdisjoint(first.test["customer_id"])

    full_distribution = frame.groupby(["Channel", "Region"]).size() / len(frame)
    train_distribution = first.train.groupby(["Channel", "Region"]).size() / len(first.train)
    test_distribution = first.test.groupby(["Channel", "Region"]).size() / len(first.test)
    assert (full_distribution - train_distribution).abs().max() < 0.02
    assert (full_distribution - test_distribution).abs().max() < 0.02
