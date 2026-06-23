import numpy as np
import pandas as pd

from joint_imputation_clustering.analysis.marketing import (
    build_marketing_assignments,
    summarize_marketing_assignments,
)
from joint_imputation_clustering.data.wholesale import WHOLESALE_FEATURES


def _raw_frame() -> pd.DataFrame:
    rows = []
    for customer_id in range(1, 9):
        cluster = 0 if customer_id <= 4 else 1
        row = {
            "customer_id": customer_id,
            "Channel": 1 if cluster == 0 else 2,
            "Region": 1 if customer_id % 2 else 3,
        }
        for feature_index, feature in enumerate(WHOLESALE_FEATURES):
            row[feature] = (10 if cluster == 0 else 100) + feature_index
        rows.append(row)
    return pd.DataFrame(rows)


def test_marketing_profiles_include_spend_shares_lifts_and_association() -> None:
    raw = _raw_frame()
    assignments = build_marketing_assignments(
        raw,
        train_customer_ids=np.array([1, 2, 5, 6]),
        test_customer_ids=np.array([3, 4, 7, 8]),
        labels_train=np.array([0, 0, 1, 1]),
        labels_test=np.array([0, 0, 1, 1]),
        model_name="example",
        metadata={"scenario_id": 1},
    )
    profiles, association = summarize_marketing_assignments(
        assignments,
        metadata_columns=["scenario_id"],
    )

    assert profiles["n_clients"].sum() == 8
    assert np.isclose(profiles["client_share"].sum(), 1.0)
    assert np.isclose(profiles["spend_share"].sum(), 1.0)
    cluster_zero = profiles.loc[profiles["cluster"] == 0].iloc[0]
    cluster_one = profiles.loc[profiles["cluster"] == 1].iloc[0]
    assert cluster_zero["channel_1_lift"] == 2.0
    assert cluster_one["channel_2_lift"] == 2.0
    assert set(association["external_variable"]) == {"Channel", "Region"}
    channel_row = association.loc[
        association["external_variable"] == "Channel"
    ].iloc[0]
    assert channel_row["cramers_v_bias_corrected"] > 0.9
