from types import SimpleNamespace

import numpy as np
import pandas as pd

from joint_imputation_clustering.data.wholesale import (
    WHOLESALE_FEATURES,
    wholesale_frame_from_uci_dataset,
)


def _fake_uci_dataset() -> SimpleNamespace:
    n = 440
    features = pd.DataFrame(
        {
            "Channel": np.where(np.arange(n) % 3 == 0, 2, 1),
            **{
                feature: np.arange(1, n + 1, dtype=int) + offset
                for offset, feature in enumerate(WHOLESALE_FEATURES)
            },
        }
    )
    targets = pd.DataFrame({"Region": (np.arange(n) % 3) + 1})
    return SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets),
        variables=pd.DataFrame(
            {"name": ["Channel", "Region"] + WHOLESALE_FEATURES}
        ),
        metadata={"uci_id": 292, "num_instances": 440},
    )


def test_loader_combines_features_and_target_and_adds_customer_id() -> None:
    loaded = wholesale_frame_from_uci_dataset(_fake_uci_dataset())

    assert loaded.frame.shape == (440, 9)
    assert loaded.frame["customer_id"].tolist()[:3] == [1, 2, 3]
    assert loaded.frame["customer_id"].iloc[-1] == 440
    assert loaded.frame.columns.tolist() == [
        "customer_id",
        "Channel",
        "Region",
        *WHOLESALE_FEATURES,
    ]
    assert loaded.metadata["uci_id"] == 292
    assert len(loaded.data_sha256) == 64
