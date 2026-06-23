from pathlib import Path

import pytest

from joint_imputation_clustering.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_wholesale_smoke_config_is_valid() -> None:
    config = load_config(ROOT / "configs" / "wholesale_smoke.yaml")
    assert config.data_source == "wholesale_uci"
    assert config.section("problem")["k"] == 4
    assert config.section("randomness")["split_seed"] == 42


def test_wholesale_final_template_requires_closed_hyperparameters() -> None:
    with pytest.raises(ValueError, match="rho"):
        load_config(ROOT / "configs" / "wholesale_final.template.yaml")
