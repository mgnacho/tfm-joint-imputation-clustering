from pathlib import Path

from joint_imputation_clustering.config import load_config


def test_full_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "toy_full.yaml")
    assert config.section("problem")["train_fraction"] == 0.70
    assert config.section("hyperparameters")["lambda_values"][0] == 0.0
