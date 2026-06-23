from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def experiment_name(self) -> str:
        return str(self.raw["experiment_name"])

    @property
    def output_root(self) -> Path:
        return Path(self.raw.get("output_root", "results"))

    @property
    def continue_on_error(self) -> bool:
        return bool(self.raw.get("continue_on_error", True))

    @property
    def save_incremental(self) -> bool:
        return bool(self.raw.get("save_incremental", True))

    @property
    def create_plots(self) -> bool:
        return bool(self.raw.get("create_plots", True))

    @property
    def data_source(self) -> str:
        data = self.raw.get("data", {})
        if data is None:
            return "toy"
        if not isinstance(data, dict):
            raise ValueError("data must be a mapping when present")
        return str(data.get("source", "toy")).strip().lower()

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid config section: {name}")
        return value

    def optional_section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Invalid config section: {name}")
        return value

    def _validate_common(self) -> None:
        problem = self.section("problem")
        hyper = self.section("hyperparameters")
        solver = self.section("solver")
        imputation = self.section("imputation")

        train_fraction = float(problem["train_fraction"])
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("problem.train_fraction must be in (0, 1)")
        if int(problem["k"]) < 2:
            raise ValueError("problem.k must be at least 2")

        for rate in problem["missing_rates"]:
            if not 0.0 < float(rate) < 1.0:
                raise ValueError("All missing rates must be in (0, 1)")
        if not problem["missing_seeds"]:
            raise ValueError("At least one missing seed is required")

        if not hyper["rho_values"]:
            raise ValueError(
                "At least one rho value is required. Fill the final Wholesale template "
                "after the toy campaign is closed."
            )
        if not hyper["lambda_values"]:
            raise ValueError(
                "At least one lambda value is required. Fill the final Wholesale template "
                "after the toy campaign is closed."
            )
        if any(float(value) < 0 for value in hyper["rho_values"]):
            raise ValueError("rho values must be non-negative")
        if any(float(value) < 0 for value in hyper["lambda_values"]):
            raise ValueError("lambda values must be non-negative")

        if not imputation["candidate_names"]:
            raise ValueError("At least one imputation candidate is required")

        for key in ["reference_time_limit", "train_time_limit", "test_time_limit"]:
            if float(solver[key]) <= 0:
                raise ValueError(f"solver.{key} must be positive")
        for key in ["reference_mip_gap", "mip_gap"]:
            value = float(solver[key])
            if not 0 <= value < 1:
                raise ValueError(f"solver.{key} must be in [0, 1)")

    def _validate_toy(self) -> None:
        problem = self.section("problem")
        randomness = self.section("randomness")
        d_values = [int(value) for value in problem["d_values"]]
        dataset_seeds = {
            int(key): int(value)
            for key, value in randomness["dataset_seeds"].items()
        }
        missing_seed_dims = sorted(set(d_values) - set(dataset_seeds))
        if missing_seed_dims:
            raise ValueError(f"No fixed dataset seed for dimensions: {missing_seed_dims}")

    def _validate_wholesale(self) -> None:
        data = self.section("data")
        problem = self.section("problem")
        randomness = self.section("randomness")
        preprocessing = self.section("preprocessing")

        if int(data.get("uci_id", 292)) != 292:
            raise ValueError("The applied case is fixed to UCI dataset 292")
        if int(data.get("expected_instances", 440)) != 440:
            raise ValueError("Wholesale Customers must contain 440 observations")
        if int(problem["k"]) != 4:
            raise ValueError("The Wholesale case fixes K=4 from the TRAIN-only preprocessing study")
        if int(randomness.get("split_seed", 42)) != 42:
            raise ValueError("The final Wholesale split is fixed with split_seed=42")

        stratify_columns = list(data.get("stratify_columns", []))
        if stratify_columns != ["Channel", "Region"]:
            raise ValueError(
                "The final Wholesale split must be stratified by [Channel, Region]"
            )

        feature_columns = list(data.get("feature_columns", []))
        expected_features = [
            "Fresh",
            "Milk",
            "Grocery",
            "Frozen",
            "Detergents_Paper",
            "Delicassen",
        ]
        if feature_columns != expected_features:
            raise ValueError(
                "Wholesale feature_columns must contain the six spending "
                "variables in the fixed order"
            )

        if str(preprocessing.get("scaler", "")).lower() != "robust":
            raise ValueError("The selected Wholesale preprocessing is RobustScaler")
        if str(preprocessing.get("fit_scope", "")).lower() != "train_complete":
            raise ValueError(
                "The Wholesale scaler must be fitted on complete TRAIN before "
                "artificial missingness"
            )

        for key in ["max_train_rows", "max_test_rows"]:
            value = data.get(key)
            if value is not None and int(value) < int(problem["k"]) + 1:
                raise ValueError(f"data.{key} is too small for K={problem['k']}")

    def validate(self) -> None:
        self._validate_common()
        if self.data_source == "toy":
            self._validate_toy()
        elif self.data_source in {"wholesale", "wholesale_uci"}:
            self._validate_wholesale()
        else:
            raise ValueError(f"Unsupported data.source: {self.data_source}")


def load_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    config = ExperimentConfig(raw=raw, source_path=source_path)
    config.validate()
    return config
