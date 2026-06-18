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

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid config section: {name}")
        return value

    def validate(self) -> None:
        problem = self.section("problem")
        hyper = self.section("hyperparameters")
        solver = self.section("solver")
        randomness = self.section("randomness")
        imputation = self.section("imputation")

        train_fraction = float(problem["train_fraction"])
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("problem.train_fraction must be in (0, 1)")

        if int(problem["k"]) < 2:
            raise ValueError("problem.k must be at least 2")

        d_values = [int(v) for v in problem["d_values"]]
        dataset_seeds = {int(k): int(v) for k, v in randomness["dataset_seeds"].items()}
        missing_seed_dims = sorted(set(d_values) - set(dataset_seeds))
        if missing_seed_dims:
            raise ValueError(f"No fixed dataset seed for dimensions: {missing_seed_dims}")

        for rate in problem["missing_rates"]:
            if not 0.0 < float(rate) < 1.0:
                raise ValueError("All missing rates must be in (0, 1)")

        if not hyper["rho_values"]:
            raise ValueError("At least one rho value is required")
        if not hyper["lambda_values"]:
            raise ValueError("At least one lambda value is required")
        if any(float(v) < 0 for v in hyper["rho_values"]):
            raise ValueError("rho values must be non-negative")
        if any(float(v) < 0 for v in hyper["lambda_values"]):
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


def load_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    config = ExperimentConfig(raw=raw, source_path=source_path)
    config.validate()
    return config
