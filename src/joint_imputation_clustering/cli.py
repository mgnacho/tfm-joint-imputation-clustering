from __future__ import annotations

import argparse
import logging
from pathlib import Path

from joint_imputation_clustering.config import load_config
from joint_imputation_clustering.experiment.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final TRAIN-TEST joint imputation and clustering experiment."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    repository_root = Path(__file__).resolve().parents[2]
    run_dir = run_experiment(config, repository_root, log_level=args.log_level)
    logging.getLogger(__name__).info("Experiment completed: %s", run_dir.resolve())


if __name__ == "__main__":
    main()
