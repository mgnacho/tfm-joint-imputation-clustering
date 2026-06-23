from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def create_run_directory(output_root: Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{experiment_name}_{timestamp}"
    run_dir = base
    counter = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{counter:02d}")
        counter += 1
    for child in ["tables/raw", "tables/aggregated", "figures", "logs", "models"]:
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def atomic_write_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series(dtype="object")
        frame = frame[columns + [column for column in frame.columns if column not in columns]]
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(run_dir: Path, config: dict[str, Any], repository_root: Path) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": git_commit(repository_root),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "scikit-learn-extra",
                "matplotlib",
                "PyYAML",
                "ucimlrepo",
                "gurobipy",
            ]
        ),
        "configuration": config,
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


def copy_config(source: Path, run_dir: Path) -> None:
    shutil.copy2(source, run_dir / source.name)
