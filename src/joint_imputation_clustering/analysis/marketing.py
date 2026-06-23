from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from joint_imputation_clustering.data.wholesale import (
    WHOLESALE_EXTERNAL,
    WHOLESALE_FEATURES,
)


def build_marketing_assignments(
    raw_frame: pd.DataFrame,
    *,
    train_customer_ids: np.ndarray,
    test_customer_ids: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    metadata = metadata or {}
    train_customer_ids = np.asarray(train_customer_ids, dtype=int)
    test_customer_ids = np.asarray(test_customer_ids, dtype=int)
    labels_train = np.asarray(labels_train, dtype=int)
    labels_test = np.asarray(labels_test, dtype=int)

    if len(train_customer_ids) != len(labels_train):
        raise ValueError("TRAIN customer IDs and labels have different lengths")
    if len(test_customer_ids) != len(labels_test):
        raise ValueError("TEST customer IDs and labels have different lengths")

    label_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "customer_id": train_customer_ids,
                    "split": "train",
                    "cluster": labels_train,
                }
            ),
            pd.DataFrame(
                {
                    "customer_id": test_customer_ids,
                    "split": "test",
                    "cluster": labels_test,
                }
            ),
        ],
        ignore_index=True,
    )
    if label_rows["customer_id"].duplicated().any():
        raise ValueError("A customer appears more than once in marketing assignments")

    columns = ["customer_id"] + WHOLESALE_EXTERNAL + WHOLESALE_FEATURES
    assignments = label_rows.merge(
        raw_frame[columns],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    if assignments[WHOLESALE_FEATURES].isna().any().any():
        raise ValueError("Marketing assignments could not be matched to raw customer data")

    assignments["model_name"] = str(model_name)
    assignments["Total_Spend"] = assignments[WHOLESALE_FEATURES].sum(axis=1)
    for key, value in metadata.items():
        assignments[str(key)] = value

    leading = ["model_name", "customer_id", "split", "cluster"]
    trailing = [column for column in assignments if column not in leading]
    return assignments[leading + trailing]


def _bias_corrected_cramers_v(table: pd.DataFrame) -> tuple[float, float, float, int]:
    if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() == 0:
        return np.nan, np.nan, np.nan, 0

    chi2, p_value, dof, _ = chi2_contingency(table.to_numpy(), correction=False)
    n = float(table.to_numpy().sum())
    rows, columns = table.shape
    if n <= 1:
        return float(chi2), float(p_value), np.nan, int(dof)

    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    cramers_v = np.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan
    return float(chi2), float(p_value), float(cramers_v), int(dof)


def summarize_marketing_assignments(
    assignments: pd.DataFrame,
    *,
    metadata_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty:
        return pd.DataFrame(), pd.DataFrame()
    metadata_columns = metadata_columns or []

    required = {
        "model_name",
        "cluster",
        "Channel",
        "Region",
        "Total_Spend",
        *WHOLESALE_FEATURES,
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Missing marketing columns: {missing}")

    model_metadata = {
        column: assignments[column].iloc[0]
        for column in metadata_columns
        if column in assignments
    }
    n_total = len(assignments)
    total_spend_all = float(assignments["Total_Spend"].sum())
    global_channel = assignments["Channel"].value_counts(normalize=True)
    global_region = assignments["Region"].value_counts(normalize=True)

    profile_rows: list[dict[str, Any]] = []
    for cluster, group in assignments.groupby("cluster", sort=True):
        row: dict[str, Any] = {
            "model_name": assignments["model_name"].iloc[0],
            "cluster": int(cluster),
            "n_clients": int(len(group)),
            "client_share": float(len(group) / n_total),
            "total_spend_sum": float(group["Total_Spend"].sum()),
            "spend_share": (
                float(group["Total_Spend"].sum() / total_spend_all)
                if total_spend_all > 0
                else np.nan
            ),
            "total_spend_mean": float(group["Total_Spend"].mean()),
            "total_spend_median": float(group["Total_Spend"].median()),
            **model_metadata,
        }

        for feature in WHOLESALE_FEATURES:
            feature_sum = float(group[feature].sum())
            row[f"{feature}_mean"] = float(group[feature].mean())
            row[f"{feature}_median"] = float(group[feature].median())
            row[f"{feature}_sum"] = feature_sum
            row[f"{feature}_mix_share"] = (
                feature_sum / float(group["Total_Spend"].sum())
                if float(group["Total_Spend"].sum()) > 0
                else np.nan
            )

        for channel in sorted(assignments["Channel"].astype(int).unique()):
            share = float((group["Channel"].astype(int) == channel).mean())
            row[f"channel_{channel}_share"] = share
            row[f"channel_{channel}_lift"] = (
                share / float(global_channel.loc[channel])
                if float(global_channel.loc[channel]) > 0
                else np.nan
            )

        for region in sorted(assignments["Region"].astype(int).unique()):
            share = float((group["Region"].astype(int) == region).mean())
            row[f"region_{region}_share"] = share
            row[f"region_{region}_lift"] = (
                share / float(global_region.loc[region])
                if float(global_region.loc[region]) > 0
                else np.nan
            )

        median_ratios = {
            feature: float(group[feature].median())
            / max(float(assignments[feature].median()), 1e-12)
            for feature in WHOLESALE_FEATURES
        }
        row["dominant_category_relative_median"] = max(median_ratios, key=median_ratios.get)
        row["dominant_category_relative_ratio"] = float(max(median_ratios.values()))
        profile_rows.append(row)

    association_rows: list[dict[str, Any]] = []
    for external in WHOLESALE_EXTERNAL:
        table = pd.crosstab(assignments["cluster"], assignments[external])
        chi2, p_value, cramers_v, dof = _bias_corrected_cramers_v(table)
        association_rows.append(
            {
                "model_name": assignments["model_name"].iloc[0],
                "external_variable": external,
                "chi_square": chi2,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "cramers_v_bias_corrected": cramers_v,
                "n_clients": n_total,
                **model_metadata,
            }
        )

    return pd.DataFrame(profile_rows), pd.DataFrame(association_rows)


def save_marketing_plots(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    output_dir: Path,
    *,
    filename_prefix: str,
) -> None:
    if assignments.empty or profiles.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    global_medians = assignments[WHOLESALE_FEATURES].median().replace(0, np.nan)
    relative = (
        assignments.groupby("cluster")[WHOLESALE_FEATURES].median()
        .divide(global_medians, axis=1)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(WHOLESALE_FEATURES))
    for cluster in relative.index:
        ax.plot(
            positions,
            relative.loc[cluster].to_numpy(dtype=float),
            marker="o",
            label=f"Cluster {cluster}",
        )
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(WHOLESALE_FEATURES, rotation=35, ha="right")
    ax.set_ylabel("Mediana del clúster / mediana global")
    ax.set_title("Perfil relativo de gasto por clúster")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}_relative_median_profile.png", dpi=180)
    plt.close(fig)

    shares = profiles.set_index("cluster")[["client_share", "spend_share"]].sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    shares.plot(kind="bar", ax=ax)
    ax.set_xlabel("Clúster")
    ax.set_ylabel("Proporción")
    ax.set_title("Peso en clientes y gasto total")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}_client_spend_share.png", dpi=180)
    plt.close(fig)

    channel_table = pd.crosstab(
        assignments["cluster"],
        assignments["Channel"],
        normalize="index",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    channel_table.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Clúster")
    ax.set_ylabel("Proporción dentro del clúster")
    ax.set_title("Composición por canal")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename_prefix}_channel_composition.png", dpi=180)
    plt.close(fig)
