from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


def align_labels_to_reference(reference_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
    reference_labels = np.asarray(reference_labels, dtype=int)
    labels = np.asarray(labels, dtype=int)
    table = contingency_matrix(reference_labels, labels)
    reference_indices, label_indices = linear_sum_assignment(-table)
    reference_values = np.unique(reference_labels)
    label_values = np.unique(labels)
    mapping = {
        int(label_values[label_index]): int(reference_values[reference_index])
        for reference_index, label_index in zip(reference_indices, label_indices)
    }
    return np.asarray([mapping.get(int(value), int(value)) for value in labels], dtype=int)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_comparison(
    x_test: np.ndarray,
    labels_kmeans: np.ndarray,
    labels_pam: np.ndarray,
    labels_l1: np.ndarray,
    centers_kmeans: np.ndarray,
    centers_pam: np.ndarray,
    centers_l1: np.ndarray,
    k: int,
    output_path: Path,
) -> None:
    dimension = x_test.shape[1]
    labels_for_plot = [
        align_labels_to_reference(labels_l1, labels_kmeans),
        align_labels_to_reference(labels_l1, labels_pam),
        labels_l1,
    ]
    titles = ["K-means completo", "PAM completo", "P-mediana L1 completa"]
    centers = [centers_kmeans, centers_pam, centers_l1]

    if dimension == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        all_points = np.vstack([x_test, *centers])
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        padding = 0.08 * (maximum - minimum + 1e-9)
        for axis, labels, center_values, title in zip(axes, labels_for_plot, centers, titles):
            axis.scatter(x_test[:, 0], x_test[:, 1], c=labels, cmap="tab10", vmin=0, vmax=k - 1)
            axis.scatter(
                center_values[:, 0],
                center_values[:, 1],
                marker="X",
                s=160,
                c="black",
                edgecolor="white",
                label="Centro",
            )
            axis.set_title(title)
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")
            axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
            axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
            axis.grid(True)
            axis.legend()
    elif dimension == 3:
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        all_points = np.vstack([x_test, *centers])
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        padding = 0.08 * (maximum - minimum + 1e-9)
        aspect = maximum - minimum + 1e-9
        for position, (labels, center_values, title) in enumerate(
            zip(labels_for_plot, centers, titles), start=1
        ):
            axis = fig.add_subplot(1, 3, position, projection="3d")
            axis.scatter(
                x_test[:, 0], x_test[:, 1], x_test[:, 2], c=labels, cmap="tab10", vmin=0, vmax=k - 1
            )
            axis.scatter(
                center_values[:, 0],
                center_values[:, 1],
                center_values[:, 2],
                marker="X",
                s=160,
                c="black",
                edgecolor="white",
            )
            axis.set_title(title)
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")
            axis.set_zlabel("x3")
            axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
            axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
            axis.set_zlim(minimum[2] - padding[2], maximum[2] + padding[2])
            axis.set_box_aspect(aspect)
            axis.view_init(elev=20, azim=35)
    else:
        return
    fig.suptitle("Referencias completas sobre TEST", y=1.02)
    _save(fig, output_path)


def _line_by_rho(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    for rho, group in frame.groupby("rho"):
        summary = group.groupby("lambda_center", as_index=False)[value_column].mean().sort_values("lambda_center")
        axis.plot(summary["lambda_center"], summary[value_column], marker="o", label=f"rho={rho:g}")
        for _, row in group.iterrows():
            axis.scatter(row["lambda_center"], row[value_column], alpha=0.25, s=20)
    axis.set_xlabel("lambda normalizada")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    _save(fig, output_path)


def create_summary_plots(
    proposed: pd.DataFrame,
    baselines: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if proposed.empty:
        return
    for (dimension, missing_rate), subset in proposed.groupby(["d", "missing_rate_target"]):
        suffix = f"d{dimension}_miss{int(round(100 * missing_rate))}"
        _line_by_rho(
            subset,
            "ari_ref_l1model",
            "ARI TEST frente a p-mediana L1 completa",
            f"ARI frente a lambda | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"01_ari_vs_lambda_{suffix}.png",
        )
        _line_by_rho(
            subset,
            "rmse",
            "RMSE TEST",
            f"RMSE frente a lambda | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"02_rmse_vs_lambda_{suffix}.png",
        )
        _line_by_rho(
            subset,
            "silhouette_common_manhattan",
            "Silhouette Manhattan sobre X_test completo",
            f"Silhouette común frente a lambda | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"03_silhouette_vs_lambda_{suffix}.png",
        )
        _line_by_rho(
            subset,
            "train_gap",
            "MIP gap relativo TRAIN",
            f"Gap frente a lambda | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"04_gap_vs_lambda_{suffix}.png",
        )
        _line_by_rho(
            subset,
            "n_incomplete_centers",
            "Número de centros originalmente incompletos",
            f"Centros incompletos frente a lambda | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"05_incomplete_centers_{suffix}.png",
        )
        _line_by_rho(
            subset,
            "mean_center_l1_displacement",
            "Desplazamiento L1 medio de centros",
            f"Desplazamiento de centros | d={dimension}, missing={missing_rate:.0%}",
            figures_dir / f"06_center_displacement_{suffix}.png",
        )

        pivot = subset.pivot_table(
            index="lambda_center", columns="rho", values="ari_ref_l1model", aggfunc="mean"
        ).sort_index()
        fig, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(pivot.values, aspect="auto", vmin=-0.05, vmax=1.0)
        fig.colorbar(image, ax=axis, label="ARI TEST frente a L1")
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_xlabel("rho")
        axis.set_ylabel("lambda")
        axis.set_title(f"Heatmap ARI | d={dimension}, missing={missing_rate:.0%}")
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        fig.tight_layout()
        _save(fig, figures_dir / f"07_heatmap_ari_{suffix}.png")

    certification = (
        proposed.groupby(["d", "lambda_center"], as_index=False)["train_is_certified"].mean()
    )
    for dimension, subset in certification.groupby("d"):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(subset["lambda_center"], 100 * subset["train_is_certified"], marker="o")
        axis.set_xlabel("lambda normalizada")
        axis.set_ylabel("TRAIN certificado dentro de tolerancia (%)")
        axis.set_ylim(0, 105)
        axis.set_title(f"Certificación por lambda | d={dimension}")
        axis.grid(True)
        fig.tight_layout()
        _save(fig, figures_dir / f"08_certification_by_lambda_d{dimension}.png")

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(proposed["train_gap"], proposed["ari_ref_l1model"], alpha=0.65)
    axis.set_xlabel("MIP gap relativo TRAIN")
    axis.set_ylabel("ARI TEST frente a L1")
    axis.set_title("Gap TRAIN y recuperación de la partición")
    axis.grid(True)
    fig.tight_layout()
    _save(fig, figures_dir / "09_gap_vs_test_ari.png")

    if not baselines.empty:
        summary = (
            baselines.groupby(["method", "cluster_algo"], as_index=False)["ari_ref_l1model"]
            .mean()
            .sort_values("ari_ref_l1model")
        )
        labels = summary["method"] + " + " + summary["cluster_algo"]
        fig, axis = plt.subplots(figsize=(10, max(6, 0.35 * len(summary))))
        axis.barh(labels, summary["ari_ref_l1model"])
        axis.set_xlabel("ARI TEST medio frente a L1")
        axis.set_title("Baselines secuenciales")
        axis.grid(axis="x")
        fig.tight_layout()
        _save(fig, figures_dir / "10_baseline_mean_ari.png")


def plot_objective_components(proposed: pd.DataFrame, figures_dir: Path) -> None:
    if proposed.empty:
        return
    components = [
        ("train_clustering_normalized", "Clustering normalizado"),
        ("train_imputation_normalized", "Imputación normalizada"),
        ("train_center_penalty_normalized", "Centros normalizados"),
    ]
    for dimension, subset in proposed.groupby("d"):
        fig, axis = plt.subplots(figsize=(9, 6))
        for column, label in components:
            summary = subset.groupby("lambda_center", as_index=False)[column].mean()
            axis.plot(summary["lambda_center"], summary[column], marker="o", label=label)
        axis.set_xlabel("lambda normalizada")
        axis.set_ylabel("Componente medio")
        axis.set_title(f"Componentes crudos normalizados del objetivo | d={dimension}")
        axis.grid(True)
        axis.legend()
        fig.tight_layout()
        _save(fig, figures_dir / f"11_objective_components_d{dimension}.png")
