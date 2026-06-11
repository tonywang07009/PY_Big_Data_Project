"""
Legacy clustering utilities for county profile exploration.

The formal prediction mainline is the Gate 7 county-month `donation_count`
ensemble. Clustering outputs are exploratory context only and are no longer
part of the formal Gate 0-7 execution path.

The original workflow clusters county-level mean profiles and produces
the legacy `cluster_assignments.csv` / `clustering_pca.png` outputs.

This module also provides a second workflow that:
  - builds county representatives from monthly PROFILE_COLS mean+std
  - clusters counties in that representative space
  - centers each county-month point by its county-cluster centroid
  - runs PCA on the centered monthly points
  - auto-groups nearby county distributions into per-figure PCA charts
"""
from __future__ import annotations

import json
from pathlib import Path

from plotting_setup import configure_matplotlib_cache
configure_matplotlib_cache()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_config import COUNTY_ENGLISH_NAMES, COUNTY_NAMES

PROFILE_COLS = [
    "donation_ratio", "housing_burden", "price_income_ratio",
    "log_total_count", "avg_invoice_value", "carrier_usage_ratio",
    "n_active_industries", "industry_hhi",
]

#  transsmison to the munrcal label
def county_profiles(cm: pd.DataFrame) -> pd.DataFrame:
    prof = cm.groupby("county_label")[PROFILE_COLS].mean()
    prof["county_name"] = prof.index.map(COUNTY_NAMES)
    prof["county_english_name"] = (
        prof.index.to_series().map(COUNTY_ENGLISH_NAMES).fillna(prof["county_name"]).values
    )
    return prof


def county_representatives(cm: pd.DataFrame) -> pd.DataFrame:
    grouped = cm.groupby("county_label")[PROFILE_COLS].agg(["mean", "std"]).fillna(0.0)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped["county_name"] = grouped.index.map(COUNTY_NAMES)
    grouped["county_english_name"] = (
        grouped.index.to_series().map(COUNTY_ENGLISH_NAMES).fillna(grouped["county_name"]).values
    )
    return grouped.reset_index()


def _county_rep_cols() -> list[str]:
    cols = []
    for col in PROFILE_COLS:
        cols.extend([f"{col}_mean", f"{col}_std"])
    return cols


def _mean_cols() -> list[str]:
    return [f"{col}_mean" for col in PROFILE_COLS]


def _k_search(X: np.ndarray) -> tuple[list[int], list[float], list[float], int]:
    if len(X) < 3:
        raise ValueError("K-Means clustering needs at least three scoped counties.")

    ks = list(range(2, min(8, len(X) - 1) + 1))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(X, km.labels_)))
    best_k = int(ks[int(np.argmax(sils))])
    return ks, inertias, sils, best_k


def _plot_k_search(ks: list[int], inertias: list[float], sils: list[float],
                   best_k: int, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertias, "o-")
    ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    ax[1].plot(ks, sils, "o-", color="tab:green")
    ax[1].axvline(best_k, ls="--", color="red", label=f"best K={best_k}")
    ax[1].set(title="Silhouette score", xlabel="K", ylabel="Silhouette")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _auto_plot_groups(county_centers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if len(county_centers) < 3:
        out = county_centers.copy()
        out["plot_group"] = 0
        return out, {
            "k_values": [1],
            "inertia": [0.0],
            "silhouette": [],
            "best_k": 1,
        }

    X = county_centers[["pc1_mean", "pc2_mean"]].to_numpy()
    ks, inertias, sils, best_k = _k_search(X)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    out = county_centers.copy()
    out["plot_group"] = km.labels_
    return out, {
        "k_values": ks,
        "inertia": inertias,
        "silhouette": sils,
        "best_k": best_k,
    }


def _plot_grouped_pca(points: pd.DataFrame, output_root: Path) -> list[str]:
    saved = []
    cmap = plt.get_cmap("tab10")
    for plot_group in sorted(points["plot_group"].unique()):
        sub = points[points["plot_group"] == plot_group].copy()
        labels = sorted(sub["county_plot_label"].dropna().unique().tolist())
        county_order = {name: idx for idx, name in enumerate(labels)}

        fig, ax = plt.subplots(figsize=(8, 6))
        for county_label, county_df in sub.groupby("county_plot_label", sort=True):
            color = cmap(county_order[county_label] % 10)
            ax.scatter(
                county_df["pc1"],
                county_df["pc2"],
                s=28,
                alpha=0.75,
                color=color,
                label=county_label,
            )
        ax.set(
            title=f"County-month PCA distribution group {int(plot_group)}",
            xlabel="PC1",
            ylabel="PC2",
        )
        ax.legend(fontsize=8, markerscale=1.0)
        fig.tight_layout()
        filename = f"clustering_pca_group_{int(plot_group)}.png"
        fig.savefig(output_root / filename, dpi=130)
        plt.close(fig)
        saved.append(filename)
    return saved


def run_v2(cm: pd.DataFrame, outdir: str = "outputs") -> dict:

    output_root = Path(outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    county_rep = county_representatives(cm)
    rep_cols = _county_rep_cols()
    mean_cols = _mean_cols()

    rep_scaler = StandardScaler()

    X_rep = rep_scaler.fit_transform(county_rep[rep_cols].to_numpy())
    ks, inertias, sils, best_k = _k_search(X_rep)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_rep)
    county_rep["cluster"] = km.labels_

    # Month-level points only have PROFILE_COLS. We therefore center them
    # in the standardized county-mean feature space, while county std
    # features remain part of the county-level clustering step.


    # TODO: the bugger : for nomalized 
    mean_scaler = StandardScaler()
    county_means_scaled = mean_scaler.fit_transform(county_rep[mean_cols].to_numpy())
    mean_centers = (
        pd.DataFrame(county_means_scaled, columns=mean_cols)
        .assign(cluster=county_rep["cluster"].to_numpy())
        .groupby("cluster", sort=True)[mean_cols]
        .mean()
    )

    county_cluster_map = county_rep[["county_label", "cluster"]]
    points = cm[["county_label", "county_name", "month"] + PROFILE_COLS].copy()
    points["county_english_name"] = (
        points["county_label"].map(COUNTY_ENGLISH_NAMES).fillna(points["county_name"])
    )
    points["county_plot_label"] = points["county_english_name"]
    points = points.merge(county_cluster_map, on="county_label", how="left")
    point_scaled = mean_scaler.transform(points[PROFILE_COLS].to_numpy())
    point_scaled = pd.DataFrame(point_scaled, columns=mean_cols, index=points.index)
    centroid_lookup = mean_centers.to_dict(orient="index")
    centroid_matrix = np.vstack([
        [centroid_lookup[cluster][col] for col in mean_cols]
        for cluster in points["cluster"].to_numpy()
    ])
    centered = point_scaled[mean_cols].to_numpy() - centroid_matrix

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(centered)
    points["pc1"], points["pc2"] = pcs[:, 0], pcs[:, 1]

    county_centers = (
        points.groupby(["county_label", "county_name"], sort=True)[["pc1", "pc2"]]
        .mean()
        .rename(columns={"pc1": "pc1_mean", "pc2": "pc2_mean"})
        .reset_index()
    )
    county_centers, plot_metrics = _auto_plot_groups(county_centers)
    points = points.merge(
        county_centers[["county_label", "plot_group"]],
        on="county_label",
        how="left",
    )
    county_rep = county_rep.merge(
        county_centers[["county_label", "plot_group", "pc1_mean", "pc2_mean"]],
        on="county_label",
        how="left",
    )

    figure_files = _plot_grouped_pca(points, output_root)

    county_rep.to_csv(output_root / "county_cluster_v2.csv", index=False)
    points[[
        "county_label", "county_name", "county_english_name", "month",
        "cluster", "plot_group", "pc1", "pc2",
    ]].to_csv(output_root / "county_month_pca_points_v2.csv", index=False)
    with open(output_root / "clustering_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "county_cluster_search": {
                    "k_values": ks,
                    "inertia": inertias,
                    "silhouette": sils,
                    "best_k": best_k,
                },
                "plot_group_search": plot_metrics,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "figure_files": figure_files,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[clustering_v2] county-cluster best K = {best_k}, silhouette = {max(sils):.3f}")
    for c in sorted(county_rep["cluster"].unique()):
        members = county_rep[county_rep["cluster"] == c]["county_name"].tolist()
        print(f"  v2 cluster {c}: {members}")
    for g in sorted(points["plot_group"].unique()):
        members = sorted(points[points["plot_group"] == g]["county_name"].unique().tolist())
        print(f"  plot group {g}: {members}")

    return {
        "county_clusters": county_rep,
        "points": points,
        "metrics": {
            "county_cluster_search": {
                "k_values": ks,
                "inertia": inertias,
                "silhouette": sils,
                "best_k": best_k,
            },
            "plot_group_search": plot_metrics,
            "figure_files": figure_files,
        },
    }


def run(cm: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:

    Path(outdir).mkdir(parents=True, exist_ok=True)
    prof = county_profiles(cm)

    X = StandardScaler().fit_transform(prof[PROFILE_COLS].values)
    ks, inertias, sils, best_k = _k_search(X)
    _plot_k_search(ks, inertias, sils, best_k, Path(outdir) / "clustering_elbow_silhouette.png")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    prof["cluster"] = km.labels_

    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    prof["pc1"], prof["pc2"] = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(prof["pc1"], prof["pc2"], c=prof["cluster"], cmap="tab10", s=120)
    for _, r in prof.iterrows():
        ax.annotate(r["county_english_name"], (r["pc1"], r["pc2"]),
                    fontsize=9, xytext=(4, 4), textcoords="offset points")
        
    ax.set(title=f"County clusters (K={best_k}) - PCA 2D",
           xlabel="PC1", ylabel="PC2")
    plt.colorbar(sc, label="cluster")
    fig.tight_layout()
    fig.savefig(f"{outdir}/clustering_pca.png", dpi=130)
    plt.close(fig)

    prof.reset_index().to_csv(f"{outdir}/cluster_assignments.csv", index=False)
    
    with open(f"{outdir}/clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"k_values": ks, "inertia": inertias, "silhouette": sils, "best_k": best_k},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[clustering] best K = {best_k}, silhouette = {max(sils):.3f}")

    for c in sorted(prof["cluster"].unique()):
        members = prof[prof["cluster"] == c]["county_name"].tolist()
        print(f"  cluster {c}: {members}")
    return prof


if __name__ == "__main__":
    cm = pd.read_csv("model_outputs/county_month_matrix.csv")
    run(cm, "model_outputs")
    run_v2(cm, "model_outputs")
