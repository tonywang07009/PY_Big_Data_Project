"""
MecDonate - Step 2: K-Means clustering of counties.

Each county is summarised by the time-average of its behavioural /
economic profile. The optimal K is chosen with the Elbow method
(inertia) and validated with the Silhouette score; PCA gives a 2-D view.
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


def county_profiles(cm: pd.DataFrame) -> pd.DataFrame:
    prof = cm.groupby("county_label")[PROFILE_COLS].mean()
    prof["county_name"] = prof.index.map(COUNTY_NAMES)
    prof["county_english_name"] = (
        prof.index.to_series().map(COUNTY_ENGLISH_NAMES).fillna(prof["county_name"]).values
    )
    return prof


def run(cm: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    prof = county_profiles(cm)
    if len(prof) < 3:
        raise ValueError("K-Means clustering needs at least three scoped counties.")

    X = StandardScaler().fit_transform(prof[PROFILE_COLS].values)

    ks = range(2, min(8, len(prof) - 1) + 1)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    best_k = int(list(ks)[int(np.argmax(sils))])

    # --- elbow + silhouette plot ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(ks), inertias, "o-")
    ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    ax[1].plot(list(ks), sils, "o-", color="tab:green")
    ax[1].axvline(best_k, ls="--", color="red", label=f"best K={best_k}")
    ax[1].set(title="Silhouette score", xlabel="K", ylabel="Silhouette")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/clustering_elbow_silhouette.png", dpi=130)
    plt.close(fig)

    # --- fit final model + PCA view ---
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    prof["cluster"] = km.labels_
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    prof["pc1"], prof["pc2"] = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(prof["pc1"], prof["pc2"], c=prof["cluster"],
                    cmap="tab10", s=120)
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
        json.dump({"k_values": list(ks), "inertia": inertias,
                   "silhouette": sils, "best_k": best_k}, f,
                  ensure_ascii=False, indent=2)

    print(f"[clustering] best K = {best_k}, silhouette = {max(sils):.3f}")
    for c in sorted(prof["cluster"].unique()):
        members = prof[prof["cluster"] == c]["county_name"].tolist()
        print(f"  cluster {c}: {members}")
    return prof


if __name__ == "__main__":
    cm = pd.read_csv("model_outputs/county_month_matrix.csv")
    run(cm, "model_outputs")
