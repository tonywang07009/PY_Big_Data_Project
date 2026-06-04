"""
MecDonate - Step 2: K-Means clustering of counties.

Each county is summarised by the time-average of its behavioural /
economic profile. The optimal K is chosen with the Elbow method
(inertia) and validated with the Silhouette score; PCA gives a 2-D view.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

COUNTY_NAMES = {
    0: "南投縣", 1: "台中市", 2: "台北市", 3: "台南市", 4: "台東縣",
    5: "嘉義市", 6: "嘉義縣", 7: "基隆市", 8: "宜蘭縣", 9: "屏東縣",
    10: "彰化縣", 11: "新北市", 12: "新竹市", 13: "新竹縣", 14: "桃園市",
    15: "花蓮縣", 16: "苗栗縣", 17: "雲林縣", 18: "高雄市",
}

PROFILE_COLS = [
    "donation_ratio", "housing_burden", "price_income_ratio",
    "log_total_count", "avg_invoice_value", "carrier_usage_ratio",
    "n_active_industries", "industry_hhi",
]


def county_profiles(cm: pd.DataFrame) -> pd.DataFrame:
    prof = cm.groupby("county_label")[PROFILE_COLS].mean()
    prof["county"] = prof.index.map(COUNTY_NAMES)
    return prof


def run(cm: pd.DataFrame, outdir: str = "outputs") -> pd.DataFrame:
    prof = county_profiles(cm)
    X = StandardScaler().fit_transform(prof[PROFILE_COLS].values)

    ks = range(2, 9)
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
        ax.annotate(r["county"], (r["pc1"], r["pc2"]),
                    fontsize=9, xytext=(4, 4), textcoords="offset points",
                    fontfamily="Microsoft JhengHei")
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
        members = prof[prof["cluster"] == c]["county"].tolist()
        print(f"  cluster {c}: {members}")
    return prof


if __name__ == "__main__":
    cm = pd.read_csv("outputs/county_month_matrix.csv")
    run(cm)
