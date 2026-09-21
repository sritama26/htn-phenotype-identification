#!/usr/bin/env python3


from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, pairwise_distances_argmin
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

CONFIG = {
    "input": "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv",
    "id_col": "PAT_MRN_ID_ENCRYPT",
    "output_dir": "batch_outputs/clustering_median_strdzn/affinity_propagation",

    "drop_procedural": True,
    "n_pcs": 19,                

  
    "subsample_n": 20_000,       
    "preference_quantile": 0.5,  
    "damping": 0.95,              
    "max_iter": 1000,
    "convergence_iter": 25,
    "random_state": 0,

    "run_sweep": True,          
                                 
    "sweep_quantiles": [0.05, 0.1, 0.25, 0.5, 0.75],

    "umap_sample_n": 50_000,     
}

BINARY = ["has_heart_failure","has_ckd","has_atrial_fibrillation","has_cerebrovascular",
"has_pvd","has_valvular","has_cardiomyopathy","has_copd","has_osa","has_tobacco_use",
"has_I10","has_E78","has_E11","has_Z79","has_Z00","has_Z01","has_Z12","has_I25","has_R06",
"has_E66","med_diuretic","med_beta_blocker","med_ace_inhibitor","med_arb","med_ccb",
"med_alpha_blocker","med_central_alpha2_agonist","med_vasodilator","med_statin",
"med_other_lipid_lowering","med_antiplatelet","med_anticoagulant","med_nitrate",
"med_antiarrhythmic","med_antidiabetic_oral","med_sglt2_inhibitor","med_glp1_agonist",
"med_insulin","med_acetaminophen","med_0_9_sodium_chloride","med_sodium_chloride_0_9_flush",
"med_ondansetron_hcl_pf","med_pantoprazole_sodium","med_cholecalciferol_vitamin_d3",
"med_propofol","med_iopamidol","SEX_FEMALE"]
CONTINUOUS = ["creatinine_value","gfr_value","bun_value","sodium_value","potassium_value",
"calcium_value","glucose_value","height_value","weight_value","bmi_value",
"bp_systolic_value","bp_diastolic_value"]
PROCEDURAL = []


def load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    print(f"Reading input: {p.resolve()}")
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    print(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def preprocess(df, cfg):
    if cfg["id_col"] not in df.columns:
        raise KeyError(
            f"ID column '{cfg['id_col']}' is missing from the input file. "
            "Check CONFIG['input']."
        )

    ids = df[cfg["id_col"]].astype(str).to_numpy()

   
    binary = [
        c for c in BINARY
        if not (cfg["drop_procedural"] and c in PROCEDURAL)
    ]
    feats = binary + CONTINUOUS

    missing = [c for c in feats if c not in df.columns]
    if missing:
        print("\nERROR: expected clustering features are missing:")
        for c in missing:
            print(f"  - {c}")
        print("\nFirst 30 columns actually present in the loaded file:")
        print(list(df.columns[:30]))
        raise KeyError(
            f"{len(missing)} expected clustering features are missing. "
            "Verify CONFIG['input'] points to the cluster-ready master file."
        )


    X = (
        df[feats]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    print(f"Using {len(binary)} binary + {len(CONTINUOUS)} continuous features "
          f"= {len(feats)} total features")
    return ids, X, feats


def run_ap_on_subsample(Xsub, cfg, pref_q=None):
    """Fit AP on the subsample using a precomputed COSINE similarity."""
    pref_q = cfg["preference_quantile"] if pref_q is None else pref_q
    S = cosine_similarity(Xsub).astype("float64")               
    tri = S[np.triu_indices_from(S, k=1)]
    preference = float(np.quantile(tri, pref_q))
    ap = AffinityPropagation(affinity="precomputed", preference=preference,
                             damping=cfg["damping"], max_iter=cfg["max_iter"],
                             convergence_iter=cfg["convergence_iter"],
                             random_state=cfg["random_state"])
    labels = ap.fit_predict(S)
    exemplars = ap.cluster_centers_indices_
    converged = exemplars is not None and len(exemplars) > 0
    return labels, exemplars, preference, converged


def fig_sizes(sizes, out):
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(sizes)), 4.5))
    s = sizes.sort_values(ascending=False)
    ax.bar(range(len(s)), s.values, color="steelblue", edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=90, fontsize=7)
    ax.set_ylabel("n patients"); ax.set_xlabel("AP cluster (sorted by size)")
    ax.set_title(f"Affinity Propagation cluster sizes ({len(s)} clusters, "
                 f"{s.sum():,} patients)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_umap(pca_scores, labels, ids, cfg, out):
    n = min(cfg["umap_sample_n"], len(labels))
    rng = np.random.default_rng(cfg["random_state"])
    idx = rng.choice(len(labels), n, replace=False)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                        random_state=cfg["random_state"]).fit_transform(pca_scores[idx])
        xlab, ylab = "UMAP1", "UMAP2"
    except Exception as e:
        print(f"[umap] umap-learn unavailable ({e}); falling back to PC1/PC2.")
        emb = pca_scores[idx][:, :2]; xlab, ylab = "PC1", "PC2"
    lab = labels[idx]
    uniq = pd.Series(lab).value_counts()
    top = set(uniq.index[:20])                       
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for i, c in enumerate(sorted(top, key=lambda c: -uniq[c])):
        m = lab == c
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.4, color=cmap(i % 20),
                   linewidths=0, rasterized=True, label=f"C{c} ({uniq[c]:,})")
    other = ~np.isin(lab, list(top))
    if other.any():
        ax.scatter(emb[other, 0], emb[other, 1], s=2, alpha=0.2, color="lightgray",
                   linewidths=0, rasterized=True, label="other clusters")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Affinity Propagation clusters — {xlab}/{ylab} on {n:,}-patient sample")
    ax.legend(loc="upper right", markerscale=4, fontsize=7, ncol=2,
              title="Cluster (n in sample)")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def fig_profile_heatmap(X, labels, feats, out, max_clusters=25):
    d = X.copy(); d["cluster"] = labels
    means = d.groupby("cluster")[feats].mean()
    overall = X[feats].mean(); sd = X[feats].std().replace(0, np.nan)
    z = ((means - overall) / sd).fillna(0.0)

    order = d["cluster"].value_counts()
    keep = list(order.index[:max_clusters])
    z = z.loc[keep]
    fig, ax = plt.subplots(figsize=(min(22, 0.32*len(feats)+3), 0.4*len(z)+2))
    vmax = np.nanpercentile(np.abs(z.values), 98)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticks(range(len(z))); ax.set_yticklabels([f"C{c} (n={order[c]:,})" for c in z.index],
                                                     fontsize=7)
    ax.set_title("Standardized cluster profiles (red = above cohort mean, blue = below)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z vs cohort")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def main(cfg=CONFIG):
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    df = load(cfg["input"])
    ids, X, feats = preprocess(df, cfg)
    print(f"Loaded {len(X):,} patients x {X.shape[1]} features")

    pca = PCA(n_components=cfg["n_pcs"], random_state=cfg["random_state"])
    scores = pca.fit_transform(X.to_numpy()).astype("float32")
    print(f"PCA: {cfg['n_pcs']} comps, cumulative variance "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

    rng = np.random.default_rng(cfg["random_state"])
    sub_idx = rng.choice(len(scores), min(cfg["subsample_n"], len(scores)), replace=False)
    Xsub = scores[sub_idx]

   
    if cfg["run_sweep"]:
        print("\nPreference sweep (cluster counts on the subsample, cosine similarity):")
        for q in cfg["sweep_quantiles"]:
            lab, ex, pref, ok = run_ap_on_subsample(Xsub, cfg, pref_q=q)
            print(f"  quantile={q:<4} preference={pref:9.4f} -> "
                  f"{'no convergence' if not ok else str(len(ex))+' clusters'}")
        print("\nSet CONFIG['preference_quantile'] and run_sweep=False to finalize.")
        return

    labels_sub, exemplars, preference, converged = run_ap_on_subsample(Xsub, cfg)
    if not converged:
        raise SystemExit("AP did not converge. Increase 'damping' (e.g. 0.97), raise "
                         "'max_iter', or lower 'preference_quantile'.")
    print(f"AP found {len(exemplars)} exemplars (preference={preference:.4f})")

    exemplar_vecs = Xsub[exemplars]
    labels = pairwise_distances_argmin(scores, exemplar_vecs, metric="cosine")
    print(f"Assigned all {len(labels):,} patients to {len(exemplar_vecs)} exemplars")

    # save labels + sizes + profiles
    pd.DataFrame({cfg["id_col"]: ids, "ap_cluster": labels}).to_csv(
        out / "patient_ap_clusters.csv", index=False)
    sizes = pd.Series(labels).value_counts().sort_index()
    sizes_df = pd.DataFrame({"n_patients": sizes,
                             "percent": (100*sizes/len(labels)).round(3)})
    sizes_df.index.name = "cluster"; sizes_df.to_csv(out / "cluster_sizes.csv")
    d = X.copy(); d["cluster"] = labels
    d.groupby("cluster")[feats].mean().to_csv(out / "cluster_profiles.csv")

   
    try:
        s_idx = rng.choice(len(scores), min(20000, len(scores)), replace=False)
        if len(np.unique(labels[s_idx])) > 1:
            sil = silhouette_score(scores[s_idx], labels[s_idx], metric="cosine")
            print(f"Silhouette (cosine, 20k sample): {sil:.3f}")
    except Exception as e:
        print(f"[silhouette] skipped: {e}")

  
    fig_sizes(sizes, out / "fig_ap_cluster_sizes.png")
    fig_umap(scores, labels, ids, cfg, out / "fig_ap_umap.png")
    fig_profile_heatmap(X, labels, feats, out / "fig_ap_profile_heatmap.png")

    print(f"\nDone. {len(sizes)} clusters. Outputs in {out.resolve()}")
    print(sizes_df.sort_values('n_patients', ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()