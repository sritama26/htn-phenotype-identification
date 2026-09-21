#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
try:
    from scipy.cluster.hierarchy import cophenet
except Exception:  
    cophenet = None

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances_argmin

try:
    import hdbscan  
except Exception:
    hdbscan = None

try:
    import umap  
except Exception:
    umap = None



DEFAULT_INPUT = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv"
DEFAULT_OUTPUT_ROOT = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/final_clustering_outputs"
PATIENT_COL = "PAT_MRN_ID_ENCRYPT"


FINAL_CLUSTER_COLUMNS = [
    "PAT_MRN_ID_ENCRYPT", "has_heart_failure", "has_ckd", "has_atrial_fibrillation",
    "has_cerebrovascular",  "has_pvd",  "has_valvular", "has_cardiomyopathy","has_copd","has_osa", "has_tobacco_use",
    "has_I10", "has_E78","has_E11", "has_Z79", "has_Z00", "has_Z01", "has_Z12",
    "has_I25", "has_R06","has_E66",
    "med_diuretic", "med_beta_blocker", "med_ace_inhibitor", "med_arb",  "med_ccb",
    "med_alpha_blocker", "med_central_alpha2_agonist","med_vasodilator",
    "med_statin", "med_other_lipid_lowering", "med_antiplatelet","med_anticoagulant",
    "med_nitrate","med_antiarrhythmic", "med_antidiabetic_oral", "med_sglt2_inhibitor",
    "med_glp1_agonist", "med_insulin", "med_acetaminophen", "med_0_9_sodium_chloride",
    "med_sodium_chloride_0_9_flush", "med_ondansetron_hcl_pf", "med_pantoprazole_sodium",
    "med_cholecalciferol_vitamin_d3", "med_propofol", "med_iopamidol",
    "SEX_FEMALE",
    "creatinine_value", "gfr_value",  "bun_value", "sodium_value", "potassium_value",
    "calcium_value", "glucose_value",  "height_value", "weight_value",
    "bmi_value", "bp_systolic_value", "bp_diastolic_value",
]


def read_any(path: str | Path) -> pd.DataFrame:
    """Read csv/parquet using exact path first, then same basename csv/parquet."""
    path = Path(path)
    candidates = [path]
    if path.suffix.lower() != ".parquet":
        candidates.append(path.with_suffix(".parquet"))
    if path.suffix.lower() != ".csv":
        candidates.append(path.with_suffix(".csv"))

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            print(f"Reading: {p}")
            if p.suffix.lower() == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)

    raise FileNotFoundError(
        "Could not find input file. Tried: " + ", ".join(str(p) for p in candidates)
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_feature_list(outdir: Path, features: list[str]) -> None:
    (outdir / "feature_columns.txt").write_text("\n".join(features) + "\n")


def select_features(df: pd.DataFrame, remove_ckd: bool = False) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Use every input column except PAT_MRN_ID_ENCRYPT as a clustering feature."""
    if PATIENT_COL not in df.columns:
        raise ValueError(f"Missing required patient ID column: {PATIENT_COL}")

    features = [c for c in df.columns if c != PATIENT_COL]
    if remove_ckd and "has_ckd" in features:
        features.remove("has_ckd")
    if not features:
        raise ValueError("No clustering features remain after removing the patient ID column.")

    Xdf = df[features].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    n_nan = int(Xdf.isna().sum().sum())
    if n_nan:
        warnings.warn(f"Found {n_nan:,} NaN/inf values. Median-imputing inside clustering script.")
        Xdf = Xdf.fillna(Xdf.median()).fillna(0)

    return df[PATIENT_COL].copy(), Xdf, features

def as_float32(Xdf: pd.DataFrame) -> np.ndarray:
    return np.asarray(Xdf.to_numpy(dtype=np.float32, copy=True))


def safe_sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def safe_silhouette(X: np.ndarray, labels: np.ndarray, max_n: int, seed: int) -> float:
    labels = np.asarray(labels)
    valid = np.unique(labels)
    if len(valid) < 2:
        return np.nan
    idx = safe_sample_indices(len(labels), max_n=max_n, seed=seed)
    sampled_labels = labels[idx]
    if len(np.unique(sampled_labels)) < 2:
        return np.nan
    try:
        return float(silhouette_score(X[idx], sampled_labels, metric="euclidean"))
    except Exception as exc:
        warnings.warn(f"Silhouette failed: {exc}")
        return np.nan


def safe_ch_db(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = np.nan
    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = np.nan
    return ch, db


def write_cluster_profiles(
    ids: pd.Series,
    Xdf: pd.DataFrame,
    labels: np.ndarray,
    outdir: Path,
    label_col: str,
    top_n: int = 25,
) -> None:
    tmp = Xdf.copy()
    tmp.insert(0, PATIENT_COL, ids.values)
    tmp[label_col] = labels

    n = len(tmp)
    sizes = tmp.groupby(label_col).size().rename("cluster_size")
    pct = (sizes / n * 100).rename("cluster_pct")
    means = tmp.groupby(label_col)[Xdf.columns].mean()
    profiles = pd.concat([sizes, pct, means], axis=1).reset_index()
    profiles.to_csv(outdir / f"{label_col.lower()}_profiles.csv", index=False)

    
    global_mean = Xdf.mean(axis=0)
    rows = []
    for cl, row in means.iterrows():
        delta = row - global_mean
        top = delta.abs().sort_values(ascending=False).head(top_n).index
        for rank, feat in enumerate(top, start=1):
            rows.append(
                {
                    label_col: cl,
                    "rank": rank,
                    "feature": feat,
                    "cluster_mean": row[feat],
                    "overall_mean": global_mean[feat],
                    "difference": delta[feat],
                    "abs_difference": abs(delta[feat]),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / f"{label_col.lower()}_driving_features.csv", index=False)


def run_umap_plot(
    ids: pd.Series,
    X: np.ndarray,
    labels: np.ndarray,
    outdir: Path,
    prefix: str,
    max_n: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    if umap is None:
        msg = (
            "UMAP was not run because umap-learn is not installed.\n"
            "Install it in your environment with: pip install umap-learn\n"
        )
        (outdir / f"{prefix}_UMAP_NOT_RUN.txt").write_text(msg)
        print(msg.strip())
        return

    idx = safe_sample_indices(len(labels), max_n=max_n, seed=seed)
    Xs = X[idx]
    labs = np.asarray(labels)[idx]
    sampled_ids = ids.iloc[idx].to_numpy()

    effective_neighbors = max(2, min(n_neighbors, len(Xs) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        low_memory=True,
    )
    emb = reducer.fit_transform(Xs)

    umap_df = pd.DataFrame(
        {
            PATIENT_COL: sampled_ids,
            "UMAP1": emb[:, 0],
            "UMAP2": emb[:, 1],
            "cluster": labs,
        }
    )
    umap_df.to_csv(outdir / f"{prefix}_umap_coordinates.csv", index=False)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labs, s=4, alpha=0.75)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"{prefix}: UMAP visualization, colored by cluster")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_umap.png", dpi=220)
    plt.close()

def run_hdbscan_experiment(
    df: pd.DataFrame,
    outdir: Path,
    name: str,
    remove_ckd: bool,
    min_cluster_size: int,
    min_samples: Optional[int],
    umap_sample: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    skip_umap: bool,
    metric_sample: int,
    seed: int,
    jobs: int,
) -> None:
    if hdbscan is None:
        raise ImportError(
            "hdbscan is not installed. Install it in your environment with: pip install hdbscan"
        )

    ensure_dir(outdir)
    ids, Xdf, features = select_features(df, remove_ckd=remove_ckd)
    write_feature_list(outdir, features)
    X = as_float32(Xdf)

    print("\n" + "=" * 80)
    print(f"Running HDBSCAN: {name}")
    print(f"Patients: {X.shape[0]:,}; features: {X.shape[1]:,}; remove_ckd={remove_ckd}")
    print(f"Output: {outdir}")
    print("=" * 80)

    t0 = time.time()
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=jobs,
        prediction_data=False,
    )
    labels = model.fit_predict(X)
    elapsed = time.time() - t0

    probs = getattr(model, "probabilities_", np.full(X.shape[0], np.nan))
    outlier_scores = getattr(model, "outlier_scores_", np.full(X.shape[0], np.nan))

    assign = pd.DataFrame(
        {
            PATIENT_COL: ids.values,
            "HDBSCAN_Cluster": labels,
            "HDBSCAN_Probability": probs,
            "HDBSCAN_Outlier_Score": outlier_scores,
        }
    )
    assign.to_csv(outdir / "hdbscan_assignments.csv", index=False)

    write_cluster_profiles(ids, Xdf, labels, outdir, label_col="HDBSCAN_Cluster")

    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    noise_n = int((labels == -1).sum())
    noise_pct = float(noise_n / len(labels) * 100)

    if n_clusters >= 2 and int(non_noise.sum()) >= 2:
        X_non = X[non_noise]
        y_non = labels[non_noise]
        sil = safe_silhouette(X_non, y_non, max_n=metric_sample, seed=seed)
        idx_metric = safe_sample_indices(len(y_non), max_n=metric_sample, seed=seed)
        ch, db = safe_ch_db(X_non[idx_metric], y_non[idx_metric])
    else:
        sil, ch, db = np.nan, np.nan, np.nan

    metrics = pd.DataFrame(
        [
            {
                "experiment": name,
                "n_patients": X.shape[0],
                "n_features": X.shape[1],
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples if min_samples is not None else "None",
                "n_clusters_excluding_noise": n_clusters,
                "noise_n": noise_n,
                "noise_pct": noise_pct,
                "silhouette_non_noise_sample": sil,
                "calinski_harabasz_non_noise_sample": ch,
                "davies_bouldin_non_noise_sample": db,
                "metric_sample_max_n": metric_sample,
                "runtime_seconds": elapsed,
            }
        ]
    )
    metrics.to_csv(outdir / "hdbscan_metrics.csv", index=False)

    if not skip_umap:
        run_umap_plot(
            ids, X, labels, outdir,
            prefix="hdbscan",
            max_n=umap_sample,
            seed=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
        )
    print(f"Finished HDBSCAN {name}: {n_clusters} clusters, noise={noise_pct:.2f}%")


def assign_to_centroids_chunked(
    X: np.ndarray,
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    labels = np.empty(X.shape[0], dtype=centroid_labels.dtype)
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        nearest = pairwise_distances_argmin(X[start:end], centroids, metric="euclidean")
        labels[start:end] = centroid_labels[nearest]
    return labels


def plot_dendrogram(Z: np.ndarray, outdir: Path, truncate_p: int = 30) -> None:
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=truncate_p, leaf_rotation=90, leaf_font_size=9)
    plt.title("Hierarchical clustering dendrogram, Ward linkage, sampled patients")
    plt.xlabel("Sampled patient groups")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(outdir / "hierarchical_dendrogram.png", dpi=220)
    plt.close()


def run_hierarchical_experiment(
    df: pd.DataFrame,
    outdir: Path,
    sample_size: int,
    k_values: list[int],
    selected_k: int,
    umap_sample: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    skip_umap: bool,
    metric_sample: int,
    max_cophenetic_n: int,
    seed: int,
) -> None:
    ensure_dir(outdir)
    ids, Xdf, features = select_features(df, remove_ckd=False)
    write_feature_list(outdir, features)
    X = as_float32(Xdf)

    print("\n" + "=" * 80)
    print("Running Hierarchical clustering: all features")
    print(f"Patients: {X.shape[0]:,}; features: {X.shape[1]:,}; sample_size={sample_size:,}")
    print(f"Output: {outdir}")
    print("=" * 80)

    rng = np.random.default_rng(seed)
    if X.shape[0] <= sample_size:
        sample_idx = np.arange(X.shape[0])
    else:
        sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)

    Xs = X[sample_idx]
    ids_sample = ids.iloc[sample_idx].reset_index(drop=True)

    t0 = time.time()
    # Ward linkage uses Euclidean geometry and is fit on the sampled all-feature matrix.
    Z = linkage(Xs, method="ward", optimal_ordering=False)
    linkage_elapsed = time.time() - t0
    plot_dendrogram(Z, outdir)

    coph_corr = np.nan
    if cophenet is not None and len(Xs) <= max_cophenetic_n:
        try:
            coph_corr, _ = cophenet(Z, pdist(Xs, metric="euclidean"))
            coph_corr = float(coph_corr)
        except Exception as exc:
            warnings.warn(f"Cophenetic correlation failed: {exc}")
            coph_corr = np.nan

    rows = []
    sample_labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        labs = fcluster(Z, t=k, criterion="maxclust").astype(int)
        sample_labels_by_k[k] = labs
        sil = safe_silhouette(Xs, labs, max_n=metric_sample, seed=seed)
        idx_metric = safe_sample_indices(len(labs), max_n=metric_sample, seed=seed)
        ch, db = safe_ch_db(Xs[idx_metric], labs[idx_metric])
        rows.append(
            {
                "k": k,
                "sample_n": len(Xs),
                "n_features": X.shape[1],
                "silhouette_sample": sil,
                "calinski_harabasz_sample": ch,
                "davies_bouldin_sample": db,
                "cophenetic_corr": coph_corr,
                "linkage_runtime_seconds": linkage_elapsed,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "hierarchical_metrics.csv", index=False)

    
    full_labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        sample_labels = sample_labels_by_k[k]
        centroid_labels = np.array(sorted(np.unique(sample_labels)), dtype=int)
        centroids = np.vstack([Xs[sample_labels == lab].mean(axis=0) for lab in centroid_labels])
        full_labels = assign_to_centroids_chunked(X, centroids, centroid_labels)
        full_labels_by_k[k] = full_labels

        k_outdir = ensure_dir(outdir / f"k_{k}")
        assignments = pd.DataFrame({PATIENT_COL: ids.values, "Hier_Cluster": full_labels})
        assignments.to_csv(k_outdir / f"hierarchical_assignments_k{k}.csv", index=False)
        sample_assignments = pd.DataFrame(
            {PATIENT_COL: ids_sample.values, "Hier_Sample_Cluster": sample_labels}
        )
        sample_assignments.to_csv(k_outdir / f"hierarchical_sample_assignments_k{k}.csv", index=False)
        write_cluster_profiles(ids, Xdf, full_labels, k_outdir, label_col="Hier_Cluster")
        if not skip_umap:
            run_umap_plot(
                ids, X, full_labels, k_outdir,
                prefix=f"hierarchical_k{k}",
                max_n=umap_sample,
                seed=seed,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
            )
        print(f"Finished forced hierarchical K={k}")

    
    if metrics["silhouette_sample"].notna().any():
        best_k = int(metrics.loc[metrics["silhouette_sample"].idxmax(), "k"])
    else:
        best_k = int(k_values[0])

    if selected_k not in full_labels_by_k:
        raise RuntimeError(
            f"selected_k={selected_k} was not materialized. "
            "It must be included in the hierarchical K values."
        )

    selected_labels = full_labels_by_k[selected_k]
    selected_dir = outdir / f"k_{selected_k}"

    final_assignments = pd.DataFrame(
        {PATIENT_COL: ids.values, "Hier_Cluster": selected_labels}
    )
    final_assignments.to_csv(
        outdir / "hierarchical_assignments_final.csv", index=False
    )

    
    profile_src = selected_dir / "hier_cluster_profiles.csv"
    driving_src = selected_dir / "hier_cluster_driving_features.csv"
    if profile_src.exists():
        profile_src.replace(outdir / "hierarchical_cluster_profiles_final.csv")
        
        pd.read_csv(outdir / "hierarchical_cluster_profiles_final.csv").to_csv(
            profile_src, index=False
        )
    if driving_src.exists():
        driving_src.replace(outdir / "hierarchical_driving_features_final.csv")
        pd.read_csv(outdir / "hierarchical_driving_features_final.csv").to_csv(
            driving_src, index=False
        )

    pd.DataFrame([{
        "final_selected_k": selected_k,
        "best_k_by_silhouette": best_k,
        "selection_rule_for_final": "user_parameter_selected_k",
    }]).to_csv(outdir / "hierarchical_k_selection_summary.csv", index=False)

    (outdir / "hierarchical_selected_k.txt").write_text(
        f"Final selected K = {selected_k}\n"
        f"Best silhouette among evaluated K values = {best_k}\n"
        f"Evaluated K values = {list(k_values)}\n"
        "Root-level final files always use --selected-k.\n"
        "All evaluated K outputs are retained in separate k_<K> directories.\n"
    )
    print(
        f"Finished all hierarchical runs. Final selected K={selected_k}; "
        f"best silhouette K={best_k}"
    )


# ----------------------------- MAIN -----------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HDBSCAN and hierarchical clustering on final cluster-ready matrix.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to master_clustering_ready.csv or parquet.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=max(os.cpu_count() or 1, 1), help="CPU jobs for HDBSCAN core distances.")

    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=1000)
    parser.add_argument("--hdbscan-min-samples", type=int, default=-1, help="Use -1 for HDBSCAN default None.")

    parser.add_argument("--hier-sample-size", type=int, default=5000, help="Sample size for Ward linkage.")
    parser.add_argument(
        "--selected-k",
        type=int,
        required=True,
        help="Exact K used for root-level final hierarchical assignments and profiles.",
    )
    parser.add_argument(
        "--hier-k-values", nargs="+", type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Exact hierarchical K values to force. Every listed K is assigned and saved.",
    )
    parser.add_argument("--max-cophenetic-n", type=int, default=5000, help="Skip cophenetic corr if sample is larger than this.")

    parser.add_argument("--umap-sample", type=int, default=10000, help="Maximum patients used in each UMAP plot.")
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP neighborhood size.")
    parser.add_argument("--umap-min-dist", type=float, default=0.10, help="UMAP minimum distance.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP generation.")
    parser.add_argument("--metric-sample", type=int, default=10000, help="Maximum patients used for expensive metrics.")
    parser.add_argument(
        "--only",
        choices=["all", "hdbscan", "hierarchical"],
        default="all",
        help="Which experiments to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selected_k < 2:
        raise ValueError("--selected-k must be at least 2.")
    args.hier_k_values = sorted(set(args.hier_k_values + [args.selected_k]))
    min_samples = None if args.hdbscan_min_samples < 0 else args.hdbscan_min_samples

    df = read_any(args.input)
    print(f"Input shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    output_root = ensure_dir(args.output_root)

    if args.only in {"all", "hdbscan"}:
        run_hdbscan_experiment(
            df=df,
            outdir=output_root / "hdbscan_all_features",
            name="hdbscan_all_features",
            remove_ckd=False,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=min_samples,
            umap_sample=args.umap_sample,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            skip_umap=args.skip_umap,
            metric_sample=args.metric_sample,
            seed=args.seed,
            jobs=args.jobs,
        )

    if args.only in {"all", "hierarchical"}:
        run_hierarchical_experiment(
            df=df,
            outdir=output_root / "hierarchical_all_features",
            sample_size=args.hier_sample_size,
            k_values=args.hier_k_values,
            selected_k=args.selected_k,
            umap_sample=args.umap_sample,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            skip_umap=args.skip_umap,
            metric_sample=args.metric_sample,
            max_cophenetic_n=args.max_cophenetic_n,
            seed=args.seed,
        )

    print("\nDone. Output root:", output_root)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        main()
