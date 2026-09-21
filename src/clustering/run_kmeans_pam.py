#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ID_COL = "PAT_MRN_ID_ENCRYPT"
SEX_COL = "SEX_FEMALE"

EXPECTED_COLUMNS = [
    'PAT_MRN_ID_ENCRYPT', 'has_heart_failure', 'has_ckd', 'has_atrial_fibrillation',
    'has_cerebrovascular', 'has_pvd', 'has_valvular', 'has_cardiomyopathy', 'has_copd',
    'has_osa', 'has_tobacco_use', 'has_I10', 'has_E78', 'has_E11', 'has_Z79', 'has_Z00',
    'has_Z01', 'has_Z12', 'has_I25', 'has_R06', 'has_E66', 'med_diuretic',
    'med_beta_blocker', 'med_ace_inhibitor', 'med_arb', 'med_ccb', 'med_alpha_blocker',
    'med_central_alpha2_agonist', 'med_vasodilator', 'med_statin',
    'med_other_lipid_lowering', 'med_antiplatelet', 'med_anticoagulant', 'med_nitrate',
    'med_antiarrhythmic', 'med_antidiabetic_oral', 'med_sglt2_inhibitor',
    'med_glp1_agonist', 'med_insulin', 'med_acetaminophen', 'med_0_9_sodium_chloride',
    'med_sodium_chloride_0_9_flush', 'med_ondansetron_hcl_pf', 'med_pantoprazole_sodium',
    'med_cholecalciferol_vitamin_d3', 'med_propofol', 'med_iopamidol', 'SEX_FEMALE',
    'creatinine_value', 'gfr_value', 'bun_value', 'sodium_value', 'potassium_value',
    'calcium_value', 'glucose_value', 'height_value', 'weight_value', 'bmi_value',
    'bp_systolic_value', 'bp_diastolic_value'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-means and PAM/Gower on 3 feature sets.")
    parser.add_argument("--input", required=True, help="Path to final master cluster CSV/parquet.")
    parser.add_argument("--output_dir", default="cluster_outputs", help="Base output directory.")
    parser.add_argument("--id_col", default=ID_COL, help="Patient ID column.")
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="K values to evaluate. Best k is selected automatically using silhouette.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--silhouette_sample",
        type=int,
        default=10000,
        help="Rows sampled for silhouette to avoid an O(n^2) full computation.",
    )
    parser.add_argument(
        "--pam_sample_size",
        type=int,
        default=10000,
        help="Rows sampled for CLARA-like PAM/k-medoids. Use 5000-20000 depending on memory.",
    )
    parser.add_argument(
        "--assign_chunk_size",
        type=int,
        default=50000,
        help="Rows processed at a time when assigning full cohort to medoids.",
    )
    parser.add_argument(
        "--gower_quantile_low",
        type=float,
        default=0.01,
        help="Lower quantile used for continuous-feature Gower range.",
    )
    parser.add_argument(
        "--gower_quantile_high",
        type=float,
        default=0.99,
        help="Upper quantile used for continuous-feature Gower range.",
    )
    parser.add_argument("--skip_kmeans", action="store_true")
    parser.add_argument("--skip_pam", action="store_true")
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def make_feature_sets(df: pd.DataFrame, id_col: str) -> Dict[str, List[str]]:
    has_cols = [c for c in df.columns if c.startswith("has_")]
    med_cols = [c for c in df.columns if c.startswith("med_")]

    all_non_id = [c for c in df.columns if c != id_col]
    sex_cols = [SEX_COL] if SEX_COL in df.columns else []
    lab_vital_cols = [
        c for c in all_non_id
        if c not in has_cols and c not in med_cols and c not in sex_cols
    ]

    feature_sets = {
        "diagnosis_only": has_cols,
        "diagnosis_meds": has_cols + med_cols,
        "all_labs_vitals": has_cols + med_cols + sex_cols + lab_vital_cols,
    }

    for name, cols in feature_sets.items():
        if not cols:
            raise ValueError(f"Feature set {name} has no columns. Check column names.")

    return feature_sets


def split_binary_continuous(columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Name-based fallback used only before values are inspected."""
    binary_cols = [
        c for c in columns
        if c.startswith("has_") or c.startswith("med_") or c == SEX_COL
    ]
    continuous_cols = [c for c in columns if c not in binary_cols]
    return binary_cols, continuous_cols


def is_binary_like(s: pd.Series) -> bool:
    """Return True for 0/1 columns, including one-hot columns created upstream."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return False
    vals = set(pd.unique(x))
    return vals <= {0, 1}


def infer_binary_continuous(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Classify features using names and observed values.

    This keeps has_*, med_*, and SEX_FEMALE as binary and also catches any
    future one-hot columns produced by features_cluster.py, such as RACE_* columns.
    """
    binary_cols = []
    continuous_cols = []
    for c in columns:
        if c.startswith("has_") or c.startswith("med_") or c == SEX_COL or is_binary_like(df[c]):
            binary_cols.append(c)
        else:
            continuous_cols.append(c)
    return binary_cols, continuous_cols


def coerce_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False).astype(np.int8)
    x = s.copy()
    if x.dtype == object:
        x = x.astype(str).str.strip().str.lower().replace({
            "true": "1", "yes": "1", "y": "1", "present": "1",
            "false": "0", "no": "0", "n": "0", "absent": "0",
            "nan": np.nan, "none": np.nan, "": np.nan,
        })
    x = pd.to_numeric(x, errors="coerce").fillna(0)
    return (x != 0).astype(np.int8)


def prepare_matrix(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
   
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    binary_cols, continuous_cols = infer_binary_continuous(df, columns)
    X = pd.DataFrame(index=df.index)

    for c in binary_cols:
        X[c] = coerce_binary(df[c])

    for c in continuous_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        med = x.median(skipna=True)
        if pd.isna(med):
            med = 0.0
        X[c] = x.fillna(med).astype(float)

    return X[list(columns)], binary_cols, continuous_cols


def make_kmeans_matrix(X_prepared: pd.DataFrame, binary_cols: Sequence[str], continuous_cols: Sequence[str]) -> np.ndarray:
    
    parts = []
    if binary_cols:
        parts.append(X_prepared[list(binary_cols)].to_numpy(dtype=np.float32, copy=True))
    if continuous_cols:
        parts.append(X_prepared[list(continuous_cols)].to_numpy(dtype=np.float32, copy=True))
    if not parts:
        raise ValueError("No features available for K-means.")
    return np.hstack(parts)


def safe_sample_indices(n: int, sample_size: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n)
    return rng.choice(n, size=sample_size, replace=False)


def profile_clusters(X_prepared: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    temp = X_prepared.copy()
    temp["cluster"] = labels
    profile = temp.groupby("cluster").mean(numeric_only=True).round(4)
    counts = pd.Series(labels).value_counts().sort_index().rename("n_patients")
    profile.insert(0, "n_patients", counts)
    profile.insert(1, "pct_patients", (counts / len(labels) * 100).round(2))
    return profile.reset_index()


def choose_best_k(metrics_df: pd.DataFrame, silhouette_col: str) -> Tuple[int, str]:
    valid_sil = metrics_df.dropna(subset=[silhouette_col])
    if not valid_sil.empty:
        row = valid_sil.loc[valid_sil[silhouette_col].idxmax()]
        return int(row["k"]), silhouette_col

    if "davies_bouldin_sample" in metrics_df.columns:
        valid_db = metrics_df.dropna(subset=["davies_bouldin_sample"])
        if not valid_db.empty:
            row = valid_db.loc[valid_db["davies_bouldin_sample"].idxmin()]
            return int(row["k"]), "davies_bouldin_sample_lowest"

    if "calinski_harabasz_sample" in metrics_df.columns:
        valid_ch = metrics_df.dropna(subset=["calinski_harabasz_sample"])
        if not valid_ch.empty:
            row = valid_ch.loc[valid_ch["calinski_harabasz_sample"].idxmax()]
            return int(row["k"]), "calinski_harabasz_sample_highest"

    return int(metrics_df["k"].iloc[0]), "fallback_first_k"


def save_auto_selected_outputs(
    out_dir: Path,
    metrics_df: pd.DataFrame,
    best_k: int,
    selected_by: str,
    cluster_col_prefix: str,
    distance_col_prefix: str | None = None,
) -> None:
    selected = metrics_df.loc[metrics_df["k"] == best_k].copy()
    selected.insert(0, "selected_by", selected_by)
    selected.to_csv(out_dir / "auto_selected_k.csv", index=False)

    assignments = pd.read_csv(out_dir / f"assignments_k{best_k}.csv")
    rename_map = {f"{cluster_col_prefix}_k{best_k}": cluster_col_prefix}
    if distance_col_prefix is not None:
        rename_map[f"{distance_col_prefix}_k{best_k}"] = distance_col_prefix
    assignments = assignments.rename(columns=rename_map)
    assignments.to_csv(out_dir / "assignments_final.csv", index=False)

    shutil.copyfile(out_dir / f"cluster_profiles_k{best_k}.csv", out_dir / "cluster_profiles_final.csv")

    medoids_file = out_dir / f"medoids_k{best_k}.csv"
    if medoids_file.exists():
        shutil.copyfile(medoids_file, out_dir / "medoids_final.csv")


def run_kmeans_for_feature_set(
    df: pd.DataFrame,
    id_col: str,
    set_name: str,
    columns: Sequence[str],
    output_base: Path,
    k_values: Sequence[int],
    silhouette_sample: int,
    random_state: int,
) -> None:
    out_dir = output_base / "kmeans" / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_prepared, binary_cols, continuous_cols = prepare_matrix(df, columns)
    X = make_kmeans_matrix(X_prepared, binary_cols, continuous_cols)

    np.save(out_dir / "feature_matrix_shape.npy", np.array(X.shape))
    pd.Series(columns, name="feature").to_csv(out_dir / "features_used.csv", index=False)

    sample_idx = safe_sample_indices(len(df), silhouette_sample, random_state)
    X_sil = X[sample_idx]

    metrics_rows = []

    for k in k_values:
        print(f"[KMEANS] {set_name}: k={k}", flush=True)
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=300)
        labels = model.fit_predict(X)

        labels_sil = labels[sample_idx]
        if len(np.unique(labels_sil)) > 1:
            sil = silhouette_score(X_sil, labels_sil, metric="euclidean")
            db = davies_bouldin_score(X_sil, labels_sil)
            ch = calinski_harabasz_score(X_sil, labels_sil)
        else:
            sil = np.nan
            db = np.nan
            ch = np.nan

        metrics_rows.append({
            "algorithm": "kmeans_no_extra_scaling",
            "feature_set": set_name,
            "k": k,
            "n_patients": len(df),
            "n_features": X.shape[1],
            "n_binary_features": len(binary_cols),
            "n_continuous_features": len(continuous_cols),
            "continuous_preprocessing": "used_as_already_present_in_input_no_scaler",
            "inertia": float(model.inertia_),
            "silhouette_sample_euclidean": float(sil) if not pd.isna(sil) else np.nan,
            "davies_bouldin_sample": float(db) if not pd.isna(db) else np.nan,
            "calinski_harabasz_sample": float(ch) if not pd.isna(ch) else np.nan,
            "silhouette_sample_n": len(sample_idx),
            "random_state": random_state,
        })

        assignments = pd.DataFrame({
            id_col: df[id_col].values,
            f"kmeans_cluster_k{k}": labels,
        })
        assignments.to_csv(out_dir / f"assignments_k{k}.csv", index=False)

        profile = profile_clusters(X_prepared, labels)
        profile.to_csv(out_dir / f"cluster_profiles_k{k}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    best_k, selected_by = choose_best_k(metrics_df, "silhouette_sample_euclidean")
    save_auto_selected_outputs(
        out_dir=out_dir,
        metrics_df=metrics_df,
        best_k=best_k,
        selected_by=selected_by,
        cluster_col_prefix="kmeans_cluster",
    )
    print(f"[KMEANS] {set_name}: auto-selected k={best_k} using {selected_by}", flush=True)


def fit_gower_ranges(
    X: pd.DataFrame,
    continuous_cols: Sequence[str],
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    if not continuous_cols:
        return np.array([]), np.array([]), np.array([])
    cont = X[list(continuous_cols)].to_numpy(dtype=np.float64, copy=False)
    lows = np.nanquantile(cont, q_low, axis=0)
    highs = np.nanquantile(cont, q_high, axis=0)
    ranges = highs - lows
    ranges[~np.isfinite(ranges) | (ranges == 0)] = 1.0
    return lows.astype(np.float32), highs.astype(np.float32), ranges.astype(np.float32)


def gower_distance_block(
    A: pd.DataFrame,
    B: pd.DataFrame,
    binary_cols: Sequence[str],
    continuous_cols: Sequence[str],
    cont_lows: np.ndarray,
    cont_highs: np.ndarray,
    cont_ranges: np.ndarray,
) -> np.ndarray:
   
    n_a = len(A)
    n_b = len(B)
    n_features = len(binary_cols) + len(continuous_cols)
    if n_features == 0:
        raise ValueError("No features available for Gower distance.")

    D = np.zeros((n_a, n_b), dtype=np.float32)

    if binary_cols:
        A_bin = A[list(binary_cols)].to_numpy(dtype=np.int8, copy=False)
        B_bin = B[list(binary_cols)].to_numpy(dtype=np.int8, copy=False)
        for j in range(A_bin.shape[1]):
            D += (A_bin[:, [j]] != B_bin[:, [j]].T).astype(np.float32)

    if continuous_cols:
        A_cont = A[list(continuous_cols)].to_numpy(dtype=np.float32, copy=True)
        B_cont = B[list(continuous_cols)].to_numpy(dtype=np.float32, copy=True)

        A_cont = np.clip(A_cont, cont_lows, cont_highs)
        B_cont = np.clip(B_cont, cont_lows, cont_highs)

        for j in range(A_cont.shape[1]):
            D += np.minimum(
                np.abs(A_cont[:, [j]] - B_cont[:, [j]].T) / cont_ranges[j],
                1.0,
            ).astype(np.float32)

    D /= np.float32(n_features)
    return D


def kmedoids_from_precomputed(
    D: np.ndarray,
    k: int,
    random_state: int,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    
    try:
        from sklearn_extra.cluster import KMedoids  # type: ignore

        method = "sklearn_extra.KMedoids(method='pam')"
        model = KMedoids(
            n_clusters=k,
            metric="precomputed",
            method="pam",
            init="k-medoids++",
            random_state=random_state,
            max_iter=max_iter,
        )
        labels = model.fit_predict(D)
        medoids = np.asarray(model.medoid_indices_, dtype=int)
        inertia = float(model.inertia_)
        return medoids, labels, inertia, method
    except Exception as exc:
        warnings.warn(
            "sklearn-extra KMedoids is unavailable or failed. "
            "Using alternate-update k-medoids fallback instead. "
            f"Original error: {repr(exc)}"
        )

    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.argmin(D[:, medoids], axis=1)

    for _ in range(max_iter):
        old_medoids = medoids.copy()
        labels = np.argmin(D[:, medoids], axis=1)
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                nearest_dist = np.min(D[:, medoids], axis=1)
                medoids[cluster_id] = int(np.argmax(nearest_dist))
                continue
            subD = D[np.ix_(members, members)]
            medoids[cluster_id] = members[int(np.argmin(subD.sum(axis=1)))]
        if np.array_equal(np.sort(old_medoids), np.sort(medoids)):
            break

    labels = np.argmin(D[:, medoids], axis=1)
    inertia = float(np.min(D[:, medoids], axis=1).sum())
    return medoids, labels, inertia, "fallback alternate-update k-medoids"


def assign_full_to_medoids(
    X_prepared: pd.DataFrame,
    medoid_rows: pd.DataFrame,
    binary_cols: Sequence[str],
    continuous_cols: Sequence[str],
    cont_lows: np.ndarray,
    cont_highs: np.ndarray,
    cont_ranges: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X_prepared)
    labels = np.empty(n, dtype=np.int32)
    nearest_distance = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D_chunk = gower_distance_block(
            X_prepared.iloc[start:end], medoid_rows,
            binary_cols, continuous_cols,
            cont_lows, cont_highs, cont_ranges,
        )
        labels[start:end] = np.argmin(D_chunk, axis=1).astype(np.int32)
        nearest_distance[start:end] = np.min(D_chunk, axis=1).astype(np.float32)
        print(f"    assigned rows {start:,}-{end:,}", flush=True)

    return labels, nearest_distance


def run_pam_gower_for_feature_set(
    df: pd.DataFrame,
    id_col: str,
    set_name: str,
    columns: Sequence[str],
    output_base: Path,
    k_values: Sequence[int],
    pam_sample_size: int,
    silhouette_sample: int,
    assign_chunk_size: int,
    random_state: int,
    q_low: float,
    q_high: float,
) -> None:
    out_dir = output_base / "pam_gower" / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_prepared, binary_cols, continuous_cols = prepare_matrix(df, columns)
    pd.Series(columns, name="feature").to_csv(out_dir / "features_used.csv", index=False)

    n = len(df)
    sample_idx = safe_sample_indices(n, pam_sample_size, random_state)
    X_sample = X_prepared.iloc[sample_idx].reset_index(drop=True)

    cont_lows, cont_highs, cont_ranges = fit_gower_ranges(
        X_prepared, continuous_cols, q_low=q_low, q_high=q_high
    )

    print(f"[PAM/GOWER] {set_name}: computing sample distance matrix for n={len(sample_idx):,}", flush=True)
    D_sample = gower_distance_block(
        X_sample, X_sample, binary_cols, continuous_cols,
        cont_lows, cont_highs, cont_ranges,
    )
    np.fill_diagonal(D_sample, 0.0)

    metrics_rows = []

    for k in k_values:
        print(f"[PAM/GOWER] {set_name}: k={k}", flush=True)
        medoid_sample_positions, sample_labels, inertia, method_used = kmedoids_from_precomputed(
            D_sample, k=k, random_state=random_state, max_iter=100
        )

        if len(np.unique(sample_labels)) > 1:
            if len(sample_labels) > silhouette_sample:
                sil_idx = safe_sample_indices(len(sample_labels), silhouette_sample, random_state)
                D_sil = D_sample[np.ix_(sil_idx, sil_idx)]
                labels_sil = sample_labels[sil_idx]
                sil = silhouette_score(D_sil, labels_sil, metric="precomputed")
            else:
                sil = silhouette_score(D_sample, sample_labels, metric="precomputed")
        else:
            sil = np.nan

        medoid_full_positions = sample_idx[medoid_sample_positions]
        medoid_rows = X_prepared.iloc[medoid_full_positions].reset_index(drop=True)
        medoid_ids = df[id_col].iloc[medoid_full_positions].tolist()

        labels_full, nearest_distance = assign_full_to_medoids(
            X_prepared, medoid_rows,
            binary_cols, continuous_cols,
            cont_lows, cont_highs, cont_ranges,
            chunk_size=assign_chunk_size,
        )

        assignments = pd.DataFrame({
            id_col: df[id_col].values,
            f"pam_gower_cluster_k{k}": labels_full,
            f"pam_gower_distance_to_medoid_k{k}": nearest_distance,
        })
        assignments.to_csv(out_dir / f"assignments_k{k}.csv", index=False)

        profile = profile_clusters(X_prepared, labels_full)
        profile.to_csv(out_dir / f"cluster_profiles_k{k}.csv", index=False)

        medoid_table = pd.DataFrame({
            "cluster": np.arange(k, dtype=int),
            "medoid_sample_position": medoid_sample_positions,
            "medoid_full_row_position": medoid_full_positions,
            id_col: medoid_ids,
        })
        medoid_table.to_csv(out_dir / f"medoids_k{k}.csv", index=False)

        metrics_rows.append({
            "algorithm": "pam_gower_sampled_then_chunked_assignment",
            "feature_set": set_name,
            "k": k,
            "n_patients": n,
            "n_features": len(columns),
            "n_binary_features": len(binary_cols),
            "n_continuous_features": len(continuous_cols),
            "continuous_preprocessing": "used_as_already_present_in_input_for_gower_range_calculation",
            "pam_sample_size": len(sample_idx),
            "inertia_sample_gower": inertia,
            "mean_full_distance_to_medoid": float(np.mean(nearest_distance)),
            "silhouette_sample_gower": float(sil) if not pd.isna(sil) else np.nan,
            "method_used": method_used,
            "gower_continuous_range": f"quantile_{q_low}_{q_high}",
            "random_state": random_state,
            "medoid_patient_ids_json": json.dumps([str(x) for x in medoid_ids]),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    best_k, selected_by = choose_best_k(metrics_df, "silhouette_sample_gower")
    save_auto_selected_outputs(
        out_dir=out_dir,
        metrics_df=metrics_df,
        best_k=best_k,
        selected_by=selected_by,
        cluster_col_prefix="pam_gower_cluster",
        distance_col_prefix="pam_gower_distance_to_medoid",
    )
    print(f"[PAM/GOWER] {set_name}: auto-selected k={best_k} using {selected_by}", flush=True)


def write_run_summary(df: pd.DataFrame, feature_sets: Dict[str, List[str]], output_dir: Path, id_col: str, args: argparse.Namespace) -> None:
    n_rows = len(df)
    n_unique = df[id_col].nunique(dropna=True)
    duplicate_count = n_rows - n_unique

    summary_rows = []
    for name, cols in feature_sets.items():
        binary_cols, continuous_cols = infer_binary_continuous(df, cols)
        summary_rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "n_binary_features": len(binary_cols),
            "n_continuous_features": len(continuous_cols),
            "features": ",".join(cols),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "feature_set_summary.csv", index=False)

    with open(output_dir / "run_config.txt", "w", encoding="utf-8") as f:
        f.write("Run configuration\n")
        f.write("=================\n")
        f.write(f"input={args.input}\n")
        f.write(f"n_rows={n_rows}\n")
        f.write(f"n_unique_{id_col}={n_unique}\n")
        f.write(f"duplicate_patient_rows={duplicate_count}\n")
        f.write(f"k_values={args.k_values}\n")
        f.write("selected_k=auto_by_highest_silhouette\n")
        f.write("expected_input=master_clustering_ready.csv from features_cluster.py\n")
        f.write(f"pam_sample_size={args.pam_sample_size}\n")
        f.write(f"silhouette_sample={args.silhouette_sample}\n")
        f.write(f"random_state={args.random_state}\n")
        f.write("\nNote: K-means uses Euclidean distance on binary columns plus continuous columns as already present in the input file.\n")
        f.write("No RobustScaler, no StandardScaler, and no second z-score normalization are applied.\n")
        f.write("PAM/Gower uses sampled k-medoids plus chunked full-cohort assignment.\n")
        f.write("Continuous columns in Gower are converted to distance contributions using a quantile-based range.\n")
        if n_unique == 324_957:
            f.write("Confirmed: total unique patients equals 324,957.\n")
        else:
            f.write("Warning: total unique patients does not equal 324,957. Check input file.\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = read_table(args.input)

    if args.id_col not in df.columns:
        raise ValueError(f"ID column '{args.id_col}' not found in input file.")

    before = len(df)
    df = df.drop_duplicates(subset=[args.id_col]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after:,} duplicate patient rows based on {args.id_col}.", flush=True)

    print(f"Loaded {len(df):,} rows; unique patients = {df[args.id_col].nunique():,}", flush=True)
    if df[args.id_col].nunique() != 324_957:
        print("WARNING: unique patient count is not 324,957. Continuing with provided file.", flush=True)

    feature_sets = make_feature_sets(df, args.id_col)
    write_run_summary(df, feature_sets, output_dir, args.id_col, args)

    for set_name, columns in feature_sets.items():
        print(f"\n=== Feature set: {set_name} ({len(columns)} features) ===", flush=True)
        if not args.skip_kmeans:
            run_kmeans_for_feature_set(
                df=df,
                id_col=args.id_col,
                set_name=set_name,
                columns=columns,
                output_base=output_dir,
                k_values=args.k_values,
                silhouette_sample=args.silhouette_sample,
                random_state=args.random_state,
            )
        if not args.skip_pam:
            run_pam_gower_for_feature_set(
                df=df,
                id_col=args.id_col,
                set_name=set_name,
                columns=columns,
                output_base=output_dir,
                k_values=args.k_values,
                pam_sample_size=args.pam_sample_size,
                silhouette_sample=args.silhouette_sample,
                assign_chunk_size=args.assign_chunk_size,
                random_state=args.random_state,
                q_low=args.gower_quantile_low,
                q_high=args.gower_quantile_high,
            )

    print(f"\nDone. Results saved under: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
