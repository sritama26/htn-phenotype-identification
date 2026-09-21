#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ID_COL = "PAT_MRN_ID_ENCRYPT"
EXPECTED_N_PATIENTS = 324_957

BINARY_FEATURES = [
    "has_heart_failure","has_ckd","has_atrial_fibrillation", "has_cerebrovascular","has_pvd","has_valvular",
    "has_cardiomyopathy","has_copd", "has_osa","has_tobacco_use","has_I10","has_E78","has_E11", "has_Z79",
    "has_Z00","has_Z01","has_Z12","has_I25","has_R06", "has_E66",
    "med_diuretic", "med_beta_blocker", "med_ace_inhibitor","med_arb","med_ccb","med_alpha_blocker",
    "med_central_alpha2_agonist","med_vasodilator", "med_statin","med_other_lipid_lowering",
    "med_antiplatelet","med_anticoagulant", "med_nitrate","med_antiarrhythmic","med_antidiabetic_oral",
    "med_sglt2_inhibitor","med_glp1_agonist","med_insulin","med_acetaminophen", "med_0_9_sodium_chloride",
    "med_sodium_chloride_0_9_flush", "med_ondansetron_hcl_pf", "med_pantoprazole_sodium","med_cholecalciferol_vitamin_d3", "med_propofol", "med_iopamidol",
    "SEX_FEMALE",
]

CONTINUOUS_FEATURES = [
    "creatinine_value",
    "gfr_value",
    "bun_value",
    "sodium_value",
    "potassium_value",
    "calcium_value",
    "glucose_value",
    "height_value",
    "weight_value",
    "bmi_value",
    "bp_systolic_value",
    "bp_diastolic_value",
]


PROCEDURAL_FEATURES = [
    "has_Z79",
    "has_Z00",
    "has_Z01",
    "has_Z12",
    "med_acetaminophen",
    "med_0_9_sodium_chloride",
    "med_sodium_chloride_0_9_flush",
    "med_ondansetron_hcl_pf",
    "med_pantoprazole_sodium",
    "med_cholecalciferol_vitamin_d3",
    "med_propofol",
    "med_iopamidol",
]

EXPECTED_COLUMNS = [ID_COL, *BINARY_FEATURES, *CONTINUOUS_FEATURES]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Scanpy Leiden or PhenoGraph-Leiden on the patient master."
    )
    parser.add_argument("--input", type=Path, default=Path("master_cluster.parquet"))
    parser.add_argument(
        "--method",
        choices=["scanpy_leiden", "phenograph"],
        default="scanpy_leiden",
        help="scanpy_leiden is recommended for the full 324,957-patient cohort.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("batch_outputs/clustering_median_strdzn/phenograph"))
    parser.add_argument("--n-pcs", type=int, default=27)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "manhattan", "correlation"],
        default=None,
        help=(
            "Default is cosine for scanpy_leiden and euclidean for phenograph. "
            "Euclidean is the safer primary metric after PCA; compare cosine in a sensitivity run."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--drop-procedural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude screening/administrative and nonspecific inpatient medication features.",
    )
    parser.add_argument(
        "--continuous-already-scaled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use this only when the 12 continuous columns are already scaled.",
    )
    parser.add_argument(
        "--scaler",
        choices=["robust", "standard"],
        default="robust",
        help="Applied only to continuous columns; binary columns remain 0/1.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Clip scaled continuous values to +/- this value; use a negative value to disable.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="PhenoGraph only: clusters smaller than this are relabeled -1.",
    )
    parser.add_argument(
        "--phenograph-nn-method",
        choices=["brute", "kdtree"],
        default="brute",
        help="PhenoGraph nearest-neighbor search implementation.",
    )
    parser.add_argument(
        "--run-umap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create a separate UMAP on a random subsample for visualization.",
    )
    parser.add_argument("--umap-sample-size", type=int, default=50_000)
    parser.add_argument(
        "--save-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the sparse graph separately as graph_connectivities.npz.",
    )
    parser.add_argument(
        "--save-h5ad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the original master columns plus the cluster label as Parquet.",
    )
    parser.add_argument("--top-drivers", type=int, default=15)
    parser.add_argument(
        "--make-driver-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create PNG images of top driving features for each cluster.",
    )
    parser.add_argument(
        "--driver-plot-top-n",
        type=int,
        default=10,
        help="Number of top drivers to show in each cluster plot.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use Parquet or CSV.")


def resolution_string(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def validate_binary(series: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_original = series.notna() & numeric.isna()
    if invalid_original.any():
        examples = series.loc[invalid_original].astype(str).unique()[:5]
        raise ValueError(
            f"Binary column {column!r} contains nonnumeric values: {examples.tolist()}"
        )

    observed = numeric.dropna().unique()
    invalid_values = observed[~np.isin(observed, [0, 1])]
    if len(invalid_values):
        raise ValueError(
            f"Binary column {column!r} must contain only 0/1/NaN; "
            f"found {invalid_values[:10].tolist()}"
        )
    return numeric


def safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    return value if np.isfinite(value) and value > 0 else np.nan


def save_driver_plots(driver_percentages: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One image per cluster
    for cluster in driver_percentages.columns:
        top = driver_percentages[cluster].sort_values(ascending=False).head(top_n)
        fig_h = max(4.5, 0.45 * len(top) + 1.5)
        plt.figure(figsize=(10, fig_h))
        plt.barh(top.index[::-1], top.values[::-1])
        plt.xlabel("Percent contribution to cluster phenotype distinctiveness")
        plt.ylabel("Feature")
        plt.title(f"Cluster {cluster}: top {len(top)} driving factors")
        plt.tight_layout()
        plt.savefig(plots_dir / f"cluster_{cluster}_top_drivers.png", dpi=200, bbox_inches="tight")
        plt.close()

    
    top_features = []
    for cluster in driver_percentages.columns:
        top_features.extend(driver_percentages[cluster].sort_values(ascending=False).head(top_n).index.tolist())
    top_features = list(dict.fromkeys(top_features))
    if top_features:
        mat = driver_percentages.loc[top_features, driver_percentages.columns]
        fig_w = max(8, 1.2 * len(mat.columns) + 3)
        fig_h = max(6, 0.35 * len(mat.index) + 2)
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(mat.values, aspect='auto')
        plt.colorbar(label='Percent contribution')
        plt.xticks(range(len(mat.columns)), [str(c) for c in mat.columns])
        plt.yticks(range(len(mat.index)), mat.index)
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        plt.title(f'Top driving factors across clusters (top {top_n} per cluster)')
        plt.tight_layout()
        plt.savefig(plots_dir / 'all_clusters_top_drivers_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()

def main() -> None:
    args = parse_args()

    if args.n_pcs < 2:
        raise ValueError("--n-pcs must be at least 2")
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if args.resolution <= 0:
        raise ValueError("--resolution must be greater than 0")
    if args.top_drivers < 1:
        raise ValueError("--top-drivers must be at least 1")

  
    metric = args.metric
    if metric is None:
        metric = "cosine" if args.method == "scanpy_leiden" else "euclidean"

    run_name = (
        f"{args.method}_k{args.n_neighbors}_res{resolution_string(args.resolution)}_"
        f"{metric}"
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}", flush=True)
    df = load_table(args.input)

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise KeyError(
            "The final master is missing required columns:\n  - "
            + "\n  - ".join(missing_columns)
        )

    extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_columns:
        print(
            "Ignoring columns outside the declared final-master schema: "
            + ", ".join(extra_columns),
            file=sys.stderr,
            flush=True,
        )

 
    df = df.loc[:, EXPECTED_COLUMNS].copy()

    if len(df) != EXPECTED_N_PATIENTS:
        print(
            f"WARNING: expected {EXPECTED_N_PATIENTS:,} patients but found {len(df):,}.",
            file=sys.stderr,
            flush=True,
        )

    if df[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} contains missing patient IDs")
    if df[ID_COL].duplicated().any():
        duplicate_count = int(df[ID_COL].duplicated().sum())
        raise ValueError(
            f"{ID_COL} contains {duplicate_count:,} duplicate rows. "
            "The master must have one row per unique patient."
        )

    ids = df[ID_COL].astype(str).copy()
    raw_features = df.drop(columns=ID_COL).copy()

    binary_columns_used = BINARY_FEATURES.copy()
    if args.drop_procedural:
        binary_columns_used = [
            c for c in binary_columns_used if c not in PROCEDURAL_FEATURES
        ]
    dropped_procedural = [
        c for c in BINARY_FEATURES if c not in binary_columns_used
    ]

    binary = pd.DataFrame(index=df.index)
    binary_missing_counts: dict[str, int] = {}
    for column in binary_columns_used:
        values = validate_binary(raw_features[column], column)
        binary_missing_counts[column] = int(values.isna().sum())
        # These are derived disease/medication/sex indicator columns, so missing
        # is interpreted as flag not present. Change upstream if NaN means unknown.
        binary[column] = values.fillna(0).astype(np.float32)


    continuous_raw = pd.DataFrame(index=df.index)
    continuous_imputed = pd.DataFrame(index=df.index)
    missing_indicators = pd.DataFrame(index=df.index)
    imputation_medians: dict[str, float] = {}

    for column in CONTINUOUS_FEATURES:
        values = pd.to_numeric(raw_features[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        continuous_raw[column] = values

        if values.notna().sum() == 0:
            raise ValueError(
                f"Continuous column {column!r} is completely missing and cannot be imputed"
            )

        median = float(values.median())
        imputation_medians[column] = median
        continuous_imputed[column] = values.fillna(median).astype(np.float32)

        if values.isna().any():
            missing_indicators[f"{column}_missing"] = (
                values.isna().astype(np.float32)
            )

    continuous_model = continuous_imputed.copy()
    scaler_name: str | None = None

    if not args.continuous_already_scaled:
        if args.scaler == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        continuous_model.loc[:, :] = scaler.fit_transform(continuous_model).astype(
            np.float32
        )
        scaler_name = args.scaler
    else:
        scaler_name = "already_scaled"


    clip_applied: float | None = None
    if args.clip >= 0:
        pre_clip_max_abs = float(continuous_model.abs().to_numpy().max())
        n_clipped = int((continuous_model.abs() > float(args.clip)).to_numpy().sum())
        continuous_model.loc[:, :] = continuous_model.clip(
            lower=-float(args.clip), upper=float(args.clip)
        ).astype(np.float32)
        clip_applied = float(args.clip)
        print(
            f"Winsorized continuous features to +/-{args.clip:g}: "
            f"clipped {n_clipped:,} values "
            f"(pre-clip max |value| = {pre_clip_max_abs:.2f})",
            flush=True,
        )


    X = pd.concat([binary, continuous_model, missing_indicators], axis=1)

    constant_columns = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_columns:
        print(
            "Dropping constant features: " + ", ".join(constant_columns),
            file=sys.stderr,
            flush=True,
        )
        X = X.drop(columns=constant_columns)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values remain after preprocessing: {bad}")

    model_matrix = X.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(model_matrix).all():
        raise ValueError("The clustering matrix contains non-finite values")

    max_pcs = min(model_matrix.shape[0] - 1, model_matrix.shape[1] - 1)
    if max_pcs < 2:
        raise ValueError(f"Too few rows/features for PCA: {model_matrix.shape}")
    n_pcs_used = min(args.n_pcs, max_pcs)

    print(
        f"Patients={len(df):,}; model features={X.shape[1]}; "
        f"PCs={n_pcs_used}; method={args.method}; metric={metric}",
        flush=True,
    )


    adata = ad.AnnData(
        X=model_matrix,
        obs=pd.DataFrame(index=pd.Index(ids.to_numpy(), name=ID_COL)),
        var=pd.DataFrame(index=pd.Index(X.columns, name="feature")),
    )

    sc.pp.pca(
        adata,
        n_comps=n_pcs_used,
        zero_center=True,
        svd_solver="randomized",
        random_state=args.random_state,
    )
    pca_scores = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)

    modularity: float | None = None


    if args.method == "scanpy_leiden":
        sc.pp.neighbors(
            adata,
            n_neighbors=args.n_neighbors,
            n_pcs=n_pcs_used,
            metric=metric,
            method="umap",
            random_state=args.random_state,
        )
        sc.tl.leiden(
            adata,
            resolution=args.resolution,
            key_added="cluster",
            flavor="igraph",
            directed=False,
            use_weights=True,
            n_iterations=-1,
            random_state=args.random_state,
        )
        cluster_labels = np.asarray(
            adata.obs["cluster"].astype(str), dtype=np.int32
        )
        graph = adata.obsp["connectivities"].tocsr()

    else:
        try:
            import phenograph
        except ImportError as exc:
            raise ImportError(
                "PhenoGraph mode requires the optional package: pip install phenograph"
            ) from exc

        if metric in {"cosine", "correlation"} and args.phenograph_nn_method != "brute":
            print(
                "WARNING: cosine/correlation generally require brute-force neighbor "
                "search in PhenoGraph; switching --phenograph-nn-method to brute.",
                file=sys.stderr,
                flush=True,
            )
            phenograph_nn_method = "brute"
        else:
            phenograph_nn_method = args.phenograph_nn_method

        cluster_labels, graph, modularity_value = phenograph.cluster(
            pca_scores,
            clustering_algo="leiden",
            k=args.n_neighbors,
            directed=False,
            prune=False,
            min_cluster_size=args.min_cluster_size,
            jaccard=True,
            primary_metric=metric,
            n_jobs=args.n_jobs,
            nn_method=phenograph_nn_method,
            resolution_parameter=args.resolution,
            n_iterations=-1,
            use_weights=True,
            seed=args.random_state,
        )
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
        graph = sparse.csr_matrix(graph)
        modularity = float(modularity_value)

        adata.obsp["connectivities"] = graph
        adata.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "params": {
                "method": "phenograph_jaccard",
                "n_neighbors": args.n_neighbors,
                "metric": metric,
                "n_pcs": n_pcs_used,
            },
        }
        adata.obs["cluster"] = pd.Categorical(cluster_labels.astype(str))

    if cluster_labels.shape[0] != len(df):
        raise RuntimeError(
            f"Clustering returned {len(cluster_labels):,} labels for {len(df):,} patients"
        )

    adata.obs["cluster"] = pd.Categorical(cluster_labels.astype(str))
    adata.uns["clustering_config"] = {
        "method": args.method,
        "n_neighbors": args.n_neighbors,
        "resolution": args.resolution,
        "metric": metric,
        "n_pcs": n_pcs_used,
        "random_state": args.random_state,
        "modularity": modularity,
    }


    labels = pd.DataFrame(
        {
            ID_COL: ids.to_numpy(),
            "cluster": cluster_labels,
            "clustering_method": args.method,
        }
    )
    labels.to_csv(output_dir / "patient_clusters.csv", index=False)
    labels.to_parquet(output_dir / "patient_clusters.parquet", index=False)

    cluster_sizes = (
        labels.groupby("cluster", observed=True)
        .size()
        .rename("n_patients")
        .to_frame()
        .sort_index()
    )
    cluster_sizes["percent"] = (
        100.0 * cluster_sizes["n_patients"] / len(labels)
    )
    cluster_sizes.to_csv(output_dir / "cluster_sizes.csv")

    if args.save_combined:
        combined = df.copy()
        combined["cluster"] = cluster_labels
        combined["clustering_method"] = args.method
        combined.to_parquet(output_dir / "master_with_clusters.parquet", index=False)


    binary_profile_wide = None
    if profile_binary_columns := [c for c in binary.columns if c in X.columns]:
        binary_profile = binary[profile_binary_columns].copy()
        binary_profile["cluster"] = cluster_labels
        binary_prevalence_long = (
            binary_profile.groupby("cluster", observed=True)
            .mean()
            .mul(100.0)
        )
        binary_prevalence = binary_prevalence_long.T
        binary_prevalence.to_csv(
            output_dir / "cluster_binary_prevalence_percent.csv"
        )
        binary_profile_wide = binary_prevalence_long.reset_index()

    
    continuous_profile_wide = (
        continuous_imputed.assign(cluster=cluster_labels)
        .groupby("cluster", observed=True)[CONTINUOUS_FEATURES]
        .mean()
        .reset_index()
    )
    pam_profile = cluster_sizes.reset_index().rename(columns={"percent": "pct_patients"})
    if binary_profile_wide is not None:
        pam_profile = pam_profile.merge(binary_profile_wide, on="cluster", how="left")
    pam_profile = pam_profile.merge(continuous_profile_wide, on="cluster", how="left")
    ordered_cols = ["cluster", "n_patients", "pct_patients"] + profile_binary_columns + CONTINUOUS_FEATURES
    ordered_cols = [c for c in ordered_cols if c in pam_profile.columns]
    pam_profile = pam_profile.loc[:, ordered_cols].sort_values("cluster")
    pam_profile.to_csv(output_dir / "cluster_profiles_pam_format.csv", index=False)

    if not missing_indicators.empty:
        missing_profile = missing_indicators[
            [c for c in missing_indicators.columns if c in X.columns]
        ].copy()
        missing_profile["cluster"] = cluster_labels
        missing_prevalence = (
            missing_profile.groupby("cluster", observed=True)
            .mean()
            .mul(100.0)
            .T
        )
        missing_prevalence.to_csv(
            output_dir / "cluster_missingness_prevalence_percent.csv"
        )

  
    observed_continuous = continuous_raw.copy()
    observed_continuous["cluster"] = cluster_labels
    observed_summary = observed_continuous.groupby("cluster", observed=True)[
        CONTINUOUS_FEATURES
    ].agg(["count", "mean", "median", "std", "min", "max"])
    observed_summary.to_csv(
        output_dir / "cluster_continuous_summary_observed.csv"
    )


    imputed_continuous = continuous_imputed.copy()
    imputed_continuous["cluster"] = cluster_labels
    imputed_summary = imputed_continuous.groupby("cluster", observed=True)[
        CONTINUOUS_FEATURES
    ].agg(["mean", "median", "std"])
    imputed_summary.to_csv(
        output_dir / "cluster_continuous_summary_imputed.csv"
    )

   
    model_profile = X.copy()
    model_profile["cluster"] = cluster_labels
    model_means = model_profile.groupby("cluster", observed=True).mean().T
    model_means.to_csv(output_dir / "cluster_feature_means_model_space.csv")

    overall_mean = X.mean(axis=0)
    overall_std = X.apply(safe_std, axis=0)
    standardized_cluster_difference = model_means.sub(overall_mean, axis=0).div(
        overall_std, axis=0
    )
    standardized_cluster_difference = standardized_cluster_difference.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    standardized_cluster_difference.to_csv(
        output_dir / "cluster_feature_standardized_difference.csv"
    )

    abs_standardized = standardized_cluster_difference.abs()
    driver_percentages = abs_standardized.div(abs_standardized.sum(axis=0), axis=1).mul(100.0)
    driver_percentages.to_csv(output_dir / "cluster_driver_percentages.csv")

    top_driver_rows: list[dict[str, Any]] = []
    for cluster in standardized_cluster_difference.columns:
        scores = standardized_cluster_difference[cluster]
        pct_scores = driver_percentages[cluster]
        ordered = scores.abs().sort_values(ascending=False).head(args.top_drivers)
        for rank, feature in enumerate(ordered.index, start=1):
            score = float(scores.loc[feature])
            top_driver_rows.append(
                {
                    "cluster": int(cluster),
                    "rank": rank,
                    "feature": feature,
                    "standardized_difference": score,
                    "driver_percent_of_phenotype": float(pct_scores.loc[feature]),
                    "direction": "higher/enriched" if score >= 0 else "lower/depleted",
                }
            )
    pd.DataFrame(top_driver_rows).to_csv(
        output_dir / "cluster_top_driving_features.csv", index=False
    )

    if args.make_driver_plots:
        save_driver_plots(driver_percentages, output_dir, args.driver_plot_top_n)


    variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"])
    pca_variance = pd.DataFrame(
        {
            "PC": np.arange(1, n_pcs_used + 1),
            "explained_variance_ratio": variance_ratio,
            "cumulative_explained_variance": np.cumsum(variance_ratio),
        }
    )
    pca_variance.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    pca_loadings = pd.DataFrame(
        adata.varm["PCs"],
        index=adata.var_names,
        columns=[f"PC{i}" for i in range(1, n_pcs_used + 1)],
    )
    pca_loadings.to_csv(output_dir / "pca_loadings.csv")

    pd.DataFrame(
        {
            "feature": X.columns,
            "feature_type": [
                "continuous_scaled"
                if c in CONTINUOUS_FEATURES
                else "missingness_indicator"
                if c.endswith("_missing")
                else "binary"
                for c in X.columns
            ],
        }
    ).to_csv(output_dir / "features_used.csv", index=False)


    if args.run_umap:
        sample_n = min(args.umap_sample_size, len(df))
        rng = np.random.default_rng(args.random_state)
        sample_index = np.sort(rng.choice(len(df), size=sample_n, replace=False))

        umap_adata = ad.AnnData(
            X=np.ascontiguousarray(pca_scores[sample_index], dtype=np.float32),
            obs=pd.DataFrame(
                {
                    ID_COL: ids.iloc[sample_index].to_numpy(),
                    "cluster": pd.Categorical(
                        cluster_labels[sample_index].astype(str)
                    ),
                },
                index=pd.Index(ids.iloc[sample_index].to_numpy(), name=ID_COL),
            ),
        )
        sc.pp.neighbors(
            umap_adata,
            n_neighbors=args.n_neighbors,
            use_rep="X",
            metric=metric,
            method="umap",
            random_state=args.random_state,
        )
        sc.tl.umap(umap_adata, random_state=args.random_state)

        umap_coordinates = pd.DataFrame(
            {
                ID_COL: ids.iloc[sample_index].to_numpy(),
                "cluster": cluster_labels[sample_index],
                "UMAP1": umap_adata.obsm["X_umap"][:, 0],
                "UMAP2": umap_adata.obsm["X_umap"][:, 1],
            }
        )
        umap_coordinates.to_csv(
            output_dir / "umap_sample_coordinates.csv", index=False
        )

    if args.save_graph:
        save_npz(output_dir / "graph_connectivities.npz", graph.tocsr())

    if args.save_h5ad:
        adata.write_h5ad(output_dir / "clustered.h5ad", compression="gzip")

    n_outliers = int((cluster_labels == -1).sum())
    unique_clusters = sorted(set(cluster_labels.tolist()) - {-1})

    metadata = {
        "input": args.input,
        "output_directory": output_dir,
        "method": args.method,
        "n_patients": len(df),
        "expected_n_patients": EXPECTED_N_PATIENTS,
        "n_input_binary_features": len(BINARY_FEATURES),
        "n_input_continuous_features": len(CONTINUOUS_FEATURES),
        "n_model_features": X.shape[1],
        "model_features": X.columns.tolist(),
        "drop_procedural": args.drop_procedural,
        "dropped_procedural_features": dropped_procedural,
        "dropped_constant_features": constant_columns,
        "binary_missing_counts": binary_missing_counts,
        "continuous_imputation_medians": imputation_medians,
        "continuous_scaler": scaler_name,
        "clip_scaled_continuous": clip_applied,
        "n_pcs_requested": args.n_pcs,
        "n_pcs_used": n_pcs_used,
        "n_neighbors": args.n_neighbors,
        "metric": metric,
        "resolution": args.resolution,
        "random_state": args.random_state,
        "n_clusters_excluding_outlier_label": len(unique_clusters),
        "cluster_labels_excluding_outlier": unique_clusters,
        "n_outliers_label_minus_one": n_outliers,
        "phenograph_modularity": modularity,
        "run_umap": args.run_umap,
        "umap_sample_size": min(args.umap_sample_size, len(df))
        if args.run_umap
        else None,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=make_json_safe)

    print("\nClustering complete", flush=True)
    print(f"Output directory: {output_dir.resolve()}", flush=True)
    print(f"Clusters excluding -1: {len(unique_clusters)}", flush=True)
    print(f"Outliers (-1): {n_outliers:,}", flush=True)
    if modularity is not None:
        print(f"PhenoGraph modularity Q: {modularity:.6f}", flush=True)
    print("\nCluster sizes:", flush=True)
    print(cluster_sizes.to_string(), flush=True)


if __name__ == "__main__":
    main()