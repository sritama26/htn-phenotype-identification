#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


INPUT_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv"

OUTPUT_DIR = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/pca_outputs"

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"

VARIANCE_THRESHOLD = 0.85
TOP_N_FEATURES_PER_PC = 5

# For scatter plot only
MAX_SCATTER_POINTS = 10000
RANDOM_STATE = 42


def read_any(path):
   
    base, _ = os.path.splitext(path)

    for p in dict.fromkeys([path, base + ".parquet", base + ".csv"]):
        if os.path.exists(p):
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            return pd.read_csv(p)

    raise FileNotFoundError(f"Could not find {path}, {base}.parquet, or {base}.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    df = read_any(INPUT_PATH)

    print("=" * 80)
    print("PCA ON MASTER CLUSTERING READY FILE")
    print("=" * 80)

    print(f"\nInput file: {INPUT_PATH}")
    print(f"Shape: {df.shape[0]:,} patients x {df.shape[1]:,} columns")

    if PATIENT_COL not in df.columns:
        raise ValueError(f"{PATIENT_COL} not found in input file.")

    ids = df[[PATIENT_COL]].copy()

    X = df.drop(columns=[PATIENT_COL]).copy()


    X = X.apply(pd.to_numeric, errors="coerce")

    
    n_nan = int(X.isna().sum().sum())

    if n_nan > 0:
        print(f"\nWARNING: Found {n_nan:,} NaN values. Filling with column median.")
        X = X.fillna(X.median()).fillna(0)

    feature_names = X.columns.tolist()

    print(f"\nFeatures used for PCA: {X.shape[1]:,}")
    print(f"Patients used for PCA: {X.shape[0]:,}")

 
    print("\nRunning PCA...")

    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X)

    explained_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_ratio)

    
    n_components_85 = int(np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1)

    print("\nPCA variance summary:")
    print(f"  Variance threshold: {VARIANCE_THRESHOLD * 100:.1f}%")
    print(f"  Number of original features: {X.shape[1]:,}")
    print(f"  Number of PCs needed: {n_components_85:,}")
    print(f"  Cumulative variance explained: {cumulative_variance[n_components_85 - 1] * 100:.2f}%")

    explained_df = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(explained_ratio))],
        "component_number": np.arange(1, len(explained_ratio) + 1),
        "explained_variance": pca_full.explained_variance_,
        "explained_variance_ratio": explained_ratio,
        "explained_variance_percent": explained_ratio * 100,
        "cumulative_variance_ratio": cumulative_variance,
        "cumulative_variance_percent": cumulative_variance * 100,
        "selected_for_85_percent": np.arange(1, len(explained_ratio) + 1) <= n_components_85
    })

    explained_path = os.path.join(OUTPUT_DIR, "pca_explained_variance.csv")
    explained_df.to_csv(explained_path, index=False)

 
    selected_pc_names = [f"PC{i + 1}" for i in range(n_components_85)]

    pca_scores = pd.DataFrame(
        X_pca_full[:, :n_components_85],
        columns=selected_pc_names
    )

    pca_scores_out = pd.concat(
        [
            ids.reset_index(drop=True),
            pca_scores.reset_index(drop=True)
        ],
        axis=1
    )

    scores_path = os.path.join(OUTPUT_DIR, "pca_scores_85var.csv")
    pca_scores_out.to_csv(scores_path, index=False)

    parquet_path = os.path.join(OUTPUT_DIR, "pca_scores_85var.parquet")

    try:
        pca_scores_out.to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception:
        parquet_saved = False

    
    scatter_path = None

    if n_components_85 >= 2:
        scatter_df = pca_scores_out[[PATIENT_COL, "PC1", "PC2"]].copy()

        if len(scatter_df) > MAX_SCATTER_POINTS:
            scatter_df_plot = scatter_df.sample(
                n=MAX_SCATTER_POINTS,
                random_state=RANDOM_STATE
            )
            scatter_note = f"Sampled {MAX_SCATTER_POINTS:,} patients for plotting"
        else:
            scatter_df_plot = scatter_df.copy()
            scatter_note = f"Plotted all {len(scatter_df_plot):,} patients"

        pc1_var = explained_ratio[0] * 100
        pc2_var = explained_ratio[1] * 100

        plt.figure(figsize=(9, 7))
        plt.scatter(
            scatter_df_plot["PC1"],
            scatter_df_plot["PC2"],
            s=8,
            alpha=0.5
        )

        plt.xlabel(f"PC1 ({pc1_var:.2f}% variance)")
        plt.ylabel(f"PC2 ({pc2_var:.2f}% variance)")
        plt.title("PCA scatter plot: PC1 vs PC2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_path = os.path.join(OUTPUT_DIR, "pca_scatter_pc1_pc2.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()

        print(f"\nPCA scatter plot saved -> {scatter_path}")
        print(f"  {scatter_note}")
    else:
        print("\nPCA scatter plot was not created because fewer than 2 PCs were selected.")

    loadings = pd.DataFrame(
        pca_full.components_[:n_components_85].T,
        index=feature_names,
        columns=selected_pc_names
    )

    loadings.index.name = "feature"

    loadings_path = os.path.join(OUTPUT_DIR, "pca_loadings_85var.csv")
    loadings.to_csv(loadings_path)


    top_rows = []

    for pc in selected_pc_names:
        pc_loadings = loadings[pc].copy()

        top_features = (
            pc_loadings
            .abs()
            .sort_values(ascending=False)
            .head(TOP_N_FEATURES_PER_PC)
            .index
        )

        for rank, feature in enumerate(top_features, start=1):
            top_rows.append({
                "PC": pc,
                "rank": rank,
                "feature": feature,
                "loading": pc_loadings.loc[feature],
                "abs_loading": abs(pc_loadings.loc[feature])
            })

    top_features_df = pd.DataFrame(top_rows)

    top_features_path = os.path.join(OUTPUT_DIR, "pca_top_features_85var.csv")
    top_features_df.to_csv(top_features_path, index=False)


    plt.figure(figsize=(10, 6))

    plt.plot(
        explained_df["component_number"],
        explained_df["cumulative_variance_percent"],
        marker="o"
    )

    plt.axhline(
        y=VARIANCE_THRESHOLD * 100,
        linestyle="--",
        label=f"{VARIANCE_THRESHOLD * 100:.0f}% variance"
    )

    plt.axvline(
        x=n_components_85,
        linestyle="--",
        label=f"{n_components_85} PCs"
    )

    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


    summary_path = os.path.join(OUTPUT_DIR, "pca_summary.txt")

    with open(summary_path, "w") as f:
        f.write("PCA SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Input file: {INPUT_PATH}\n")
        f.write(f"Patients: {X.shape[0]:,}\n")
        f.write(f"Original features: {X.shape[1]:,}\n")
        f.write(f"Variance threshold: {VARIANCE_THRESHOLD * 100:.1f}%\n")
        f.write(f"Number of PCs selected: {n_components_85:,}\n")
        f.write(
            f"Cumulative variance explained: "
            f"{cumulative_variance[n_components_85 - 1] * 100:.2f}%\n\n"
        )

        f.write("Selected PCs:\n")

        for i in range(n_components_85):
            f.write(
                f"  PC{i + 1}: "
                f"individual variance = {explained_ratio[i] * 100:.2f}%, "
                f"cumulative = {cumulative_variance[i] * 100:.2f}%\n"
            )

        f.write("\nOutput files:\n")
        f.write(f"  PCA scores CSV: {scores_path}\n")
        f.write(f"  Explained variance CSV: {explained_path}\n")
        f.write(f"  PCA loadings CSV: {loadings_path}\n")
        f.write(f"  Top features CSV: {top_features_path}\n")
        f.write(f"  Cumulative variance plot: {plot_path}\n")

        if scatter_path is not None:
            f.write(f"  PCA scatter plot: {scatter_path}\n")

        if parquet_saved:
            f.write(f"  PCA scores parquet: {parquet_path}\n")

   
    print("\n" + "=" * 80)
    print("PCA OUTPUT SUMMARY")
    print("=" * 80)

    print("\nSaved PCA outputs:")
    print(f"  PCA scores CSV:         {scores_path}")

    if parquet_saved:
        print(f"  PCA scores parquet:     {parquet_path}")

    print(f"  Explained variance:     {explained_path}")
    print(f"  PCA loadings:           {loadings_path}")
    print(f"  Top features per PC:    {top_features_path}")
    print(f"  Cumulative plot:        {plot_path}")

    if scatter_path is not None:
        print(f"  PCA scatter plot:       {scatter_path}")

    print(f"  Summary:                {summary_path}")

    print("\nTop explained variance rows:")
    print(explained_df.head(n_components_85).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()