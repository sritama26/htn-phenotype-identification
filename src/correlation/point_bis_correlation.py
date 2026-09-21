#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

MASTER_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master.csv"
OUTPUT_DIR  = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file"
CORR_DIR  = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/correlation"
OUTPUT_NAME = "master_clustering_ready.csv"

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"

SCALER = "robust"          


CATEGORICAL = []

SEX_COL = "SEX"
SEX_BINARY_COL = "SEX_FEMALE"

CORR_THRESHOLD = 0.80
MAX_CORR_PAIRS_TO_PRINT = 200

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler
}


# ----------------------------- HELPERS ---------------------------------------
def read_any(path):
  
    base, _ = os.path.splitext(path)

    for p in dict.fromkeys([path, base + ".parquet", base + ".csv"]):
        if os.path.exists(p):
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            return pd.read_csv(p)

    raise FileNotFoundError(f"None of {base}.(parquet|csv) found")


def _looks_numeric(s):
    
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def sex_to_female_binary(v):
   
    if pd.isna(v):
        return 0

    low = str(v).strip().lower()

    # Handles values like "F. Female", "A. Female", etc.
    if "." in low:
        prefix, rest = low.split(".", 1)
        if len(prefix.strip()) <= 3:
            low = rest.strip()

    if low in {"female", "f"}:
        return 1

    return 0


def detect_binary_and_continuous(feature_df):
  
    binary_cols = []
    continuous_cols = []

    for col in feature_df.columns:
        vals = set(pd.unique(feature_df[col].dropna()))

        if vals <= {0, 1}:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols


def point_biserial_correlation_filter(out, patient_col, threshold, output_dir):
    

    print("\n" + "=" * 80)
    print("POINT-BISERIAL CORRELATION ANALYSIS")
    print("=" * 80)

    feature_df = out.drop(columns=[patient_col], errors="ignore").copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Drop constant columns
    # ------------------------------------------------------------------
    nunique = feature_df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()

    if constant_cols:
        print(f"\nConstant columns found: {len(constant_cols):,}")
        print("These columns have only one unique value and will be dropped:")

        for c in constant_cols:
            print(f"  {c}")

        feature_df = feature_df.drop(columns=constant_cols)

    if feature_df.shape[1] <= 1:
        print("\nNot enough non-constant columns for correlation analysis.")

        dropped_cols = constant_cols.copy()
        out_filtered = out.drop(columns=dropped_cols, errors="ignore")

        dropped_path = os.path.join(output_dir, "dropped_correlated_columns.csv")
        pd.DataFrame({"dropped_column": dropped_cols}).to_csv(dropped_path, index=False)

        return out_filtered, pd.DataFrame(), dropped_cols

   
    binary_cols, continuous_cols = detect_binary_and_continuous(feature_df)

    print("\nFeature type summary:")
    print(f"  Binary features:        {len(binary_cols):,}")
    print(f"  Continuous features:    {len(continuous_cols):,}")
    print(f"  Point-Biserial pairs:   {len(binary_cols) * len(continuous_cols):,}")
    print(f"  Correlation cutoff:     abs(correlation) >= {threshold}")

    feature_type_df = pd.DataFrame({
        "feature": binary_cols + continuous_cols,
        "feature_type": ["binary"] * len(binary_cols) + ["continuous"] * len(continuous_cols)
    })

    feature_type_path = os.path.join(output_dir, "point_biserial_feature_types.csv")
    feature_type_df.to_csv(feature_type_path, index=False)

    if len(binary_cols) == 0 or len(continuous_cols) == 0:
        print("\nPoint-Biserial correlation cannot be computed because binary or continuous columns are missing.")

        dropped_cols = constant_cols.copy()
        out_filtered = out.drop(columns=dropped_cols, errors="ignore")

        dropped_path = os.path.join(output_dir, "dropped_correlated_columns.csv")
        pd.DataFrame({"dropped_column": dropped_cols}).to_csv(dropped_path, index=False)

        return out_filtered, pd.DataFrame(), dropped_cols


    rows = []
    pb_matrix = pd.DataFrame(index=binary_cols, columns=continuous_cols, dtype=float)

    for b_col in binary_cols:
        for c_col in continuous_cols:
            corr = feature_df[b_col].corr(feature_df[c_col], method="pearson")

            if pd.isna(corr):
                continue

            pb_matrix.loc[b_col, c_col] = corr

            rows.append({
                "binary_feature": b_col,
                "continuous_feature": c_col,
                "correlation_method": "Point-Biserial",
                "correlation": corr,
                "abs_correlation": abs(corr)
            })

    pair_df = pd.DataFrame(rows)

    matrix_path = os.path.join(output_dir, "point_biserial_correlation_matrix.csv")
    pb_matrix.to_csv(matrix_path)

    if pair_df.empty:
        print("\nNo valid Point-Biserial correlations found.")

        dropped_cols = constant_cols.copy()
        out_filtered = out.drop(columns=dropped_cols, errors="ignore")

        dropped_path = os.path.join(output_dir, "dropped_correlated_columns.csv")
        pd.DataFrame({"dropped_column": dropped_cols}).to_csv(dropped_path, index=False)

        return out_filtered, pair_df, dropped_cols

    pair_df = pair_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    all_pairs_path = os.path.join(output_dir, "point_biserial_all_correlation_pairs.csv")
    pair_df.to_csv(all_pairs_path, index=False)

    high_corr_pairs = pair_df[pair_df["abs_correlation"] >= threshold].copy()

    high_corr_path = os.path.join(output_dir, "point_biserial_high_correlation_pairs.csv")
    high_corr_pairs.to_csv(high_corr_path, index=False)

    print(f"\nPoint-Biserial matrix saved      -> {matrix_path}")
    print(f"All Point-Biserial pairs saved   -> {all_pairs_path}")
    print(f"High-correlation pairs saved     -> {high_corr_path}")
    print(f"Feature type file saved          -> {feature_type_path}")

    if high_corr_pairs.empty:
        print("\nNo highly correlated Point-Biserial pairs found.")

        dropped_cols = constant_cols.copy()
        out_filtered = out.drop(columns=dropped_cols, errors="ignore")

        dropped_path = os.path.join(output_dir, "dropped_correlated_columns.csv")
        pd.DataFrame({"dropped_column": dropped_cols}).to_csv(dropped_path, index=False)

        print(f"\nDropped column list saved -> {dropped_path}")

        return out_filtered, high_corr_pairs, dropped_cols

    print(f"\nHighly correlated Point-Biserial pairs found: {len(high_corr_pairs):,}")

    print("\nTop highly correlated Point-Biserial pairs:")

    max_print = min(MAX_CORR_PAIRS_TO_PRINT, len(high_corr_pairs))

    for _, row in high_corr_pairs.head(max_print).iterrows():
        print(
            f"  {row['binary_feature']} (binary)"
            f"  <-->  "
            f"{row['continuous_feature']} (continuous)"
            f"  corr={row['correlation']:.4f}"
            f"  abs_corr={row['abs_correlation']:.4f}"
        )

    if len(high_corr_pairs) > max_print:
        print(f"\nOnly printed top {max_print} pairs.")
        print("Full list is saved in point_biserial_high_correlation_pairs.csv")


    correlated_drop_cols = []

    kept_cols = set(feature_df.columns)

    for _, row in high_corr_pairs.iterrows():
        binary_feature = row["binary_feature"]
        continuous_feature = row["continuous_feature"]

        if binary_feature in kept_cols and continuous_feature in kept_cols:
            correlated_drop_cols.append(continuous_feature)
            kept_cols.remove(continuous_feature)

    dropped_cols = constant_cols + correlated_drop_cols
    dropped_cols = list(dict.fromkeys(dropped_cols))

    dropped_path = os.path.join(output_dir, "dropped_correlated_columns.csv")

    pd.DataFrame({
        "dropped_column": dropped_cols
    }).to_csv(dropped_path, index=False)

    print(f"\nColumns dropped: {len(dropped_cols):,}")

    for c in dropped_cols:
        print(f"  {c}")

    print(f"\nDropped column list saved -> {dropped_path}")

 
    out_filtered = out.drop(columns=dropped_cols, errors="ignore")

    print("\nFeature count before Point-Biserial filtering:", out.shape[1] - 1)
    print("Feature count after Point-Biserial filtering: ", out_filtered.shape[1] - 1)

    return out_filtered, high_corr_pairs, dropped_cols


def main():
    df = read_any(MASTER_PATH)
    n0 = len(df)

    print(f"Master: {n0:,} patients x {df.shape[1]} columns")

    ids = df[[PATIENT_COL]].copy()
    feat = df.drop(columns=[PATIENT_COL])

    if SEX_COL in feat.columns:
        feat[SEX_BINARY_COL] = feat[SEX_COL].map(sex_to_female_binary).astype("int8")

        sex_counts = feat[SEX_BINARY_COL].value_counts().reindex([1, 0], fill_value=0)

        print("\nSEX binary counts before filtering:")
        print(f"  SEX_FEMALE = 1, female: {sex_counts.loc[1]:,}")
        print(f"  SEX_FEMALE = 0, rest:   {sex_counts.loc[0]:,}")

        feat = feat.drop(columns=[SEX_COL])
    else:
        print("\nWARNING: SEX column not found. SEX_FEMALE was not created.")

    cat_cols = [c for c in CATEGORICAL if c in feat.columns]

    print("\nCategorical columns included in clustering:")
    if cat_cols:
        for c in cat_cols:
            print(f"  {c}")
    else:
        print("  None")

    drop_cols = [
        c for c in feat.columns
        if c not in cat_cols
        and (
            c.endswith("_unit")
            or (feat[c].dtype == object and not _looks_numeric(feat[c]))
        )
    ]

    if drop_cols:
        print(f"\nDropping {len(drop_cols):,} non-feature string/unit columns:")

        for c in drop_cols[:50]:
            print(f"  {c}")

        if len(drop_cols) > 50:
            print(f"  ... and {len(drop_cols) - 50:,} more")

    feat = feat.drop(columns=drop_cols)


    num = feat.drop(columns=cat_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

   
    binary = [
        c for c in num.columns
        if set(pd.unique(num[c].dropna())) <= {0, 1}
    ]

    continuous = [
        c for c in num.columns
        if c not in binary
    ]

    print("\nInitial feature breakdown before scaling/correlation filtering:")
    print(f"  Binary features:     {len(binary):,}")
    print(f"  Continuous features: {len(continuous):,}")
    print(f"  Categorical columns: {len(cat_cols):,}")

   
    B = (
        num[binary].fillna(0).astype("int8")
        if binary
        else pd.DataFrame(index=feat.index)
    )

   
    C = pd.DataFrame(index=feat.index)

    if continuous:
        Cc = num[continuous].copy()

        n_nan = int(Cc.isna().sum().sum())

        if n_nan:
            print(f"\nMedian-imputing {n_nan:,} residual NaNs in continuous columns")

        Cc = Cc.fillna(Cc.median()).fillna(0)

        scaler = _SCALERS[SCALER]()

        C = pd.DataFrame(
            scaler.fit_transform(Cc),
            columns=continuous,
            index=feat.index
        )

   
    D = (
        pd.get_dummies(
            feat[cat_cols],
            columns=cat_cols,
            dummy_na=False
        ).astype("int8")
        if cat_cols
        else pd.DataFrame(index=feat.index)
    )


    out = pd.concat(
        [
            ids.reset_index(drop=True),
            B.reset_index(drop=True),
            C.reset_index(drop=True),
            D.reset_index(drop=True)
        ],
        axis=1
    )

    assert len(out) == n0
    assert out.drop(columns=[PATIENT_COL]).isna().sum().sum() == 0, \
        "NaNs remain before correlation filtering"

    print("\nBefore heterogeneous correlation filtering:")
    print(f"  Patients: {out.shape[0]:,}")
    print(f"  Features: {out.shape[1] - 1:,}")

    out, high_corr_pairs, dropped_cols = point_biserial_correlation_filter(
    out=out,
    patient_col=PATIENT_COL,
    threshold=CORR_THRESHOLD,
    output_dir=OUTPUT_DIR
)

    assert len(out) == n0
    assert out.drop(columns=[PATIENT_COL]).isna().sum().sum() == 0, \
        "NaNs remain after correlation filtering"


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    out.to_csv(csv_path, index=False)

    parquet_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME.replace(".csv", ".parquet")
    )

    try:
        out.to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception:
        parquet_saved = False


    print("\n" + "=" * 80)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 80)

    print(f"\nScaler: {SCALER}")
    print(f"Correlation threshold: abs(correlation) >= {CORR_THRESHOLD}")

    print(f"\nFinal cluster-ready matrix:")
    print(f"  Patients: {out.shape[0]:,}")
    print(f"  Features: {out.shape[1] - 1:,}")

    print("\nOriginal feature counts before correlation filtering:")
    print(f"  Binary features:       {len(binary):,}")
    print(f"  Continuous features:   {len(continuous):,}")
    print(f"  One-hot categoricals:  {D.shape[1]:,}")

    print(f"\nDropped columns due to constant value or high correlation: {len(dropped_cols):,}")

    if dropped_cols:
        for c in dropped_cols:
            print(f"  {c}")

    if SEX_BINARY_COL in out.columns:
        final_sex_counts = out[SEX_BINARY_COL].value_counts().reindex([1, 0], fill_value=0)

        print("\nFinal SEX_FEMALE counts in output:")
        print(f"  SEX_FEMALE = 1, female: {final_sex_counts.loc[1]:,}")
        print(f"  SEX_FEMALE = 0, rest:   {final_sex_counts.loc[0]:,}")
    else:
        print("\nSEX_FEMALE was dropped by correlation filtering or was not created.")

    print(f"\nSaved final CSV -> {csv_path}")

    if parquet_saved:
        print(f"Saved final Parquet -> {parquet_path}")
    else:
        print("Parquet file was not saved because parquet support is unavailable.")

    print("\nSaved analysis files:")
    print(f"  {os.path.join(CORR_DIR, 'point_biserial_feature_types.csv')}")
    print(f"  {os.path.join(CORR_DIR, 'point_biserial_correlation_matrix.csv')}")
    print(f"  {os.path.join(CORR_DIR, 'point_biserial_all_correlation_pairs.csv')}")
    print(f"  {os.path.join(CORR_DIR, 'point_biserial_high_correlation_pairs.csv')}")
    print(f"  {os.path.join(CORR_DIR, 'dropped_correlated_columns.csv')}")


if __name__ == "__main__":
    main()