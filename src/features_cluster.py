#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


MASTER_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master.csv"
OUTPUT_DIR  = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file"
OUTPUT_NAME = "master_clustering_ready.csv"
PATIENT_COL = "PAT_MRN_ID_ENCRYPT"

SCALER = "robust"          


CATEGORICAL = []

SEX_COL = "SEX"
SEX_BINARY_COL = "SEX_FEMALE"

GROUP_RACE = False
RACE_COL = "RACE"


def sex_to_female_binary(v):
  
    if pd.isna(v):
        return 0

    low = str(v).strip().lower()

    
    if "." in low:
        prefix, rest = low.split(".", 1)
        if len(prefix.strip()) <= 3:
            low = rest.strip()

    if low in {"female", "f"}:
        return 1

    return 0


def group_race(v):
   
    low = str(v).strip().lower()
    if low.startswith("d. asian"):            return "Asian"
    if low.startswith("f. multiracial"):      return "Multiracial"
    if low.startswith("e. pacific islander"): return "Pacific Islander"
    if low.startswith("z.") or low in ("missing", "", "nan", "none", "null"):
        return "Unknown/Other"
    if low.startswith("a. white"):            return "White"
    if low.startswith("b. black"):            return "Black or African American"
    if low.startswith("c. american indian"):  return "American Indian or Alaska Native"
    if low.startswith("hispanic"):            return "Hispanic"
    return "Unknown/Other"


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler
}


def read_any(path):
    base, _ = os.path.splitext(path)
    for p in dict.fromkeys([path, base + ".parquet", base + ".csv"]):
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError(f"None of {base}.(parquet|csv) found")


def _looks_numeric(s):
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def main():
    df = read_any(MASTER_PATH)
    n0 = len(df)
    print(f"Master: {n0:,} patients x {df.shape[1]} columns")

    ids = df[[PATIENT_COL]].copy()
    feat = df.drop(columns=[PATIENT_COL])

    
    if SEX_COL in feat.columns:
        feat[SEX_BINARY_COL] = feat[SEX_COL].map(sex_to_female_binary).astype("int8")

        sex_counts = feat[SEX_BINARY_COL].value_counts().reindex([1, 0], fill_value=0)

        print("\nSEX binary counts:")
        print(f"  Female, SEX_FEMALE = 1: {sex_counts.loc[1]:,}")
        print(f"  Rest,   SEX_FEMALE = 0: {sex_counts.loc[0]:,}")

       
        feat = feat.drop(columns=[SEX_COL])
    else:
        print("\nWARNING: SEX column not found. SEX_FEMALE was not created.")

 
    cat_cols = [c for c in CATEGORICAL if c in feat.columns]

    
    if GROUP_RACE and RACE_COL in feat.columns:
        feat[RACE_COL] = feat[RACE_COL].map(group_race)

        print("\nFinal RACE counts, patient-wise:")
        for val, cnt in feat[RACE_COL].value_counts().items():
            print(f"  {val:<34} {cnt:,}")

        print(f"  final RACE values: {sorted(feat[RACE_COL].unique())}")


    drop_cols = [
        c for c in feat.columns
        if c not in cat_cols
        and (
            c.endswith("_unit")
            or (feat[c].dtype == object and not _looks_numeric(feat[c]))
        )
    ]

    if drop_cols:
        print(
            f"\nDropping {len(drop_cols)} non-feature columns, e.g. units: "
            f"{', '.join(drop_cols[:8])}{' ...' if len(drop_cols) > 8 else ''}"
        )

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
    assert out.drop(columns=[PATIENT_COL]).isna().sum().sum() == 0, "NaNs remain"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    out.to_csv(csv_path, index=False)

    try:
        out.to_parquet(
            os.path.join(OUTPUT_DIR, OUTPUT_NAME.replace(".csv", ".parquet")),
            index=False
        )
    except Exception:
        pass

    print(f"\nScaler: {SCALER}")
    print(f"Cluster-ready matrix: {out.shape[0]:,} patients x {out.shape[1] - 1} features")
    print(f"  binary flags 0/1:    {len(binary)}")
    print(f"  continuous scaled:   {len(continuous)} -> {', '.join(continuous)}")
    print(f"  one-hot categorical: {D.shape[1]} from {', '.join(cat_cols) if cat_cols else 'none'}")

    if SEX_BINARY_COL in out.columns:
        final_sex_counts = out[SEX_BINARY_COL].value_counts().reindex([1, 0], fill_value=0)
        print("\nFinal SEX_FEMALE counts in output:")
        print(f"  SEX_FEMALE = 1, female: {final_sex_counts.loc[1]:,}")
        print(f"  SEX_FEMALE = 0, rest:   {final_sex_counts.loc[0]:,}")

    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()