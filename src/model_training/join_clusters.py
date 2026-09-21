#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd


LABELS_PATH   = "/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median_strdzn/phenograph/scanpy_leiden_k30_res0p4_cosine/patient_clusters.csv"
#"/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median_strdzn/forced_k/k6/pam_gower/all_features/assignments_final.csv"
MASTER_PATH   = "/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv"
OUTPUT_DIR    = "/projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training"
OUTPUT_NAME   = "training_data_pam.csv"
ID_COL        = "PAT_MRN_ID_ENCRYPT"   
LABEL_COL     = "cluster"              


def find_id_column(df: pd.DataFrame, preferred: str) -> str:
    """Return the patient-id column, tolerating minor header differences."""
    if preferred in df.columns:
        return preferred
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in (preferred.lower(), "pat_mrn_id_encrypt", "pat_mrn_id",
                      "patient_id", "mrn"):
        if candidate in lowered:
            return lowered[candidate]
    sys.exit(
        f"ERROR: could not find an id column like '{preferred}'. "
        f"Columns present: {list(df.columns)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=LABELS_PATH)
    p.add_argument("--master", default=MASTER_PATH)
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--out-name", default=OUTPUT_NAME)
    p.add_argument("--join", choices=["inner", "left"], default="inner",
                   help="inner = only patients in both files (default); "
                        "left = keep every master row, label may be blank")
    p.add_argument("--on-duplicate", choices=["error", "first", "mode"],
                   default="error",
                   help="what to do if a patient has multiple cluster rows")
    args = p.parse_args()

    labels = pd.read_csv(args.labels, dtype=str)
    master = pd.read_csv(args.master, dtype={ID_COL: str})

    lbl_id = find_id_column(labels, ID_COL)
    mst_id = find_id_column(master, ID_COL)

    if LABEL_COL not in labels.columns:
        sys.exit(f"ERROR: '{LABEL_COL}' not found in {args.labels}. "
                 f"Columns: {list(labels.columns)}")

   
    labels = labels[[lbl_id, LABEL_COL]].copy()
    labels[lbl_id] = labels[lbl_id].str.strip()
    master[mst_id] = master[mst_id].astype(str).str.strip()

   
    dup_count = int(labels.duplicated(subset=[lbl_id]).sum())
    if dup_count:
        if args.on_duplicate == "error":
            dups = labels.loc[labels.duplicated(lbl_id, keep=False), lbl_id].unique()
            sys.exit(f"ERROR: {dup_count} duplicate id rows in labels "
                     f"({len(dups)} patients). Re-run with "
                     f"--on-duplicate first|mode to resolve.")
        elif args.on_duplicate == "first":
            labels = labels.drop_duplicates(subset=[lbl_id], keep="first")
        else:  
            labels = (labels.groupby(lbl_id)[LABEL_COL]
                            .agg(lambda s: s.mode(dropna=True).iloc[0])
                            .reset_index())


    labels = labels.rename(columns={lbl_id: mst_id})
    merged = master.merge(labels, on=mst_id, how=args.join, validate="m:1")

    
    total_master   = len(master)
    labeled        = int(merged[LABEL_COL].notna().sum())
    unlabeled      = len(merged) - labeled
    unmatched_lbls = len(set(labels[mst_id]) - set(master[mst_id]))

    print("--- Join summary ---")
    print(f"master rows           : {total_master}")
    print(f"unique labels         : {labels[mst_id].nunique()}")
    print(f"join type             : {args.join}")
    print(f"rows in output        : {len(merged)}")
    print(f"rows with a label     : {labeled}")
    print(f"rows without a label  : {unlabeled}")
    print(f"labels w/o master row : {unmatched_lbls}")
    print("cluster distribution  :")
    print(merged[LABEL_COL].value_counts(dropna=False).to_string())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {len(merged)} rows -> {out_path}")


if __name__ == "__main__":
    main()
