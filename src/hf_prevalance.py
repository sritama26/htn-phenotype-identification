#!/usr/bin/env python3

import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="model_training/training_data.csv")
    p.add_argument("--flag", default="has_heart_failure")
    p.add_argument("--cluster-col", default="cluster")
    p.add_argument("--out", default=None, help="optional CSV path to save the table")
    args = p.parse_args()

    df = pd.read_csv(args.data)
   
    hf = pd.to_numeric(df[args.flag], errors="coerce").fillna(0).astype(int).clip(0, 1)
    cl = df[args.cluster_col]

    total_hf = int(hf.sum())
    g = pd.DataFrame({"cluster": cl, "hf": hf})
    tbl = g.groupby("cluster").agg(n_patients=("hf", "size"),
                                   n_hf=("hf", "sum")).reset_index()
    tbl["hf_prevalence_%"] = (tbl["n_hf"] / tbl["n_patients"] * 100).round(2)
    tbl["share_of_hf_%"]  = (tbl["n_hf"] / total_hf * 100).round(2)
    tbl = tbl.sort_values("cluster").reset_index(drop=True)

    overall = hf.mean() * 100
    print(f"Total patients : {len(df):,}")
    print(f"HF patients    : {total_hf:,}  ({overall:.2f}% of cohort overall)\n")
    print(tbl.to_string(index=False))
    print("\nReading it:")
    print("  hf_prevalence_% -> is this cluster HF-enriched vs the "
          f"{overall:.1f}% baseline?  (rows do NOT sum to 100)")
    print("  share_of_hf_%   -> where do HF patients concentrate?  "
          "(sums to 100)")

    if args.out:
        tbl.to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()