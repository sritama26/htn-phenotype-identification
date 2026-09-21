#!/usr/bin/env python3


import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm



DATA_DIR = Path("/projects/f_miarc_1/Hypertension")
SCRIPT_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama")
OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = OUTPUT_DIR / "parquet_files"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

LABS_PATH = DATA_DIR / "Hypertension Labs 20250417.csv"
LABS_PARQUET = PARQUET_DIR / "Hypertension_Labs_20250417.parquet"

ENCODING = "latin1"
SEP = "|"
CSV_CHUNKSIZE = 500_000
BATCH_ROWS = 100_000
FORCE_RECONVERT = False

LONG_OUT = OUTPUT_DIR / "labs_long_clean.csv"
BYPATIENT_OUT = OUTPUT_DIR / "labs_by_patient.csv"

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
DAY_COL = "DAYS_SINCE_DX_FRST_OBSRVD"
COMPONENT_COL = "COMPONENT_NAME"
VALUE_COL = "ORD_VALUE"
UNIT_COL = "REFERENCE_UNIT"
KEEP_EXTRA = ["ABNORMAL_RESULTS_YN", "RESULT_FLAG", "RESULT_IN_RANGE_YN"]

MISSING_TOKENS = {
    "", "NAN", "NA", "N/A", "NULL", "NONE", ".", "?", "-",
    "SEE NOTE", "SEE COMMENT", "TNP", "CANCELLED", "PENDING"
}

LAB_PATTERNS = {
    "creatinine": (["CREATININE"], ["URINE", "CLEARANCE", "RATIO", "/CREAT", "BODY FLUID"]),
    "gfr": (["GFR", "GLOMERULAR FILTRATION"], []),
    "bun": (["UREA NITROGEN", "BUN"], ["RATIO", "/CREAT", "URINE"]),
    "sodium": (["SODIUM"], ["URINE", "BODY FLUID", "FECAL"]),
    "potassium": (["POTASSIUM"], ["URINE", "BODY FLUID"]),
    "calcium": (["CALCIUM"], ["URINE", "IONIZED"]),
    "magnesium": (["MAGNESIUM"], ["URINE"]),
    "glucose": (["GLUCOSE"], ["URINE", "CSF", "TOLERANCE", "BODY FLUID", "MEAN", "AVERAGE"]),
    "hba1c": (["A1C", "GLYCOHEMOGLOBIN", "GLYCATED HEMOGLOBIN", "HEMOGLOBIN A1"], []),
    "cholesterol_total": (["CHOLESTEROL"], ["HDL", "LDL", "VLDL", "NON", "RATIO", "/"]),
    "ldl": (["LDL"], ["VLDL", "RATIO", "/"]),
    "hdl": (["HDL"], ["NON", "RATIO", "/"]),
    "triglycerides": (["TRIGLYCERIDE"], []),
}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def convert_csv_to_parquet(csv_path, parquet_path, sep=",", chunksize=500_000):
    
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if parquet_path.exists() and not FORCE_RECONVERT:
        print(f"Parquet already exists, skipping conversion: {parquet_path}")
        return
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if parquet_path.exists() and FORCE_RECONVERT:
        print(f"Removing old parquet file: {parquet_path}")
        parquet_path.unlink()

    print(f"\nConverting CSV to Parquet")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")

    reader = pd.read_csv(
        csv_path,
        sep=sep,
        quotechar='"',
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        engine="c",
        low_memory=False,
    )

    writer = None
    schema = None
    total_rows = 0
    for chunk in tqdm(reader, desc=f"Converting {csv_path.name}"):
        total_rows += len(chunk)
        chunk.columns = chunk.columns.astype(str).str.strip()
        for col in chunk.columns:
            chunk[col] = chunk[col].astype("string")
        if schema is None:
            schema = pa.schema([pa.field(col, pa.string()) for col in chunk.columns])
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, schema, compression="snappy")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    print(f"Finished converting {csv_path.name}")
    print(f"Total rows written: {total_rows:,}")


def iter_parquet_batches(parquet_path, columns=None, batch_size=BATCH_ROWS):
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)
    if columns is not None:
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(f"Missing columns in {parquet_path}: {missing}. Available: {pf.schema_arrow.names}")
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
        yield batch.to_pandas()


def to_numeric(v):
    s = str(v).strip().replace(",", "")
    if s.upper() in MISSING_TOKENS:
        return np.nan
    m = _NUM_RE.search(s)
    return float(m.group()) if m else np.nan


def map_component_one(name):
    u = str(name).upper()
    hits = [lab for lab, (inc, exc) in LAB_PATTERNS.items()
            if any(t in u for t in inc) and not any(x in u for x in exc)]
    return hits[0] if hits else None, hits


def clean_lab_chunk(df):
    df = df.copy()
    mapped = df[COMPONENT_COL].map(lambda x: map_component_one(x)[0])
    df["lab"] = mapped
    clean = df[df["lab"].notna()].copy()

    clean["value"] = clean[VALUE_COL].map(to_numeric)
    clean["day"] = pd.to_numeric(clean[DAY_COL], errors="coerce")

    n0 = len(clean)
    drop_val = int(clean["value"].isna().sum())
    drop_day = int(clean["day"].isna().sum())

    clean = clean.dropna(subset=["value", "day"]).copy()
    clean["day"] = clean["day"].round(0).astype(int)

    cols = [PATIENT_COL, "lab", "day", "value", UNIT_COL] + [c for c in KEEP_EXTRA if c in clean.columns]
    clean = clean[cols].rename(columns={UNIT_COL: "unit"})
    return clean, n0, drop_val, drop_day


def main():
    convert_csv_to_parquet(LABS_PATH, LABS_PARQUET, sep=SEP, chunksize=CSV_CHUNKSIZE)

    needed_cols = [PATIENT_COL, DAY_COL, COMPONENT_COL, VALUE_COL, UNIT_COL] + KEEP_EXTRA
    pf = pq.ParquetFile(LABS_PARQUET)
    needed_cols = [c for c in needed_cols if c in pf.schema_arrow.names]

    if LONG_OUT.exists():
        LONG_OUT.unlink()

    first_write = True
    total_kept_lab_rows = 0
    total_drop_val = 0
    total_drop_day = 0
    total_usable = 0

    component_counts = {lab: Counter() for lab in LAB_PATTERNS}
    conflicts = {}

    # patient_lab_data[patient][lab] = {"days": [], "values": [], "unit_counts": Counter()}
    patient_lab_data = defaultdict(lambda: defaultdict(lambda: {"days": [], "values": [], "unit_counts": Counter()}))

    print("\nProcessing labs in Parquet batches...")

    for chunk in tqdm(iter_parquet_batches(LABS_PARQUET, needed_cols), desc="Lab batches"):
        # Conflict/reporting info from original component names.
        for name in chunk[COMPONENT_COL].dropna().unique():
            lab, hits = map_component_one(name)
            if len(hits) > 1:
                conflicts[name] = hits

        clean, n0, drop_val, drop_day = clean_lab_chunk(chunk)
        total_kept_lab_rows += n0
        total_drop_val += drop_val
        total_drop_day += drop_day
        total_usable += len(clean)

        if not clean.empty:
            clean.to_csv(LONG_OUT, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False

            for lab, sub in clean.groupby("lab"):
                component_names = chunk.loc[clean.index, COMPONENT_COL] if COMPONENT_COL in chunk.columns else None
                if component_names is not None:
                    component_counts[lab].update(component_names[clean["lab"] == lab].astype(str).tolist())

            for row in clean.itertuples(index=False):
                patient = getattr(row, PATIENT_COL)
                lab = row.lab
                rec = patient_lab_data[patient][lab]
                rec["days"].append(int(row.day))
                rec["values"].append(float(row.value))
                rec["unit_counts"][str(row.unit)] += 1

    if conflicts:
        print("WARNING: component names matching >1 lab (assigned to first); review LAB_PATTERNS:")
        for n, labs in list(conflicts.items())[:20]:
            print(f"  {n!r} -> {labs}")

    print(
        f"Kept-lab rows: {total_kept_lab_rows:,} | dropped non-numeric value: {total_drop_val:,} | "
        f"missing day: {total_drop_day:,} | usable: {total_usable:,}"
    )

    print("\nMatched COMPONENT_NAMEs per lab (measurement counts):")
    for lab in LAB_PATTERNS:
        names = component_counts[lab]
        tag = "  ".join(f"{nm} ({c})" for nm, c in names.most_common(4)) if names else "*** none ***"
        print(f"  {lab:<18} n={sum(names.values()):<7} {tag}")

    patients = sorted(patient_lab_data.keys())
    rows = []
    for patient in tqdm(patients, desc="Building labs_by_patient"):
        out = {PATIENT_COL: patient}
        for lab in LAB_PATTERNS:
            rec = patient_lab_data[patient].get(lab)
            if rec:
                pairs = sorted(zip(rec["days"], rec["values"]), key=lambda x: x[0])
                out[f"{lab}_days"] = [p[0] for p in pairs]
                out[f"{lab}_values"] = [p[1] for p in pairs]
                out[f"{lab}_unit"] = rec["unit_counts"].most_common(1)[0][0] if rec["unit_counts"] else ""
            else:
                out[f"{lab}_days"] = []
                out[f"{lab}_values"] = []
                out[f"{lab}_unit"] = ""
        rows.append(out)

    by_patient = pd.DataFrame(rows)

    
    array_cols = [c for c in by_patient.columns if c.endswith(("_days", "_values"))]
    for c in array_cols:
        by_patient[c] = by_patient[c].apply(json.dumps)

    by_patient.to_csv(BYPATIENT_OUT, index=False)

    n_pat = by_patient.shape[0]
    print(f"\nPatients with >=1 target lab: {n_pat:,}")
    print("Coverage (% of these patients with >=1 measurement):")
    for lab in LAB_PATTERNS:
        
        cov = by_patient[f"{lab}_values"].apply(lambda x: len(json.loads(x))).gt(0).mean() * 100 if n_pat else 0
        print(f"  {lab:<18} {cov:5.1f}%")

    print(f"\nSaved tidy long table -> {LONG_OUT}")
    print(f"Saved per-patient temporal arrays -> {BYPATIENT_OUT}")


if __name__ == "__main__":
    main()
