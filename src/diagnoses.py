#!/usr/bin/env python3

import re
from pathlib import Path
from collections import defaultdict, Counter
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

INPUT_PATH = DATA_DIR / "Hypertension With Comorbidities 20250404.csv"
INPUT_PARQUET = PARQUET_DIR / "Hypertension_With_Comorbidities_20250404.parquet"
OUTPUT_PATH = OUTPUT_DIR / "icd10_3digit_patient_frequencies.csv"

ENCODING = "latin1"
SEP = ","
CSV_CHUNKSIZE = 500_000
BATCH_ROWS = 100_000
FORCE_RECONVERT = False

ROLLUP_TO_3CHAR = True
PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
ICD10_COL = "CURRENT_ICD10_LIST"
DXNAME_COL = "DX_NAME"

MISSING_TOKENS = {
    "", "NAN", "NA", "N/A", "NULL", "NONE", ".", "..", "...", "?", "-",
    "UNKNOWN", "UNSPECIFIED", "*UNSPECIFIED", "*UNK", "*DELETED"
}
SPLIT_PATTERN = re.compile(r"[;,|/]+")


def convert_csv_to_parquet(csv_path, parquet_path, sep=",", chunksize=500_000):
    """Convert a large CSV to Parquet in chunks, forcing all columns to string."""
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


def codes_in_cell(cell):
   
    text = str(cell).strip()
    if text.upper() in MISSING_TOKENS:
        return []

    out = []
    for tok in SPLIT_PATTERN.split(text):
        c = tok.strip().upper().replace(" ", "")
        if c in MISSING_TOKENS:
            continue
        if ROLLUP_TO_3CHAR:
            c = c.replace(".", "")[:3]
            if len(c) < 3:
                continue
        out.append(c)

    return list(dict.fromkeys(out))


def main():
    convert_csv_to_parquet(INPUT_PATH, INPUT_PARQUET, sep=SEP, chunksize=CSV_CHUNKSIZE)

    needed_cols = [PATIENT_COL, ICD10_COL]
   
    pf = pq.ParquetFile(INPUT_PARQUET)
    all_cols = pf.schema_arrow.names
    if DXNAME_COL in all_cols:
        needed_cols.append(DXNAME_COL)

    total_rows = 0
    rows_missing_icd10 = 0
    valid_patients = set()
    patients_with_code = set()

    code_patients = defaultdict(set)
    code_name_counts = defaultdict(Counter)

    print("\nProcessing diagnosis frequencies in Parquet batches...")

    for chunk in tqdm(iter_parquet_batches(INPUT_PARQUET, needed_cols), desc="Diagnoses batches"):
        total_rows += len(chunk)

        pat = chunk[PATIENT_COL].fillna("").astype(str).str.strip()
        valid_pat = pat.str.upper().ne("") & ~pat.str.upper().isin(MISSING_TOKENS)
        valid_patients.update(pat[valid_pat].unique())

        codes_series = chunk[ICD10_COL].apply(codes_in_cell)
        has_code = codes_series.apply(len).gt(0)
        rows_missing_icd10 += int((~has_code).sum())

        dx_values = chunk[DXNAME_COL].fillna("").astype(str) if DXNAME_COL in chunk.columns else pd.Series([""] * len(chunk))

        for patient, is_valid, code_list, dx_name in zip(pat, valid_pat, codes_series, dx_values):
            if not is_valid:
                continue
            if code_list:
                patients_with_code.add(patient)
            for code in code_list:
                code_patients[code].add(patient)
                if str(dx_name).strip():
                    code_name_counts[code][str(dx_name).strip()] += 1

    total_pts = len(valid_patients)
    pts_w_code = len(patients_with_code)

    print("\n--- missing-value summary ---")
    print(f"  rows total ................... {total_rows:,}")
    print(f"  rows w/ missing ICD-10 ....... {rows_missing_icd10:,}")
    print(f"  patients total ............... {total_pts:,}")
    print(f"  patients w/ >=1 ICD-10 ....... {pts_w_code:,}")
    print(f"  patients fully missing ICD-10  {total_pts - pts_w_code:,}")

    rows = []
    for code, patients in code_patients.items():
        n_patients = len(patients)
        example_dx_name = code_name_counts[code].most_common(1)[0][0] if code_name_counts[code] else ""
        rows.append({
            "icd10_code": code,
            "n_patients": n_patients,
            "pct_of_patients": round(n_patients / max(total_pts, 1) * 100, 2),
            "example_dx_name": example_dx_name,
        })

    result = pd.DataFrame(rows).sort_values("n_patients", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} ranked codes -> {OUTPUT_PATH}\n")
    print(result.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
