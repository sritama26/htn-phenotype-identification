#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama")

LAB_PATH = BASE_DIR / "batch_outputs/filtered_labs/patient_lab_values_wide_median_imputed.csv"
VITAL_PATH = BASE_DIR / "batch_outputs/filtered_vitals/patient_vital_values_wide_median_imputed.csv"
COMORB_PATH = BASE_DIR / "batch_outputs/patient_comorbidity_flags.csv"
MED_PATH = BASE_DIR / "batch_outputs/patient_med_groups.csv"


DEMO_PATH = BASE_DIR / "batch_outputs/filtered_labs/lab_within_15_days_filtered_units_converted.csv"

OUTPUT_DIR = BASE_DIR / "batch_outputs/master_file"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MASTER = OUTPUT_DIR / "master.csv"
OUT_SUMMARY = OUTPUT_DIR / "master_summary.txt"
OUT_COLUMNS = OUTPUT_DIR / "master_columns.txt"

ENCODING = "latin1"
CHUNKSIZE = 250_000

ID_COL = "PAT_MRN_ID_ENCRYPT"


EXPECTED_LAB_COLS = [
    "PAT_MRN_ID_ENCRYPT",
    "creatinine_value",
    "creatinine_unit",
    "gfr_value",
    "gfr_unit",
    "bun_value",
    "bun_unit",
    "sodium_value",
    "sodium_unit",
    "potassium_value",
    "potassium_unit",
    "calcium_value",
    "calcium_unit",
    "glucose_value",
    "glucose_unit",
]

EXPECTED_VITAL_COLS = [
    "PAT_MRN_ID_ENCRYPT",
    "height_value",
    "weight_value",
    "bmi_value",
    "bp_systolic_value",
    "bp_diastolic_value",
]

DEMOGRAPHIC_COLS = [
    "PAT_MRN_ID_ENCRYPT",
    "SEX",
    "RACE",
    "ETHNICITY",
]


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def get_columns(path):
    return pd.read_csv(path, nrows=0, encoding=ENCODING).columns.tolist()


def check_id_column(path):
    cols = get_columns(path)

    if ID_COL not in cols:
        raise ValueError(
            f"{ID_COL} not found in {path}\n"
            f"Available columns are:\n{cols}"
        )


def check_expected_columns(path, expected_cols, file_label):
    cols = get_columns(path)

    missing = [c for c in expected_cols if c not in cols]

    if missing:
        raise ValueError(
            f"{file_label} is missing expected columns:\n{missing}\n\n"
            f"Available columns are:\n{cols}"
        )


def get_unique_ids(path):
    ids = set()

    for chunk in pd.read_csv(
        path,
        usecols=[ID_COL],
        chunksize=CHUNKSIZE,
        encoding=ENCODING,
        low_memory=False,
    ):
        chunk[ID_COL] = chunk[ID_COL].astype(str).str.strip()

        chunk = chunk[
            chunk[ID_COL].notna()
            & (chunk[ID_COL] != "")
            & (chunk[ID_COL].str.lower() != "nan")
        ]

        ids.update(chunk[ID_COL].unique())

    return ids


def read_one_row_per_patient(path, common_ids, usecols=None):
  
    parts = []
    seen = set()

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=CHUNKSIZE,
        encoding=ENCODING,
        low_memory=False,
    ):
        chunk[ID_COL] = chunk[ID_COL].astype(str).str.strip()

        chunk = chunk[chunk[ID_COL].isin(common_ids)]
        chunk = chunk[~chunk[ID_COL].isin(seen)]

        chunk = chunk.drop_duplicates(subset=[ID_COL], keep="first")

        seen.update(chunk[ID_COL].unique())
        parts.append(chunk)

        if len(seen) == len(common_ids):
            break

    if not parts:
        return pd.DataFrame(columns=[ID_COL])

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=[ID_COL], keep="first")

    return df


def read_demographics_from_lab_file(common_ids):
    
    check_file(DEMO_PATH)
    check_id_column(DEMO_PATH)
    check_expected_columns(DEMO_PATH, DEMOGRAPHIC_COLS, "Demographics file")

    demo = read_one_row_per_patient(
        DEMO_PATH,
        common_ids,
        usecols=DEMOGRAPHIC_COLS,
    )

    demo_master = pd.DataFrame({ID_COL: sorted(common_ids)})
    demo_master = demo_master.merge(demo, on=ID_COL, how="left")

    return demo_master


def merge_without_duplicate_suffix(master, new_df, source_name):
    
    duplicate_cols = [
        c for c in new_df.columns
        if c != ID_COL and c in master.columns
    ]

    if duplicate_cols:
        print(f"\nWARNING: Duplicate columns from {source_name} skipped:")
        for c in duplicate_cols:
            print(f"  {c}")

    keep_cols = [ID_COL] + [
        c for c in new_df.columns
        if c != ID_COL and c not in master.columns
    ]

    return master.merge(new_df[keep_cols], on=ID_COL, how="inner")



def main():
    print("Checking input files...")

    for path in [LAB_PATH, VITAL_PATH, COMORB_PATH, MED_PATH]:
        check_file(path)
        check_id_column(path)
        print(path)

    print("\nChecking demographics source file...")
    check_file(DEMO_PATH)
    check_id_column(DEMO_PATH)
    check_expected_columns(DEMO_PATH, DEMOGRAPHIC_COLS, "Demographics file")
    print(DEMO_PATH)

    check_expected_columns(LAB_PATH, EXPECTED_LAB_COLS, "Lab file")
    check_expected_columns(VITAL_PATH, EXPECTED_VITAL_COLS, "Vital file")

    print("\nReading unique patient IDs from each file...")

    lab_ids = get_unique_ids(LAB_PATH)
    vital_ids = get_unique_ids(VITAL_PATH)
    comorb_ids = get_unique_ids(COMORB_PATH)
    med_ids = get_unique_ids(MED_PATH)

    common_ids = lab_ids & vital_ids & comorb_ids & med_ids

    print("\nUnique patients in each file:")
    print(f"Labs:          {len(lab_ids):,}")
    print(f"Vitals:        {len(vital_ids):,}")
    print(f"Comorbidities: {len(comorb_ids):,}")
    print(f"Medications:   {len(med_ids):,}")
    print(f"Common all 4:  {len(common_ids):,}")

    print("\nReading demographics from original filtered lab file...")
    demographics = read_demographics_from_lab_file(common_ids)
    print(f"Demographics shape: {demographics.shape}")

    print("\nReading labs using existing columns...")
    labs = read_one_row_per_patient(
        LAB_PATH,
        common_ids,
        usecols=EXPECTED_LAB_COLS,
    )
    print(f"Labs shape: {labs.shape}")

    print("\nReading vitals using existing columns...")
    vitals = read_one_row_per_patient(
        VITAL_PATH,
        common_ids,
        usecols=EXPECTED_VITAL_COLS,
    )
    print(f"Vitals shape: {vitals.shape}")

    print("\nReading comorbidity flags...")
    comorb = read_one_row_per_patient(COMORB_PATH, common_ids)
    print(f"Comorbidity shape: {comorb.shape}")

    print("\nReading medication groups...")
    meds = read_one_row_per_patient(MED_PATH, common_ids)
    print(f"Medication shape: {meds.shape}")

    print("\nMerging final master file...")

    master = pd.DataFrame({ID_COL: sorted(common_ids)})

    master = merge_without_duplicate_suffix(master, demographics, "demographics")
    master = merge_without_duplicate_suffix(master, comorb, "comorbidities")
    master = merge_without_duplicate_suffix(master, meds, "medications")
    master = merge_without_duplicate_suffix(master, labs, "labs")
    master = merge_without_duplicate_suffix(master, vitals, "vitals")

    master = master.drop_duplicates(subset=[ID_COL], keep="first")

    total_unique_patients = master[ID_COL].nunique()
    columns = master.columns.tolist()

    print("\nSaving outputs...")

    master.to_csv(OUT_MASTER, index=False)

    with open(OUT_COLUMNS, "w") as f:
        for col in columns:
            f.write(col + "\n")

    with open(OUT_SUMMARY, "w") as f:
        f.write(f"Total unique patients in master file: {total_unique_patients:,}\n")
        f.write(f"Total rows in master file: {len(master):,}\n")
        f.write(f"Total columns in master file: {len(columns):,}\n\n")

        f.write("Columns:\n")
        for col in columns:
            f.write(col + "\n")

    print("\nDONE")
    print(f"Master file saved to: {OUT_MASTER}")
    print(f"Summary saved to:     {OUT_SUMMARY}")
    print(f"Columns saved to:     {OUT_COLUMNS}")

    print("\nTotal unique patients in master file:")
    print(f"{total_unique_patients:,}")

    print("\nColumns in master file:")
    for col in columns:
        print(col)


if __name__ == "__main__":
    main()