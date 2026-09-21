#!/usr/bin/env python3

import sys
import re
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter


DATA_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama/batch_outputs")
OUT_DIR = DATA_DIR / "filtered_vitals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "vitals_within_15_days_of_diagnosis.csv"

OUTPUT_FILTERED_LONG_FILE = OUT_DIR / "vitals_within_15_days_cleaned_no_unit_conversion.csv"
OUTPUT_WIDE_RAW_FILE = OUT_DIR / "patient_vital_values_wide_before_imputation.csv"
OUTPUT_WIDE_IMPUTED_FILE = OUT_DIR / "patient_vital_values_wide_median_imputed.csv"
OUTPUT_MISSING_BY_VITAL_FILE = OUT_DIR / "vital_missing_counts_by_vital.csv"
OUTPUT_MISSING_BY_PATIENT_FILE = OUT_DIR / "vital_missing_counts_by_patient.csv"
OUTPUT_IMPUTATION_SUMMARY_FILE = OUT_DIR / "vital_imputation_summary.csv"
OUTPUT_BP_UNIT_ISSUES_FILE = OUT_DIR / "bp_unit_issues.csv"

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
VITAL_COL = "vital_type"
UNIT_COL = "vital_unit"
VALUE_COL = "vital_value_numeric"

ABS_DIFF_COL = "abs_day_difference"
VITAL_DAY_COL = "vital_day"
DAY_DIFF_COL = "day_difference_vital_minus_dx"

CHUNK_SIZE = 500_000

MISSING_UNIT_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "missing",
    "unknown"
}

VITALS_TO_KEEP_FINAL = {
    "height",
    "weight",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
}

BP_VITALS = {
    "bp_systolic",
    "bp_diastolic",
}

PREFERRED_VITAL_ORDER = [
    "height",
    "weight",
    "bmi",
    "bp_systolic",
    "bp_diastolic",
]


def normalize_vital_name(x):
    if pd.isna(x):
        return pd.NA

    s = str(x).strip().lower()

    s = s.replace("-", "_")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)

    while "__" in s:
        s = s.replace("__", "_")

    s = s.strip("_")

    vital_map = {
        # height
        "ht": "height",
        "height": "height",
        "height_cm": "height",
        "height_in": "height",

        # weight
        "wt": "weight",
        "weight": "weight",
        "weight_kg": "weight",
        "weight_lb": "weight",
        "weight_lbs": "weight",

        # BMI
        "bmi": "bmi",
        "body_mass_index": "bmi",
        "body_mass_index_bmi": "bmi",

        # systolic BP
        "systolic": "bp_systolic",
        "systolic_bp": "bp_systolic",
        "systolic_blood_pressure": "bp_systolic",
        "blood_pressure_systolic": "bp_systolic",
        "bp_systolic": "bp_systolic",

        # diastolic BP
        "diastolic": "bp_diastolic",
        "diastolic_bp": "bp_diastolic",
        "diastolic_blood_pressure": "bp_diastolic",
        "blood_pressure_diastolic": "bp_diastolic",
        "bp_diastolic": "bp_diastolic",
    }

    return vital_map.get(s, s)


def normalize_bp_unit(x):
   
    if pd.isna(x):
        return pd.NA

    raw = str(x).strip()

    if raw.lower() in MISSING_UNIT_STRINGS:
        return pd.NA

    s = raw.strip()
    s = s.replace(" ", "")
    s_lower = s.lower()

    unit_map = {
        "mmhg": "mmHg",
        "mm[hg]": "mmHg",
        "millimeterofmercury": "mmHg",
        "millimetersofmercury": "mmHg",
    }

    return unit_map.get(s_lower, raw)


def clean_day_value(x):
    if pd.isna(x):
        return None

    try:
        value = float(x)
    except Exception:
        return None

    if np.isnan(value):
        return None

    if value.is_integer():
        return int(value)

    return value


def list_to_string(values):
    cleaned = []

    for v in values:
        if isinstance(v, list):
            for item in v:
                item_clean = clean_day_value(item)
                if item_clean is not None:
                    cleaned.append(item_clean)
        else:
            item_clean = clean_day_value(v)
            if item_clean is not None:
                cleaned.append(item_clean)

    cleaned = sorted(set(cleaned))
    return str(cleaned)


def combine_day_strings(day_strings):
    all_days = []

    for s in day_strings:
        if pd.isna(s):
            continue

        if isinstance(s, list):
            all_days.extend(s)
            continue

        s = str(s).strip()

        if s == "" or s == "[]":
            continue

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                all_days.extend(parsed)
            else:
                all_days.append(parsed)
        except Exception:
            all_days.append(s)

    return list_to_string(all_days)


def reduce_to_best_patient_vital(df):
   
    if df.empty:
        return df

    df = df.sort_values(
        [
            PATIENT_COL,
            VITAL_COL,
            "_abs_diff_sort",
            "_vital_day_sort",
            "_row_order"
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True
        ],
        na_position="last"
    )

    df = df.drop_duplicates(
        subset=[PATIENT_COL, VITAL_COL],
        keep="first"
    )

    return df


def remove_existing_output_files():
    output_files = [
        OUTPUT_FILTERED_LONG_FILE,
        OUTPUT_WIDE_RAW_FILE,
        OUTPUT_WIDE_IMPUTED_FILE,
        OUTPUT_MISSING_BY_VITAL_FILE,
        OUTPUT_MISSING_BY_PATIENT_FILE,
        OUTPUT_IMPUTATION_SUMMARY_FILE,
        OUTPUT_BP_UNIT_ISSUES_FILE,
    ]

    for path in output_files:
        if path.exists():
            path.unlink()


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

remove_existing_output_files()

header = pd.read_csv(INPUT_FILE, nrows=0, encoding="latin1")
available_cols = set(header.columns)

required_cols = {
    PATIENT_COL,
    VITAL_COL,
    UNIT_COL,
    VALUE_COL
}

missing_required = required_cols - available_cols

if missing_required:
    raise ValueError(
        f"Missing required columns in input file: {sorted(missing_required)}"
    )


print("\nPASS 1: finding selected vital types...")

vital_patient_sets = defaultdict(set)
all_patient_ids = set()

total_input_rows = 0

dtype_map = {
    PATIENT_COL: "string",
    VITAL_COL: "string",
    UNIT_COL: "string",
}

reader = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    low_memory=False,
    chunksize=CHUNK_SIZE,
    dtype=dtype_map
)

for chunk_number, chunk in enumerate(reader, start=1):
    total_input_rows += len(chunk)

    chunk[VITAL_COL] = chunk[VITAL_COL].map(normalize_vital_name)

    
    chunk = chunk[
        chunk[VITAL_COL].isin(VITALS_TO_KEEP_FINAL)
    ].copy()

    if chunk.empty:
        continue

    valid_patient_mask = chunk[PATIENT_COL].notna()
    valid_vital_mask = chunk[VITAL_COL].notna()

    all_patient_ids.update(
        chunk.loc[valid_patient_mask, PATIENT_COL].astype(str).unique()
    )

    for vital, ids in (
        chunk.loc[
            valid_patient_mask & valid_vital_mask,
            [VITAL_COL, PATIENT_COL]
        ]
        .groupby(VITAL_COL)[PATIENT_COL]
        .unique()
        .items()
    ):
        vital_patient_sets[vital].update(
            pd.Series(ids).dropna().astype(str).tolist()
        )

    if chunk_number % 10 == 0:
        print(f"  processed chunks: {chunk_number:,}")


all_vitals_found = sorted(
    set(vital_patient_sets.keys()).intersection(VITALS_TO_KEEP_FINAL)
)

vitals_to_keep = [
    vital for vital in PREFERRED_VITAL_ORDER
    if vital in all_vitals_found
]

print("\nInput rows:", total_input_rows)
print("Input unique patients with selected vitals:", len(all_patient_ids))

print("\nSelected vital types found:")
print(vitals_to_keep)

if len(vitals_to_keep) == 0:
    print("\nNo selected vital types found. Stopping.")
    sys.exit(0)


print("\nPASS 2: processing selected vitals without unit conversion...")

best_parts = []
days_parts = []

bp_unit_issue_counts = Counter()

filtered_rows_total = 0
global_row_counter = 0
wrote_filtered_header = False

reader = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    low_memory=False,
    chunksize=CHUNK_SIZE,
    dtype=dtype_map
)

for chunk_number, chunk in enumerate(reader, start=1):
    chunk["_row_order"] = np.arange(
        global_row_counter,
        global_row_counter + len(chunk)
    )
    global_row_counter += len(chunk)

    chunk[VITAL_COL] = chunk[VITAL_COL].map(normalize_vital_name)

    
    chunk = chunk[
        chunk[VITAL_COL].isin(vitals_to_keep)
    ].copy()

    if chunk.empty:
        continue

    filtered_rows_total += len(chunk)

    chunk[VALUE_COL] = pd.to_numeric(
        chunk[VALUE_COL],
        errors="coerce"
    )

    chunk["cleaned_vital_value"] = chunk[VALUE_COL]


    chunk["_bp_unit_norm"] = pd.NA
    chunk["bp_unit_status"] = "not_bp"

    bp_mask = chunk[VITAL_COL].isin(BP_VITALS)

    chunk.loc[bp_mask, "_bp_unit_norm"] = (
        chunk.loc[bp_mask, UNIT_COL].map(normalize_bp_unit)
    )

    bp_unit_missing_mask = bp_mask & chunk["_bp_unit_norm"].isna()
    bp_unit_valid_mask = bp_mask & (chunk["_bp_unit_norm"] == "mmHg")
    bp_unit_invalid_mask = (
        bp_mask
        & chunk["_bp_unit_norm"].notna()
        & (chunk["_bp_unit_norm"] != "mmHg")
    )

    chunk.loc[bp_unit_missing_mask, "bp_unit_status"] = "missing_assumed_mmHg"
    chunk.loc[bp_unit_valid_mask, "bp_unit_status"] = "valid_mmHg"
    chunk.loc[bp_unit_invalid_mask, "bp_unit_status"] = "invalid_bp_unit_value_set_missing"

   
    chunk.loc[bp_unit_invalid_mask, "cleaned_vital_value"] = np.nan

    invalid_unit_counts = (
        chunk.loc[bp_unit_invalid_mask, [VITAL_COL, "_bp_unit_norm"]]
        .value_counts()
        .reset_index(name="count")
    )

    for _, row in invalid_unit_counts.iterrows():
        vital = row[VITAL_COL]
        unit = row["_bp_unit_norm"]
        count = int(row["count"])
        bp_unit_issue_counts[(vital, unit)] += count

    if ABS_DIFF_COL in chunk.columns:
        chunk["_abs_diff_sort"] = pd.to_numeric(
            chunk[ABS_DIFF_COL],
            errors="coerce"
        )
    else:
        chunk["_abs_diff_sort"] = 0

    chunk["_abs_diff_sort"] = chunk["_abs_diff_sort"].fillna(999999)

    if VITAL_DAY_COL in chunk.columns:
        chunk["_vital_day_sort"] = pd.to_datetime(
            chunk[VITAL_DAY_COL],
            errors="coerce"
        )
    else:
        chunk["_vital_day_sort"] = pd.NaT

   
    if DAY_DIFF_COL in chunk.columns:
        day_source_col = DAY_DIFF_COL
    elif ABS_DIFF_COL in chunk.columns:
        day_source_col = ABS_DIFF_COL
    else:
        day_source_col = None

    if day_source_col is not None:
        day_part = chunk[[PATIENT_COL, VITAL_COL, day_source_col]].copy()

        day_part[day_source_col] = pd.to_numeric(
            day_part[day_source_col],
            errors="coerce"
        )

        day_part = (
            day_part
            .dropna(subset=[PATIENT_COL, VITAL_COL])
            .groupby([PATIENT_COL, VITAL_COL])[day_source_col]
            .apply(lambda x: list_to_string(x.dropna().tolist()))
            .reset_index(name="days")
        )

        days_parts.append(day_part)

    chunk_best = reduce_to_best_patient_vital(chunk)

    keep_cols_for_best = [
        PATIENT_COL,
        VITAL_COL,
        VALUE_COL,
        UNIT_COL,
        "cleaned_vital_value",
        "_bp_unit_norm",
        "bp_unit_status",
        "_abs_diff_sort",
        "_vital_day_sort",
        "_row_order"
    ]

    optional_cols = [
        "diagnosis_day",
        "day_difference_vital_minus_dx",
        "abs_day_difference",
        "within_plus_minus_15_days",
        "DX_NAME",
        "CURRENT_ICD10_LIST"
    ]

    keep_cols_for_best.extend([
        c for c in optional_cols
        if c in chunk_best.columns and c not in keep_cols_for_best
    ])

    chunk_best = chunk_best[keep_cols_for_best].copy()
    best_parts.append(chunk_best)

    chunk.to_csv(
        OUTPUT_FILTERED_LONG_FILE,
        index=False,
        mode="a",
        header=not wrote_filtered_header
    )
    wrote_filtered_header = True

    if len(best_parts) >= 10:
        combined_best = pd.concat(best_parts, ignore_index=True)
        combined_best = reduce_to_best_patient_vital(combined_best)
        best_parts = [combined_best]

    if len(days_parts) >= 10:
        combined_days = pd.concat(days_parts, ignore_index=True)
        combined_days = (
            combined_days
            .groupby([PATIENT_COL, VITAL_COL])["days"]
            .apply(combine_day_strings)
            .reset_index()
        )
        days_parts = [combined_days]

    if chunk_number % 10 == 0:
        print(f"  processed chunks: {chunk_number:,}")


if len(best_parts) == 0:
    print("\nNo selected vital rows found.")
    sys.exit(0)

best_df = pd.concat(best_parts, ignore_index=True)
best_df = reduce_to_best_patient_vital(best_df)

print("\nRows processed:", total_input_rows)
print("Rows after selected vital filtering:", filtered_rows_total)
print("Patient-vital records after selecting closest vital:", len(best_df))


if len(days_parts) > 0:
    days_df = pd.concat(days_parts, ignore_index=True)

    days_df = (
        days_df
        .groupby([PATIENT_COL, VITAL_COL])["days"]
        .apply(combine_day_strings)
        .reset_index()
    )
else:
    days_df = pd.DataFrame(columns=[PATIENT_COL, VITAL_COL, "days"])



bp_issue_rows = []

for (vital, unit), count in bp_unit_issue_counts.items():
    bp_issue_rows.append({
        VITAL_COL: vital,
        "invalid_bp_unit": unit,
        "row_count_value_set_missing": count
    })

bp_unit_issues = pd.DataFrame(bp_issue_rows)

if bp_unit_issues.empty:
    bp_unit_issues = pd.DataFrame(columns=[
        VITAL_COL,
        "invalid_bp_unit",
        "row_count_value_set_missing"
    ])

bp_unit_issues.to_csv(OUTPUT_BP_UNIT_ISSUES_FILE, index=False)


print("\nCreating patient-level wide vital table...")

wide_raw = best_df.pivot(
    index=PATIENT_COL,
    columns=VITAL_COL,
    values="cleaned_vital_value"
)

wide_raw = wide_raw.reindex(columns=vitals_to_keep)
wide_raw = wide_raw.apply(pd.to_numeric, errors="coerce")


if not days_df.empty:
    wide_days = days_df.pivot(
        index=PATIENT_COL,
        columns=VITAL_COL,
        values="days"
    )

    wide_days = wide_days.reindex(index=wide_raw.index, columns=vitals_to_keep)
else:
    wide_days = pd.DataFrame(index=wide_raw.index, columns=vitals_to_keep)

wide_days = wide_days.fillna("[]")


wide_raw_output = pd.DataFrame({
    PATIENT_COL: wide_raw.index.astype(str)
})

for vital in vitals_to_keep:
    wide_raw_output[f"{vital}_value_raw"] = wide_raw[vital].to_numpy()

    # Keep unit columns only for BP
    if vital in BP_VITALS:
        wide_raw_output[f"{vital}_unit"] = "mmHg"

    wide_raw_output[f"{vital}_days"] = wide_days[vital].to_numpy()
    wide_raw_output[f"{vital}_missing_before_imputation"] = (
        wide_raw[vital].isna().astype(int).to_numpy()
    )

wide_raw_output.to_csv(OUTPUT_WIDE_RAW_FILE, index=False)


print("\nCalculating missing counts before imputation...")

missing_by_vital_rows = []

for vital in vitals_to_keep:
    patients_total = len(wide_raw)
    patients_with_value = int(wide_raw[vital].notna().sum())
    patients_missing = int(wide_raw[vital].isna().sum())

    missing_by_vital_rows.append({
        VITAL_COL: vital,
        "patients_total": patients_total,
        "patients_with_value_before_imputation": patients_with_value,
        "patients_missing_before_imputation": patients_missing,
        "percent_missing_before_imputation": round(
            patients_missing / patients_total * 100, 4
        ) if patients_total > 0 else 0
    })

missing_by_vital = pd.DataFrame(missing_by_vital_rows)
missing_by_vital.to_csv(OUTPUT_MISSING_BY_VITAL_FILE, index=False)

print("\nMissing counts by vital before imputation:")
print(missing_by_vital)


patient_missing_counts = pd.DataFrame({
    PATIENT_COL: wide_raw.index.astype(str),
    "num_vitals_missing_before_imputation": wide_raw.isna().sum(axis=1).to_numpy(),
    "num_vitals_present_before_imputation": wide_raw.notna().sum(axis=1).to_numpy(),
    "total_vital_types": len(vitals_to_keep)
})

patient_missing_counts["percent_vitals_missing_before_imputation"] = (
    patient_missing_counts["num_vitals_missing_before_imputation"]
    / patient_missing_counts["total_vital_types"]
    * 100
).round(4)

patient_missing_counts.to_csv(OUTPUT_MISSING_BY_PATIENT_FILE, index=False)

print("\nPatient-wise missing counts before imputation, first 20 rows:")
print(patient_missing_counts.head(20))


print("\nPerforming median imputation...")

vital_medians = wide_raw.median(skipna=True)

wide_imputed = wide_raw.copy()

for vital in vitals_to_keep:
    median_value = vital_medians[vital]

    if pd.isna(median_value):
        print(
            f"WARNING: Vital '{vital}' has no valid numeric values. "
            f"Cannot median-impute this vital."
        )
        continue

    wide_imputed[vital] = wide_imputed[vital].fillna(median_value)


imputation_rows = []

for vital in vitals_to_keep:
    missing_before = int(wide_raw[vital].isna().sum())
    nonmissing_before = int(wide_raw[vital].notna().sum())
    missing_after = int(wide_imputed[vital].isna().sum())

    imputation_rows.append({
        VITAL_COL: vital,
        "patients_total": len(wide_raw),
        "patients_with_value_before_imputation": nonmissing_before,
        "patients_missing_before_imputation": missing_before,
        "median_used_for_imputation": vital_medians[vital],
        "patients_missing_after_imputation": missing_after
    })

imputation_summary = pd.DataFrame(imputation_rows)
imputation_summary.to_csv(OUTPUT_IMPUTATION_SUMMARY_FILE, index=False)

print("\nImputation summary:")
print(imputation_summary)



wide_final = pd.DataFrame({
    PATIENT_COL: wide_raw.index.astype(str)
})

for vital in vitals_to_keep:
    wide_final[f"{vital}_value"] = wide_imputed[vital].to_numpy()

    # Keep unit columns only for BP
    if vital in BP_VITALS:
        wide_final[f"{vital}_unit"] = "mmHg"

    wide_final[f"{vital}_days"] = wide_days[vital].to_numpy()
    wide_final[f"{vital}_missing_before_imputation"] = (
        wide_raw[vital].isna().astype(int).to_numpy()
    )

wide_final.to_csv(OUTPUT_WIDE_IMPUTED_FILE, index=False)


print("\nDone.")

print("\nSaved cleaned long-format selected vitals to:")
print(OUTPUT_FILTERED_LONG_FILE)

print("\nSaved wide selected patient-vital file before imputation to:")
print(OUTPUT_WIDE_RAW_FILE)

print("\nSaved final wide median-imputed selected patient-vital file to:")
print(OUTPUT_WIDE_IMPUTED_FILE)

print("\nSaved missing counts by vital to:")
print(OUTPUT_MISSING_BY_VITAL_FILE)

print("\nSaved patient-wise missing counts to:")
print(OUTPUT_MISSING_BY_PATIENT_FILE)

print("\nSaved imputation summary to:")
print(OUTPUT_IMPUTATION_SUMMARY_FILE)

print("\nSaved BP unit issues to:")
print(OUTPUT_BP_UNIT_ISSUES_FILE)

print("\nFinal patient-level file shape:")
print(wide_final.shape)

print("\nFinal patient-level columns:")
print(wide_final.columns.tolist())

print("\nChecking unit columns in final output:")
unit_cols = [c for c in wide_final.columns if c.endswith("_unit")]
print(unit_cols)

print("\nUnique final BP units:")
for col in ["bp_systolic_unit", "bp_diastolic_unit"]:
    if col in wide_final.columns:
        print(f"{col}: {wide_final[col].dropna().unique()}")
    else:
        print(f"{col}: column not found")