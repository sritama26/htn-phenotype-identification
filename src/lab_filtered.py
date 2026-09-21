#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter


OUT_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/filtered_labs")
DATA_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama/batch_outputs")

INPUT_FILE = DATA_DIR / "lab_within_15_days_of_diagnosis.csv"

OUTPUT_FILTERED_FILE = OUT_DIR / "lab_within_15_days_filtered_units_converted.csv"
OUTPUT_UNIT_SUMMARY_FILE = OUT_DIR / "lab_empty_unit_summary.csv"
OUTPUT_COMMON_UNITS_FILE = OUT_DIR / "lab_common_units.csv"
OUTPUT_WIDE_RAW_FILE = OUT_DIR / "patient_lab_values_wide_before_imputation.csv"
OUTPUT_WIDE_IMPUTED_FILE = OUT_DIR / "patient_lab_values_wide_median_imputed.csv"
OUTPUT_IMPUTATION_SUMMARY_FILE = OUT_DIR / "patient_lab_imputation_summary.csv"
OUTPUT_CONVERSION_ISSUES_FILE = OUT_DIR / "lab_unit_conversion_issues.csv"

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
LAB_COL = "lab_name"
UNIT_COL = "lab_unit"
VALUE_COL = "lab_value_numeric"

ABS_DIFF_COL = "abs_day_difference"
LAB_DAY_COL = "lab_day"

CHUNK_SIZE = 500_000

MISSING_UNIT_THRESHOLD = 30.0


LABS_TO_REMOVE_BY_TABLE = {
    "magnesium",
    "hba1c",
    "cholesterol_total",
    "ldl",
    "hdl",
    "triglycerides",
}

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

PREFERRED_LAB_ORDER = [
    "creatinine",
    "gfr",
    "bun",
    "sodium",
    "potassium",
    "calcium",
    "glucose",
    "magnesium",
    "hba1c",
    "cholesterol_total",
    "ldl",
    "hdl",
    "triglycerides"
]


def normalize_lab_name(x):
    if pd.isna(x):
        return pd.NA

    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    while "__" in s:
        s = s.replace("__", "_")

    return s


def normalize_unit(x):
    if pd.isna(x):
        return pd.NA

    raw = str(x).strip()

    if raw.lower() in MISSING_UNIT_STRINGS:
        return pd.NA

    s = raw
    s = s.replace("µ", "u")
    s = s.replace("μ", "u")
    s = s.replace("²", "2")
    s = s.replace("^2", "2")
    s = s.replace(" ", "")
    s = s.lower()

    unit_map = {
        "mg/dl": "mg/dL",
        "mg/dl.": "mg/dL",
        "mg/ dl": "mg/dL",
        "mg/100ml": "mg/dL",
        "mg/l": "mg/L",

        "g/dl": "g/dL",
        "g/l": "g/L",

        "mmol/l": "mmol/L",
        "mmol/l.": "mmol/L",
        "mmol": "mmol/L",

        "meq/l": "mEq/L",
        "meq/l.": "mEq/L",

        "umol/l": "umol/L",
        "µmol/l": "umol/L",
        "μmol/l": "umol/L",
        "mcmol/l": "umol/L",

        "ml/min": "mL/min",
        "ml/min/1.73m2": "mL/min/1.73m2",
        "ml/min/1.73": "mL/min/1.73m2",
        "ml/min/1.73m^2": "mL/min/1.73m2",

        "%": "%",
        "percent": "%",

        "mmol/mol": "mmol/mol",
    }

    return unit_map.get(s, raw.strip())


def convert_values_to_common_unit(values, lab, from_unit, to_unit):
  

    values = pd.to_numeric(values, errors="coerce")

    if pd.isna(from_unit) or pd.isna(to_unit):
        return pd.Series(np.nan, index=values.index), False, "missing_unit"

    from_unit = str(from_unit)
    to_unit = str(to_unit)

    if from_unit == to_unit:
        return values, True, "same_unit"

    lab = str(lab).lower()

    if lab in {"sodium", "potassium"}:
        if {from_unit, to_unit} == {"mmol/L", "mEq/L"}:
            return values, True, "equivalent_mmol_meq"

    if lab == "gfr":
        gfr_units = {"mL/min", "mL/min/1.73m2"}
        if from_unit in gfr_units and to_unit in gfr_units:
            return values, True, "treated_as_equivalent_gfr_units"

    conversion_factors = {
        # Creatinine
        ("creatinine", "mg/dL", "umol/L"): 88.4,
        ("creatinine", "umol/L", "mg/dL"): 1 / 88.4,

        # BUN
        ("bun", "mg/dL", "mmol/L"): 0.357,
        ("bun", "mmol/L", "mg/dL"): 1 / 0.357,

        # Calcium
        ("calcium", "mg/dL", "mmol/L"): 0.2495,
        ("calcium", "mmol/L", "mg/dL"): 1 / 0.2495,

        # Glucose
        ("glucose", "mg/dL", "mmol/L"): 1 / 18.0182,
        ("glucose", "mmol/L", "mg/dL"): 18.0182,

        # Magnesium
        ("magnesium", "mg/dL", "mmol/L"): 0.4114,
        ("magnesium", "mmol/L", "mg/dL"): 1 / 0.4114,

        # Cholesterol, LDL, HDL
        ("cholesterol_total", "mg/dL", "mmol/L"): 0.02586,
        ("cholesterol_total", "mmol/L", "mg/dL"): 1 / 0.02586,

        ("ldl", "mg/dL", "mmol/L"): 0.02586,
        ("ldl", "mmol/L", "mg/dL"): 1 / 0.02586,

        ("hdl", "mg/dL", "mmol/L"): 0.02586,
        ("hdl", "mmol/L", "mg/dL"): 1 / 0.02586,

        # Triglycerides
        ("triglycerides", "mg/dL", "mmol/L"): 0.01129,
        ("triglycerides", "mmol/L", "mg/dL"): 1 / 0.01129,
    }

    key = (lab, from_unit, to_unit)

    if key in conversion_factors:
        factor = conversion_factors[key]
        return values * factor, True, "converted_by_factor"


    if lab == "hba1c":
        if from_unit == "%" and to_unit == "mmol/mol":
            converted = (values - 2.15) * 10.929
            return converted, True, "converted_hba1c_percent_to_mmolmol"

        if from_unit == "mmol/mol" and to_unit == "%":
            converted = (0.09148 * values) + 2.152
            return converted, True, "converted_hba1c_mmolmol_to_percent"

    return pd.Series(np.nan, index=values.index), False, "unsupported_conversion"


def reduce_to_best_patient_lab(df):
 
    if df.empty:
        return df

    sort_cols = [
        PATIENT_COL,
        LAB_COL,
        "_abs_diff_sort",
        "_lab_day_sort",
        "_row_order"
    ]

    ascending = [
        True,
        True,
        True,
        False,
        True
    ]

    df = df.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last"
    )

    df = df.drop_duplicates(
        subset=[PATIENT_COL, LAB_COL],
        keep="first"
    )

    return df


def remove_existing_output_files():
    output_files = [
        OUTPUT_FILTERED_FILE,
        OUTPUT_UNIT_SUMMARY_FILE,
        OUTPUT_COMMON_UNITS_FILE,
        OUTPUT_WIDE_RAW_FILE,
        OUTPUT_WIDE_IMPUTED_FILE,
        OUTPUT_IMPUTATION_SUMMARY_FILE,
        OUTPUT_CONVERSION_ISSUES_FILE,
    ]

    for path in output_files:
        if path.exists():
            path.unlink()


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
remove_existing_output_files()

header = pd.read_csv(INPUT_FILE, nrows=0, encoding="latin1")
available_cols = set(header.columns)

required_cols = {
    PATIENT_COL,
    LAB_COL,
    UNIT_COL,
    VALUE_COL
}

missing_required = required_cols - available_cols

if missing_required:
    raise ValueError(
        f"Missing required columns in input file: {sorted(missing_required)}"
    )


print("\nPASS 1: calculating empty-unit summary and common units...")

lab_total_rows = Counter()
lab_empty_unit_rows = Counter()
lab_patient_sets = defaultdict(set)
lab_empty_unit_patient_sets = defaultdict(set)
lab_unit_counts = defaultdict(Counter)

total_input_rows = 0
all_input_patients = set()

dtype_map = {
    PATIENT_COL: "string",
    LAB_COL: "string",
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

    chunk[LAB_COL] = chunk[LAB_COL].map(normalize_lab_name)
    chunk["_unit_norm"] = chunk[UNIT_COL].map(normalize_unit)

    valid_patient_mask = chunk[PATIENT_COL].notna()
    valid_lab_mask = chunk[LAB_COL].notna()

    all_input_patients.update(
        chunk.loc[valid_patient_mask, PATIENT_COL].astype(str).unique()
    )

    lab_counts = chunk.loc[valid_lab_mask, LAB_COL].value_counts(dropna=True)
    for lab, count in lab_counts.items():
        lab_total_rows[lab] += int(count)

    for lab, ids in (
        chunk.loc[valid_patient_mask & valid_lab_mask, [LAB_COL, PATIENT_COL]]
        .groupby(LAB_COL)[PATIENT_COL]
        .unique()
        .items()
    ):
        lab_patient_sets[lab].update(pd.Series(ids).dropna().astype(str).tolist())

    unit_missing_mask = chunk["_unit_norm"].isna()

    empty_lab_counts = chunk.loc[
        valid_lab_mask & unit_missing_mask,
        LAB_COL
    ].value_counts(dropna=True)

    for lab, count in empty_lab_counts.items():
        lab_empty_unit_rows[lab] += int(count)

    for lab, ids in (
        chunk.loc[
            valid_patient_mask & valid_lab_mask & unit_missing_mask,
            [LAB_COL, PATIENT_COL]
        ]
        .groupby(LAB_COL)[PATIENT_COL]
        .unique()
        .items()
    ):
        lab_empty_unit_patient_sets[lab].update(
            pd.Series(ids).dropna().astype(str).tolist()
        )

    unit_count_df = (
        chunk.loc[
            valid_lab_mask & chunk["_unit_norm"].notna(),
            [LAB_COL, "_unit_norm"]
        ]
        .value_counts()
        .reset_index(name="count")
    )

    for _, row in unit_count_df.iterrows():
        lab = row[LAB_COL]
        unit = row["_unit_norm"]
        count = int(row["count"])
        lab_unit_counts[lab][unit] += count

    if chunk_number % 10 == 0:
        print(f"  processed chunks: {chunk_number:,}")


summary_rows = []

for lab in sorted(lab_patient_sets.keys()):
    total_patients = len(lab_patient_sets[lab])
    patients_empty_unit = len(lab_empty_unit_patient_sets[lab])

    percent_empty_unit = (
        patients_empty_unit / total_patients * 100
        if total_patients > 0
        else 0
    )

    common_unit = None
    common_unit_count = 0

    if len(lab_unit_counts[lab]) > 0:
        common_unit, common_unit_count = lab_unit_counts[lab].most_common(1)[0]

    summary_rows.append({
        LAB_COL: lab,
        "total_rows": lab_total_rows[lab],
        "total_patients": total_patients,
        "rows_empty_unit": lab_empty_unit_rows[lab],
        "patients_empty_unit": patients_empty_unit,
        "percent_empty_unit": round(percent_empty_unit, 4),
        "most_common_unit": common_unit,
        "most_common_unit_row_count": common_unit_count,
        "all_units_seen": dict(lab_unit_counts[lab]),
    })

unit_summary = pd.DataFrame(summary_rows)

unit_summary = unit_summary.sort_values(
    "percent_empty_unit",
    ascending=False
)

unit_summary.to_csv(OUTPUT_UNIT_SUMMARY_FILE, index=False)

all_labs_found = set(unit_summary[LAB_COL].dropna().astype(str))

labs_to_remove = sorted(
    LABS_TO_REMOVE_BY_TABLE.intersection(all_labs_found)
)

labs_to_keep_set = all_labs_found - set(labs_to_remove)

labs_to_keep = [
    lab for lab in PREFERRED_LAB_ORDER
    if lab in labs_to_keep_set
]

extra_labs = sorted(labs_to_keep_set - set(labs_to_keep))
labs_to_keep.extend(extra_labs)

common_units = {}

for _, row in unit_summary.iterrows():
    lab = row[LAB_COL]
    if lab in labs_to_keep_set:
        common_units[lab] = row["most_common_unit"]

common_units_df = pd.DataFrame([
    {
        LAB_COL: lab,
        "common_unit": common_units.get(lab)
    }
    for lab in labs_to_keep
])

common_units_df.to_csv(OUTPUT_COMMON_UNITS_FILE, index=False)

print("\nInput rows:", total_input_rows)
print("Input unique patients:", len(all_input_patients))

print("\nLabs removed because percent_empty_unit > 30%:")
print(labs_to_remove)

print("\nLabs kept:")
print(labs_to_keep)

print("\nCommon unit per kept lab:")
print(common_units_df)

if len(labs_to_keep) == 0:
    print("\nNo labs remain after filtering. Stopping.")
    sys.exit(0)

print("\nPASS 2: filtering labs, converting units, and selecting patient-level values...")

best_parts = []
filtered_patients = set()
filtered_rows_total = 0

unsupported_conversion_counts = Counter()
missing_unit_assumed_common_counts = Counter()

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

    chunk[LAB_COL] = chunk[LAB_COL].map(normalize_lab_name)
    chunk["_unit_norm"] = chunk[UNIT_COL].map(normalize_unit)

   
    chunk = chunk[
        chunk[LAB_COL].isin(labs_to_keep)
    ].copy()

    if chunk.empty:
        continue

    filtered_rows_total += len(chunk)

    filtered_patients.update(
        chunk[PATIENT_COL].dropna().astype(str).unique()
    )

    chunk[VALUE_COL] = pd.to_numeric(
        chunk[VALUE_COL],
        errors="coerce"
    )

    chunk["common_unit"] = chunk[LAB_COL].map(common_units)

    chunk["lab_unit_was_missing"] = chunk["_unit_norm"].isna().astype(int)

    chunk["_unit_for_conversion"] = chunk["_unit_norm"].where(
        chunk["_unit_norm"].notna(),
        chunk["common_unit"]
    )


    missing_unit_counts_df = (
        chunk.loc[
            chunk["lab_unit_was_missing"] == 1,
            [LAB_COL, "common_unit"]
        ]
        .dropna()
        .value_counts()
        .reset_index(name="count")
    )

    for _, row in missing_unit_counts_df.iterrows():
        lab = row[LAB_COL]
        common_unit = row["common_unit"]
        count = int(row["count"])

        missing_unit_assumed_common_counts[(lab, common_unit)] += count

    chunk["converted_lab_value"] = np.nan
    chunk["conversion_status"] = "not_processed"

    grouped = chunk.groupby(
        [LAB_COL, "_unit_for_conversion", "common_unit"],
        dropna=False
    )

    for (lab, from_unit, to_unit), idx in grouped.groups.items():
        converted_values, success, status = convert_values_to_common_unit(
            chunk.loc[idx, VALUE_COL],
            lab,
            from_unit,
            to_unit
        )

        chunk.loc[idx, "converted_lab_value"] = converted_values.to_numpy()
        chunk.loc[idx, "conversion_status"] = status

        if not success:
            unsupported_conversion_counts[(lab, from_unit, to_unit)] += len(idx)

    if ABS_DIFF_COL in chunk.columns:
        chunk["_abs_diff_sort"] = pd.to_numeric(
            chunk[ABS_DIFF_COL],
            errors="coerce"
        )
    else:
        chunk["_abs_diff_sort"] = 0

    chunk["_abs_diff_sort"] = chunk["_abs_diff_sort"].fillna(999999)

    if LAB_DAY_COL in chunk.columns:
        chunk["_lab_day_sort"] = pd.to_datetime(
            chunk[LAB_DAY_COL],
            errors="coerce"
        )
    else:
        chunk["_lab_day_sort"] = pd.NaT

    chunk_best = reduce_to_best_patient_lab(chunk)

    keep_cols_for_best = [
        PATIENT_COL,
        LAB_COL,
        VALUE_COL,
        UNIT_COL,
        "_unit_norm",
        "_unit_for_conversion",
        "common_unit",
        "lab_unit_was_missing",
        "converted_lab_value",
        "conversion_status",
        "_abs_diff_sort",
        "_lab_day_sort",
        "_row_order"
    ]

    optional_cols = [
        "diagnosis_day",
        "day_difference_lab_minus_dx",
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
        OUTPUT_FILTERED_FILE,
        index=False,
        mode="a",
        header=not wrote_filtered_header
    )
    wrote_filtered_header = True

    if len(best_parts) >= 10:
        combined_best = pd.concat(best_parts, ignore_index=True)
        combined_best = reduce_to_best_patient_lab(combined_best)
        best_parts = [combined_best]

    if chunk_number % 10 == 0:
        print(f"  processed chunks: {chunk_number:,}")


if len(best_parts) == 0:
    print("\nNo filtered lab rows found after removing high-missing-unit labs.")
    sys.exit(0)

best_df = pd.concat(best_parts, ignore_index=True)
best_df = reduce_to_best_patient_lab(best_df)

print("\nRows before removing labs:", total_input_rows)
print("Rows after removing labs:", filtered_rows_total)
print("Unique patients after removing labs:", len(filtered_patients))
print("Patient-lab records after selecting closest lab per patient-lab:", len(best_df))



issue_rows = []

for (lab, from_unit, to_unit), count in unsupported_conversion_counts.items():
    issue_rows.append({
        "issue_type": "unsupported_unit_conversion",
        LAB_COL: lab,
        "from_unit": from_unit,
        "to_common_unit": to_unit,
        "row_count": count
    })

for (lab, common_unit), count in missing_unit_assumed_common_counts.items():
    issue_rows.append({
        "issue_type": "missing_unit_assumed_as_common_unit",
        LAB_COL: lab,
        "from_unit": None,
        "to_common_unit": common_unit,
        "row_count": count
    })

conversion_issues = pd.DataFrame(issue_rows)

if conversion_issues.empty:
    conversion_issues = pd.DataFrame(columns=[
        "issue_type",
        LAB_COL,
        "from_unit",
        "to_common_unit",
        "row_count"
    ])

conversion_issues.to_csv(OUTPUT_CONVERSION_ISSUES_FILE, index=False)

print("\nCreating patient-level wide lab table...")

wide_raw = best_df.pivot(
    index=PATIENT_COL,
    columns=LAB_COL,
    values="converted_lab_value"
)

wide_raw = wide_raw.reindex(columns=labs_to_keep)

wide_raw_output = wide_raw.copy()
wide_raw_output.columns = [
    f"{lab}_value_raw"
    for lab in wide_raw_output.columns
]

wide_raw_output = wide_raw_output.reset_index()
wide_raw_output.to_csv(OUTPUT_WIDE_RAW_FILE, index=False)


print("\nPerforming median imputation...")

lab_medians = wide_raw.median(skipna=True)

wide_imputed = wide_raw.copy()

for lab in labs_to_keep:
    median_value = lab_medians[lab]

    if pd.isna(median_value):
        print(
            f"WARNING: Lab '{lab}' has no valid numeric values after conversion. "
            f"Cannot median-impute this lab."
        )
        continue

    wide_imputed[lab] = wide_imputed[lab].fillna(median_value)


imputation_rows = []

for lab in labs_to_keep:
    missing_before = int(wide_raw[lab].isna().sum())
    nonmissing_before = int(wide_raw[lab].notna().sum())
    missing_after = int(wide_imputed[lab].isna().sum())

    imputation_rows.append({
        LAB_COL: lab,
        "common_unit": common_units.get(lab),
        "patients_total": len(wide_raw),
        "patients_with_value_before_imputation": nonmissing_before,
        "patients_missing_before_imputation": missing_before,
        "median_used_for_imputation": lab_medians[lab],
        "patients_missing_after_imputation": missing_after
    })

imputation_summary = pd.DataFrame(imputation_rows)
imputation_summary.to_csv(OUTPUT_IMPUTATION_SUMMARY_FILE, index=False)


wide_final = pd.DataFrame({
    PATIENT_COL: wide_raw.index.astype(str)
})

for lab in labs_to_keep:
    wide_final[f"{lab}_value"] = wide_imputed[lab].to_numpy()
    wide_final[f"{lab}_unit"] = common_units.get(lab)
    wide_final[f"{lab}_missing_before_imputation"] = (
        wide_raw[lab].isna().astype(int).to_numpy()
    )

wide_final.to_csv(OUTPUT_WIDE_IMPUTED_FILE, index=False)


patients_per_kept_lab = (
    best_df
    .groupby(LAB_COL)[PATIENT_COL]
    .nunique()
    .reset_index(name="unique_patients_with_lab_before_imputation")
    .sort_values("unique_patients_with_lab_before_imputation", ascending=False)
)

print("\nUnique patients per kept lab before imputation:")
print(patients_per_kept_lab)

print("\nDone.")

print("\nSaved empty-unit summary to:")
print(OUTPUT_UNIT_SUMMARY_FILE)

print("\nSaved common units to:")
print(OUTPUT_COMMON_UNITS_FILE)

print("\nSaved filtered converted long-format lab file to:")
print(OUTPUT_FILTERED_FILE)

print("\nSaved wide patient-lab file before imputation to:")
print(OUTPUT_WIDE_RAW_FILE)

print("\nSaved final wide median-imputed patient-lab file to:")
print(OUTPUT_WIDE_IMPUTED_FILE)

print("\nSaved imputation summary to:")
print(OUTPUT_IMPUTATION_SUMMARY_FILE)

print("\nSaved unit conversion issues to:")
print(OUTPUT_CONVERSION_ISSUES_FILE)

print("\nFinal patient-level file shape:")
print(wide_final.shape)

print("\nFinal patient-level columns:")
print(wide_final.columns.tolist())