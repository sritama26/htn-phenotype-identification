import pandas as pd
import ast
import re
from pathlib import Path
from collections import Counter



COMORBIDITY_PATH = "/projects/f_miarc_1/Hypertension/Hypertension With Comorbidities 20250404.csv"

VITALS_BY_PATIENT_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/vitals_by_patient.csv"

OUTPUT_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/vitals_within_15_days_of_diagnosis.csv"

CHUNKSIZE = 500_000
WINDOW_DAYS = 15


vitals = {
    "height": {
        "days_col": "height_days",
        "values_col": "height_values",
        "unit_col": "height_unit",
    },
    "weight": {
        "days_col": "weight_days",
        "values_col": "weight_values",
        "unit_col": "weight_unit",
    },
    "bmi": {
        "days_col": "bmi_days",
        "values_col": "bmi_values",
        "unit_col": "bmi_unit",
    },
    "bp_systolic": {
        "days_col": "bp_systolic_days",
        "values_col": "bp_systolic_values",
        "unit_col": "bp_systolic_unit",
    },
    "bp_diastolic": {
        "days_col": "bp_diastolic_days",
        "values_col": "bp_diastolic_values",
        "unit_col": "bp_diastolic_unit",
    },
}



def parse_days_array(value):
  
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value in ["", "nan", "None", "NONE", "[]"]:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            out = []
            for x in parsed:
                try:
                    out.append(float(x))
                except Exception:
                    pass
            return out

    except Exception:
        pass

    nums = re.findall(r"-?\d+\.?\d*", value)

    out = []
    for x in nums:
        try:
            out.append(float(x))
        except Exception:
            pass

    return out


def parse_list_keep_strings(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "" or value.lower() in {"nan", "none", "null", "na", "n/a", "[]"}:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                None if pd.isna(x) else str(x).strip()
                for x in parsed
            ]

        return [str(parsed).strip()]

    except Exception:
        pass

    cleaned = value.strip()

    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()

        if cleaned == "":
            return []

        if "," in cleaned:
            return [
                x.strip().strip("'").strip('"')
                for x in cleaned.split(",")
                if x.strip() != ""
            ]

        return [
            x.strip().strip("'").strip('"')
            for x in cleaned.split()
            if x.strip() != ""
        ]

    return [value]


def get_array_value_by_index(values, index):
    
    if values is None or len(values) == 0:
        return None

    if index < len(values):
        return values[index]

    if len(values) == 1:
        return values[0]

    return None


def find_column(columns, candidates):
    
    col_map = {str(c).strip().lower(): c for c in columns}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in col_map:
            return col_map[key]

    return None


def clean_patient_id(value):
    
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "" or value.lower() in {"nan", "none", "null"}:
        return None

    return value


def missing_like(value):
    if value is None:
        return True

    value = str(value).strip().lower()

    return value in {"", "nan", "none", "null", "na", "n/a"}




vital_df = pd.read_csv(
    VITALS_BY_PATIENT_PATH,
    encoding="latin1",
    low_memory=False
)

vital_df.columns = vital_df.columns.astype(str).str.strip()

print("\n--- vitals_by_patient columns ---")
print(vital_df.columns.tolist())


mrn_col_vitals = find_column(
    vital_df.columns,
    ["PAT_MRN_ID_ENCRYPT", "MRN", "PATIENT_ID", "patient_id"]
)

if mrn_col_vitals is None:
    raise ValueError("Could not find patient ID column in vitals_by_patient.csv")

print("\nDetected vitals patient ID column:", mrn_col_vitals)



missing_required_cols = []

for vital_type, cols in vitals.items():
    for role, col_name in cols.items():
        if col_name not in vital_df.columns:
            missing_required_cols.append(col_name)

if missing_required_cols:
    raise ValueError(
        "These required columns are missing from vitals_by_patient.csv:\n"
        + str(missing_required_cols)
    )

print("\n--- Vital day/value/unit columns found ---")
for vital_type, cols in vitals.items():
    print(f"\n{vital_type}:")
    print("  day column  :", cols["days_col"])
    print("  value column:", cols["values_col"])
    print("  unit column :", cols["unit_col"])




patient_vital_days = {}

patients_with_vital_day_information = set()
patients_with_any_vital_value = set()
patients_with_any_vital_unit = set()

array_length_mismatch_counts = Counter()

for _, row in vital_df.iterrows():

    patient_id = clean_patient_id(row[mrn_col_vitals])

    if patient_id is None:
        continue

    patient_vital_days.setdefault(patient_id, [])

    for vital_type, cols in vitals.items():

        days_col = cols["days_col"]
        values_col = cols["values_col"]
        unit_col = cols["unit_col"]

        days = parse_days_array(row[days_col])
        values = parse_list_keep_strings(row[values_col])
        units = parse_list_keep_strings(row[unit_col])

        if len(days) == 0:
            continue

        patients_with_vital_day_information.add(patient_id)

        if len(values) not in {0, 1, len(days)}:
            array_length_mismatch_counts[f"{vital_type}_value_length_mismatch"] += 1

        if len(units) not in {0, 1, len(days)}:
            array_length_mismatch_counts[f"{vital_type}_unit_length_mismatch"] += 1

        for i, vital_day in enumerate(days):

            vital_value = get_array_value_by_index(values, i)
            vital_unit = get_array_value_by_index(units, i)

            if not missing_like(vital_value):
                patients_with_any_vital_value.add(patient_id)

            if not missing_like(vital_unit):
                patients_with_any_vital_unit.add(patient_id)

            patient_vital_days[patient_id].append({
                "vital_type": vital_type,
                "vital_day_column": days_col,
                "vital_day": vital_day,
                "vital_value_column": values_col,
                "vital_value": vital_value,
                "vital_unit_column": unit_col,
                "vital_unit": vital_unit,
            })


print(f"\nPatients with vital-day information: {len(patients_with_vital_day_information):,}")
print(f"Patients with any vital value found in vitals_by_patient: {len(patients_with_any_vital_value):,}")
print(f"Patients with any vital unit found in vitals_by_patient: {len(patients_with_any_vital_unit):,}")

if array_length_mismatch_counts:
    print("\nWARNING: Some value/unit arrays do not have the same length as day arrays.")
    print("When this happens, the code uses value/unit by same index if possible.")
    print("If only one value/unit is present, it reuses that value/unit for all days.")
    print(array_length_mismatch_counts)


sample = pd.read_csv(
    COMORBIDITY_PATH,
    encoding="latin1",
    nrows=10,
    low_memory=False
)

sample.columns = sample.columns.astype(str).str.strip()

print("\n--- Comorbidity columns ---")
print(sample.columns.tolist())


mrn_col_comorb = find_column(
    sample.columns,
    ["PAT_MRN_ID_ENCRYPT", "MRN", "PATIENT_ID", "patient_id"]
)

diagnosis_day_col = find_column(
    sample.columns,
    [
        "DAYS_SINCE_DX",
        "Days_Since_Dx",
        "DAYS_SINCE_FIRST_DX",
        "DAYS_FROM_DX",
        "DAYS_FROM_DIAGNOSIS",
        "DAY_OF_DIAGNOSIS",
        "DIAGNOSIS_DAY",
        "DX_DAY",
        "day_of_diagnosis",
        "DAYS_SINCE_DX_FRST_OBSRVD"
    ]
)

if mrn_col_comorb is None:
    raise ValueError("Could not find patient ID column in comorbidity file.")

if diagnosis_day_col is None:
    raise ValueError(
        "Could not find diagnosis day column in comorbidity file. "
        "Print the comorbidity columns above and replace diagnosis_day_col manually."
    )

print("\nDetected comorbidity patient ID column:", mrn_col_comorb)
print("Detected diagnosis day column:", diagnosis_day_col)


extra_cols_to_keep = [
    "CURRENT_ICD10_LIST",
    "DX_NAME",
    "PATIENT_AGE",
    "SEX",
    "RACE",
    "ETHNICITY",
    "LANGUAGE"
]

extra_cols_existing = [
    c for c in extra_cols_to_keep
    if c in sample.columns
]

usecols = [mrn_col_comorb, diagnosis_day_col] + extra_cols_existing
usecols = list(dict.fromkeys(usecols))



output_file = Path(OUTPUT_PATH)

if output_file.exists():
    output_file.unlink()

first_write = True

total_matches = 0
total_comorb_rows = 0
total_comorb_rows_with_valid_dx_day = 0

unique_patients_with_output_matches = set()

for chunk in pd.read_csv(
    COMORBIDITY_PATH,
    encoding="latin1",
    usecols=usecols,
    chunksize=CHUNKSIZE,
    low_memory=False
):

    chunk.columns = chunk.columns.astype(str).str.strip()

    total_comorb_rows += len(chunk)

    chunk["diagnosis_day_numeric"] = pd.to_numeric(
        chunk[diagnosis_day_col],
        errors="coerce"
    )

    chunk = chunk.dropna(subset=["diagnosis_day_numeric"])
    total_comorb_rows_with_valid_dx_day += len(chunk)

    matched_rows = []

    for _, row in chunk.iterrows():

        patient_id = clean_patient_id(row[mrn_col_comorb])

        if patient_id is None:
            continue

        if patient_id not in patient_vital_days:
            continue

        diagnosis_day = float(row["diagnosis_day_numeric"])

       
        closest_match_by_vital = {}

        for vital_info in patient_vital_days[patient_id]:

            vital_type = vital_info["vital_type"]
            vital_day = vital_info["vital_day"]

            day_difference = vital_day - diagnosis_day
            abs_day_difference = abs(day_difference)

            if abs_day_difference <= WINDOW_DAYS:

                candidate = {
                    "vital_type": vital_type,
                    "vital_day_column": vital_info["vital_day_column"],
                    "vital_day": vital_day,
                    "day_difference_vital_minus_dx": day_difference,
                    "abs_day_difference": abs_day_difference,
                    "vital_value_column": vital_info["vital_value_column"],
                    "vital_value": vital_info["vital_value"],
                    "vital_unit_column": vital_info["vital_unit_column"],
                    "vital_unit": vital_info["vital_unit"],
                }

                if vital_type not in closest_match_by_vital:
                    closest_match_by_vital[vital_type] = candidate

                else:
                    current_best = closest_match_by_vital[vital_type]

                    candidate_sort_key = (
                        candidate["abs_day_difference"],
                        candidate["day_difference_vital_minus_dx"] > 0
                    )

                    current_sort_key = (
                        current_best["abs_day_difference"],
                        current_best["day_difference_vital_minus_dx"] > 0
                    )

                    if candidate_sort_key < current_sort_key:
                        closest_match_by_vital[vital_type] = candidate


        for vital_type, closest in closest_match_by_vital.items():

            vital_value_numeric = pd.to_numeric(
                pd.Series([closest["vital_value"]]),
                errors="coerce"
            ).iloc[0]

            out = {
                "PAT_MRN_ID_ENCRYPT": patient_id,
                "diagnosis_day": diagnosis_day,
                "vital_type": closest["vital_type"],
                "vital_day_column": closest["vital_day_column"],
                "vital_day": closest["vital_day"],
                "day_difference_vital_minus_dx": closest["day_difference_vital_minus_dx"],
                "abs_day_difference": closest["abs_day_difference"],
                "within_plus_minus_15_days": True,

                "vital_value_column": closest["vital_value_column"],
                "vital_value": closest["vital_value"],
                "vital_value_numeric": vital_value_numeric,
                "vital_unit_column": closest["vital_unit_column"],
                "vital_unit": closest["vital_unit"],
            }

            for col in extra_cols_existing:
                out[col] = row[col]

            matched_rows.append(out)
            unique_patients_with_output_matches.add(patient_id)

    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)

        matched_df.to_csv(
            output_file,
            mode="a",
            header=first_write,
            index=False
        )

        first_write = False
        total_matches += len(matched_df)

    print(
        f"Processed comorbidity rows: {total_comorb_rows:,} | "
        f"valid diagnosis-day rows: {total_comorb_rows_with_valid_dx_day:,} | "
        f"closest vital matches so far: {total_matches:,} | "
        f"unique patients with matches so far: {len(unique_patients_with_output_matches):,}"
    )


print("\nDONE")
print(f"Total comorbidity rows processed: {total_comorb_rows:,}")
print(f"Total rows with valid diagnosis day: {total_comorb_rows_with_valid_dx_day:,}")
print(f"Total closest vital matches found: {total_matches:,}")
print(f"Unique patients in output file: {len(unique_patients_with_output_matches):,}")
print(f"Saved matched rows to: {OUTPUT_PATH}")

# Optional verification by reading output file patient column
if output_file.exists():
    output_unique_patients_check = set()

    for out_chunk in pd.read_csv(
        output_file,
        encoding="latin1",
        usecols=["PAT_MRN_ID_ENCRYPT"],
        chunksize=CHUNKSIZE,
        low_memory=False
    ):
        output_unique_patients_check.update(
            out_chunk["PAT_MRN_ID_ENCRYPT"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

    print(f"Verified unique patients by reading output file: {len(output_unique_patients_check):,}")
else:
    print("WARNING: Output file was not created because no matches were found.")