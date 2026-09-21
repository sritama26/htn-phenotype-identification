import pandas as pd
import ast
import re
from pathlib import Path
from collections import Counter




COMORBIDITY_PATH = "/projects/f_miarc_1/Hypertension/Hypertension With Comorbidities 20250404.csv"

LABS_BY_PATIENT_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/labs_by_patient.csv"

OUTPUT_PATH = "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/lab_within_15_days_of_diagnosis.csv"

CHUNKSIZE = 500_000
WINDOW_DAYS = 15



labs = [
    "creatinine",
    "gfr",
    "bun",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "glucose",
    "hba1c",
    "cholesterol_total",
    "ldl",
    "hdl",
    "triglycerides"
]



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


def find_related_lab_column(columns, lab_name, kind):
   
    columns = list(columns)
    col_map = {str(c).strip().lower(): c for c in columns}

    lab_lower = lab_name.lower()

    if kind == "value":
        exact_candidates = [
            f"{lab_name}_values",
            f"{lab_name}_value",
            f"{lab_name}_result_values",
            f"{lab_name}_result_value",
            f"{lab_name}_ord_values",
            f"{lab_name}_ord_value",
            f"{lab_name}_lab_values",
            f"{lab_name}_lab_value",
        ]

        keyword_groups = ["value", "result", "ord"]

    elif kind == "unit":
        exact_candidates = [
            f"{lab_name}_units",
            f"{lab_name}_unit",
            f"{lab_name}_reference_units",
            f"{lab_name}_reference_unit",
            f"{lab_name}_lab_units",
            f"{lab_name}_lab_unit",
        ]

        keyword_groups = ["unit"]

    else:
        raise ValueError("kind must be either 'value' or 'unit'")

   
    for cand in exact_candidates:
        key = cand.lower()
        if key in col_map:
            return col_map[key]

    
    broad_candidates = []

    for c in columns:
        c_lower = str(c).strip().lower()

        if not c_lower.startswith(lab_lower):
            continue

        if kind == "value":
            if "unit" in c_lower:
                continue
            if "day" in c_lower or "date" in c_lower:
                continue
            if any(k in c_lower for k in keyword_groups):
                broad_candidates.append(c)

        elif kind == "unit":
            if "unit" in c_lower:
                broad_candidates.append(c)

    if broad_candidates:
       
        broad_candidates = sorted(broad_candidates, key=lambda x: len(str(x)))
        return broad_candidates[0]

    return None


def missing_like(value):
    if value is None:
        return True

    value = str(value).strip().lower()

    return value in {"", "nan", "none", "null", "na", "n/a"}




lab = pd.read_csv(
    LABS_BY_PATIENT_PATH,
    encoding="latin1",
    low_memory=False
)

lab.columns = lab.columns.astype(str).str.strip()

print("\n--- labs_by_patient columns ---")
print(lab.columns.tolist())


mrn_col_lab = find_column(
    lab.columns,
    ["PAT_MRN_ID_ENCRYPT", "MRN", "PATIENT_ID", "patient_id"]
)

if mrn_col_lab is None:
    raise ValueError("Could not find patient ID column in labs_by_patient.csv")

print("\nDetected lab patient ID column:", mrn_col_lab)


lab_day_cols = {}
lab_value_cols = {}
lab_unit_cols = {}

for lab_name in labs:

    matching_day_cols = [
        c for c in lab.columns
        if str(c).lower().startswith(lab_name.lower())
        and ("day" in str(c).lower() or "days" in str(c).lower())
    ]

    value_col = find_related_lab_column(
        columns=lab.columns,
        lab_name=lab_name,
        kind="value"
    )

    unit_col = find_related_lab_column(
        columns=lab.columns,
        lab_name=lab_name,
        kind="unit"
    )

    lab_day_cols[lab_name] = matching_day_cols
    lab_value_cols[lab_name] = value_col
    lab_unit_cols[lab_name] = unit_col


print("\n--- Lab day/value/unit columns found ---")
for lab_name in labs:
    print(f"\n{lab_name}:")
    print("  day columns :", lab_day_cols[lab_name])
    print("  value column:", lab_value_cols[lab_name])
    print("  unit column :", lab_unit_cols[lab_name])


patient_lab_days = {}

patients_with_lab_day_information = set()
patients_with_any_lab_value = set()
patients_with_any_lab_unit = set()

array_length_mismatch_counts = Counter()

for _, row in lab.iterrows():

    patient_id = clean_patient_id(row[mrn_col_lab])

    if patient_id is None:
        continue

    patient_lab_days.setdefault(patient_id, [])

    for lab_name, day_cols in lab_day_cols.items():

        value_col = lab_value_cols.get(lab_name)
        unit_col = lab_unit_cols.get(lab_name)

        values = parse_list_keep_strings(row[value_col]) if value_col is not None else []
        units = parse_list_keep_strings(row[unit_col]) if unit_col is not None else []

        for day_col in day_cols:

            days = parse_days_array(row[day_col])

            if len(days) == 0:
                continue

            patients_with_lab_day_information.add(patient_id)

            if value_col is not None and len(values) not in {0, 1, len(days)}:
                array_length_mismatch_counts[f"{lab_name}_value_length_mismatch"] += 1

            if unit_col is not None and len(units) not in {0, 1, len(days)}:
                array_length_mismatch_counts[f"{lab_name}_unit_length_mismatch"] += 1

            for i, lab_day in enumerate(days):

                lab_value = get_array_value_by_index(values, i)
                lab_unit = get_array_value_by_index(units, i)

                if not missing_like(lab_value):
                    patients_with_any_lab_value.add(patient_id)

                if not missing_like(lab_unit):
                    patients_with_any_lab_unit.add(patient_id)

                patient_lab_days[patient_id].append({
                    "lab_name": lab_name,
                    "lab_day_column": day_col,
                    "lab_day": lab_day,
                    "lab_value_column": value_col,
                    "lab_value": lab_value,
                    "lab_unit_column": unit_col,
                    "lab_unit": lab_unit,
                })


print(f"\nPatients with lab-day information: {len(patients_with_lab_day_information):,}")
print(f"Patients with any lab value found in labs_by_patient: {len(patients_with_any_lab_value):,}")
print(f"Patients with any lab unit found in labs_by_patient: {len(patients_with_any_lab_unit):,}")

if array_length_mismatch_counts:
    print("\nWARNING: Some value/unit arrays do not have same length as day arrays.")
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

        if patient_id not in patient_lab_days:
            continue

        diagnosis_day = float(row["diagnosis_day_numeric"])

        closest_match_by_lab = {}

        for lab_info in patient_lab_days[patient_id]:

            lab_name = lab_info["lab_name"]
            lab_day = lab_info["lab_day"]

            day_difference = lab_day - diagnosis_day
            abs_day_difference = abs(day_difference)

            if abs_day_difference <= WINDOW_DAYS:

                candidate = {
                    "lab_name": lab_name,
                    "lab_day_column": lab_info["lab_day_column"],
                    "lab_day": lab_day,
                    "day_difference_lab_minus_dx": day_difference,
                    "abs_day_difference": abs_day_difference,
                    "lab_value_column": lab_info["lab_value_column"],
                    "lab_value": lab_info["lab_value"],
                    "lab_unit_column": lab_info["lab_unit_column"],
                    "lab_unit": lab_info["lab_unit"],
                }

                if lab_name not in closest_match_by_lab:
                    closest_match_by_lab[lab_name] = candidate

                else:
                    current_best = closest_match_by_lab[lab_name]

                    candidate_sort_key = (
                        candidate["abs_day_difference"],
                        candidate["day_difference_lab_minus_dx"] > 0
                    )

                    current_sort_key = (
                        current_best["abs_day_difference"],
                        current_best["day_difference_lab_minus_dx"] > 0
                    )

                    if candidate_sort_key < current_sort_key:
                        closest_match_by_lab[lab_name] = candidate

        for lab_name, closest in closest_match_by_lab.items():

            lab_value_numeric = pd.to_numeric(
                pd.Series([closest["lab_value"]]),
                errors="coerce"
            ).iloc[0]

            out = {
                "PAT_MRN_ID_ENCRYPT": patient_id,
                "diagnosis_day": diagnosis_day,
                "lab_name": closest["lab_name"],
                "lab_day_column": closest["lab_day_column"],
                "lab_day": closest["lab_day"],
                "day_difference_lab_minus_dx": closest["day_difference_lab_minus_dx"],
                "abs_day_difference": closest["abs_day_difference"],
                "within_plus_minus_15_days": True,

                # New columns
                "lab_value_column": closest["lab_value_column"],
                "lab_value": closest["lab_value"],
                "lab_value_numeric": lab_value_numeric,
                "lab_unit_column": closest["lab_unit_column"],
                "lab_unit": closest["lab_unit"],
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
        f"closest lab matches so far: {total_matches:,} | "
        f"unique patients with matches so far: {len(unique_patients_with_output_matches):,}"
    )



print("\nDONE")
print(f"Total comorbidity rows processed: {total_comorb_rows:,}")
print(f"Total rows with valid diagnosis day: {total_comorb_rows_with_valid_dx_day:,}")
print(f"Total closest lab matches found: {total_matches:,}")
print(f"Unique patients in output file: {len(unique_patients_with_output_matches):,}")
print(f"Saved matched rows to: {OUTPUT_PATH}")

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