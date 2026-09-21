import pandas as pd
import ast
import re
from pathlib import Path




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


def find_column(columns, candidates):
   
    col_map = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]

    return None




lab = pd.read_csv(LABS_BY_PATIENT_PATH, encoding="latin1")

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

for lab_name in labs:
    matching_cols = [
        c for c in lab.columns
        if c.lower().startswith(lab_name.lower())
        and ("day" in c.lower() or "days" in c.lower())
    ]

    lab_day_cols[lab_name] = matching_cols

print("\n--- Lab day columns found ---")
for lab_name, cols in lab_day_cols.items():
    print(lab_name, ":", cols)



patient_lab_days = {}

for _, row in lab.iterrows():

    patient_id = row[mrn_col_lab]

    if pd.isna(patient_id):
        continue

    patient_id = str(patient_id)

    patient_lab_days.setdefault(patient_id, [])

    for lab_name, day_cols in lab_day_cols.items():

        for day_col in day_cols:

            days = parse_days_array(row[day_col])

            for lab_day in days:
                patient_lab_days[patient_id].append({
                    "lab_name": lab_name,
                    "lab_day_column": day_col,
                    "lab_day": lab_day
                })

print(f"\nPatients with lab-day information: {len(patient_lab_days):,}")


sample = pd.read_csv(
    COMORBIDITY_PATH,
    encoding="latin1",
    nrows=10,
    low_memory=False
)

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

for chunk in pd.read_csv(
    COMORBIDITY_PATH,
    encoding="latin1",
    usecols=usecols,
    chunksize=CHUNKSIZE,
    low_memory=False
):

    total_comorb_rows += len(chunk)

    chunk[mrn_col_comorb] = chunk[mrn_col_comorb].astype(str)

    chunk["diagnosis_day_numeric"] = pd.to_numeric(
        chunk[diagnosis_day_col],
        errors="coerce"
    )

    chunk = chunk.dropna(subset=["diagnosis_day_numeric"])

    matched_rows = []

    for _, row in chunk.iterrows():

        patient_id = str(row[mrn_col_comorb])
        diagnosis_day = float(row["diagnosis_day_numeric"])

        if patient_id not in patient_lab_days:
            continue

        for lab_info in patient_lab_days[patient_id]:

            lab_day = lab_info["lab_day"]
            day_difference = lab_day - diagnosis_day

            if abs(day_difference) <= WINDOW_DAYS:

                out = {
                    "PAT_MRN_ID_ENCRYPT": patient_id,
                    "diagnosis_day": diagnosis_day,
                    "lab_name": lab_info["lab_name"],
                    "lab_day_column": lab_info["lab_day_column"],
                    "lab_day": lab_day,
                    "day_difference_lab_minus_dx": day_difference,
                    "within_plus_minus_15_days": True
                }

                for col in extra_cols_existing:
                    out[col] = row[col]

                matched_rows.append(out)

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

    print(f"Processed comorbidity rows: {total_comorb_rows:,} | matches so far: {total_matches:,}")


print("\nDONE")
print(f"Total matches found: {total_matches:,}")
print(f"Saved matched rows to: {OUTPUT_PATH}")