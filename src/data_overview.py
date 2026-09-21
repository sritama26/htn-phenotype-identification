#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from tqdm import tqdm


data_dir = Path("/projects/f_miarc_1/Hypertension")

output_dir = Path("/projects/f_miarc_1/Hypertension/Sritama/data_overview")
output_dir.mkdir(exist_ok=True)

parquet_dir = output_dir / "parquet_files"
parquet_dir.mkdir(exist_ok=True)

CSV_CHUNKSIZE = 500_000      
BATCH_ROWS = 100_000         
PREVIEW_ROWS = 5

FORCE_RECONVERT = False      


MAX_AGE_ZERO_ROWS_TO_PRINT = 100




files = {
    "comorbidities": {
        "csv": data_dir / "Hypertension With Comorbidities 20250404.csv",
        "parquet": parquet_dir / "Comorbidities.parquet",
        "sep": ",",
    },
    "labs": {
        "csv": data_dir / "Hypertension Labs 20250417.csv",
        "parquet": parquet_dir / "Hypertension_Labs_20250417.parquet",
        "sep": "|",
    },
    "meds": {
        "csv": data_dir / "Hypertension Medications 20250417.csv",
        "parquet": parquet_dir / "Hypertension_Medications_20250417.parquet",
        "sep": ",",
    },
    "vitals": {
        "csv": data_dir / "Hypertension Vitals 20250520.csv",
        "parquet": parquet_dir / "Hypertension Vitals 20250520.parquet",
        "sep": ",",
    },
}



def convert_csv_to_parquet(csv_path, parquet_path, sep=",", chunksize=500_000):
    """
    Convert large CSV to Parquet in chunks.
    Forces all columns to string type so schema does not change across chunks.
    """

    if parquet_path.exists() and not FORCE_RECONVERT:
        print(f"Parquet already exists, skipping conversion: {parquet_path}")
        return

    if not csv_path.exists():
        print(f"WARNING: CSV file not found, skipping: {csv_path}")
        return

   
    if parquet_path.exists() and FORCE_RECONVERT:
        print(f"Removing old parquet file: {parquet_path}")
        parquet_path.unlink()

    print(f"\nConverting CSV to Parquet:")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")

    reader = pd.read_csv(
        csv_path,
        sep=sep,
        quotechar='"',
        encoding="latin1",
        dtype=str,
        chunksize=chunksize,
        engine="c",
        low_memory=False
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
            schema = pa.schema([
                pa.field(col, pa.string()) for col in chunk.columns
            ])

        
        table = pa.Table.from_pandas(
            chunk,
            schema=schema,
            preserve_index=False
        )

        if writer is None:
            writer = pq.ParquetWriter(
                parquet_path,
                schema,
                compression="snappy"
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()

    print(f"Finished converting {csv_path.name}")
    print(f"Total rows written: {total_rows:,}")


def show_parquet_head_and_columns(parquet_path, name):
    
    if not parquet_path.exists():
        print(f"WARNING: Parquet file not found: {parquet_path}")
        return

    pf = pq.ParquetFile(parquet_path)

    print(f"\n--- {name.upper()} Shape from Parquet Metadata ---")
    print(f"Rows: {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.metadata.num_columns:,}")

    print(f"\n--- {name.upper()} Column Names List ---")
    print(pf.schema_arrow.names)

    print(f"\n--- {name.upper()} Head ---")
    first_batch = next(
        pf.iter_batches(
            batch_size=PREVIEW_ROWS,
            use_threads=True
        )
    )

    preview_df = first_batch.to_pandas()
    print(preview_df.head(PREVIEW_ROWS).to_string())


def get_unique_mrn_count_from_parquet(parquet_path, name):
    

    if not parquet_path.exists():
        print(f"WARNING: Parquet file not found: {parquet_path}")
        return 0

    pf = pq.ParquetFile(parquet_path)
    cols = pf.schema_arrow.names

    if "PAT_MRN_ID_ENCRYPT" not in cols:
        print(f"WARNING: PAT_MRN_ID_ENCRYPT not found in {name}")
        return 0

    unique_mrn = set()
    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=["PAT_MRN_ID_ENCRYPT"],
            use_threads=True
        ),
        desc=f"Counting unique MRNs in {name}"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        unique_mrn.update(
            chunk["PAT_MRN_ID_ENCRYPT"]
            .dropna()
            .unique()
        )

    print(f"\n{name} rows processed: {total_rows:,}")
    print(f"{name} unique MRNs: {len(unique_mrn):,}")

    return len(unique_mrn)


def weighted_median_from_counter(counter):
   

    total = sum(counter.values())

    if total == 0:
        return None

    midpoint1 = (total - 1) // 2
    midpoint2 = total // 2

    cumulative = 0
    value1 = None
    value2 = None

    for value in sorted(counter):
        cumulative += counter[value]

        if value1 is None and cumulative > midpoint1:
            value1 = value

        if value2 is None and cumulative > midpoint2:
            value2 = value
            break

    return (value1 + value2) / 2



MISSING_STRING_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def missing_value_mask(series):
    
    as_string = series.astype("string")
    stripped_lower = as_string.str.strip().str.lower()

    return as_string.isna() | stripped_lower.isin(MISSING_STRING_VALUES)


def null_blank_missing_counts(series):
    
    as_string = series.astype("string")
    stripped_lower = as_string.str.strip().str.lower()

    null_count = int(as_string.isna().sum())
    blank_count = int((stripped_lower == "").sum())
    text_missing_count = int(
        stripped_lower.isin(MISSING_STRING_VALUES - {""}).sum()
    )
    missing_count = int(missing_value_mask(series).sum())

    return {
        "null_count": null_count,
        "blank_count": blank_count,
        "text_missing_count": text_missing_count,
        "missing_count": missing_count,
    }


def update_missing_counter(missing_counter, column_name, series):
    
    counts = null_blank_missing_counts(series)

    for key, value in counts.items():
        missing_counter[column_name][key] += value


def update_category_counter(counter, series):
   
    valid = series.loc[~missing_value_mask(series)].astype("string").str.strip()
    counter.update(valid.tolist())


def counter_to_sorted_dict(counter):
    

    return {
        str(key): int(value)
        for key, value in sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0]))
        )
    }


def update_category_by_sex_counter(nested_counter, category_series, sex_series):
    

    category_missing = missing_value_mask(category_series)
    valid_category = category_series.loc[~category_missing].astype("string").str.strip()

    sex_for_valid_rows = sex_series.loc[valid_category.index]
    sex_missing = missing_value_mask(sex_for_valid_rows)
    sex_clean = sex_for_valid_rows.astype("string").str.strip()
    sex_clean = sex_clean.mask(sex_missing, "Missing")

    for category_value, sex_value in zip(valid_category.tolist(), sex_clean.tolist()):
        nested_counter[str(category_value)][str(sex_value)] += 1


def nested_counter_to_sorted_dict(nested_counter):
    

    return {
        str(category): counter_to_sorted_dict(counter)
        for category, counter in sorted(
            nested_counter.items(),
            key=lambda item: (-sum(item[1].values()), str(item[0]))
        )
    }




def analyze_parquet_missing_counts(parquet_path, dataset_name):
   

    if not parquet_path.exists():
        print(f"WARNING: {dataset_name} Parquet file not found: {parquet_path}")
        return {}

    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names

    missing_counts = {
        col: Counter({
            "null_count": 0,
            "blank_count": 0,
            "text_missing_count": 0,
            "missing_count": 0,
        })
        for col in columns
    }

    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=columns,
            use_threads=True
        ),
        desc=f"Counting missing values in {dataset_name}"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        for col in columns:
            update_missing_counter(missing_counts, col, chunk[col])

    result = {}

    for col in columns:
        counts = missing_counts[col]
        missing_count = int(counts.get("missing_count", 0))
        non_missing_count = int(total_rows - missing_count)
        missing_percent = (missing_count / total_rows * 100) if total_rows > 0 else 0.0

        result[col] = {
            "total_rows": int(total_rows),
            "null_count": int(counts.get("null_count", 0)),
            "blank_count": int(counts.get("blank_count", 0)),
            "text_missing_count": int(counts.get("text_missing_count", 0)),
            "missing_count": missing_count,
            "non_missing_count": non_missing_count,
            "missing_percent": round(float(missing_percent), 4),
        }

    missing_df = pd.DataFrame.from_dict(result, orient="index").reset_index()
    missing_df = missing_df.rename(columns={"index": "column"})
    missing_df = missing_df.sort_values(
        by=["missing_count", "column"],
        ascending=[False, True]
    )

    output_path = output_dir / f"{dataset_name}_missing_counts.csv"
    missing_df.to_csv(output_path, index=False)

    print(f"\n--- {dataset_name.upper()} MISSING VALUE COUNTS ---")
    print(missing_df.to_string(index=False))
    print(f"{dataset_name} missing counts saved to: {output_path}")

    return result


def find_first_existing_column(columns, candidates):
    """
    Find first candidate column using case-insensitive exact matching.
    """

    col_map = {str(c).strip().lower(): c for c in columns}

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in col_map:
            return col_map[key]

    return None


def analyze_vitals_measurement_missing_counts(parquet_path):

    if not parquet_path.exists():
        print(f"WARNING: Vitals Parquet file not found: {parquet_path}")
        return {}

    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names

    vital_name_col = find_first_existing_column(
        columns,
        [
            "FLO_MEAS_NAME",
            "MEAS_NAME",
            "VITAL_NAME",
            "VITAL_SIGN",
            "COMPONENT_NAME",
            "DISPLAY_NAME",
            "FLO_MEAS_ID_DISP_NAME",
        ]
    )

    vital_value_col = find_first_existing_column(
        columns,
        [
            "MEAS_VALUE",
            "MEAS_VALUE_NUM",
            "VALUE",
            "RESULT_VALUE",
            "VITAL_VALUE",
            "OBS_VALUE",
        ]
    )

    if vital_name_col is None:
        print("WARNING: Could not detect the vitals name column, such as FLO_MEAS_NAME.")
        print("Available vitals columns:")
        print(columns)
        return {}

    if vital_value_col is None:
        print("WARNING: Could not detect the vitals value column, such as MEAS_VALUE.")
        print("Available vitals columns:")
        print(columns)
        return {}

    vital_patterns = {
        "bmi": {
            "include": ["BMI", "BODY MASS INDEX"],
            "exclude": [],
        },
        "weight": {
            "include": ["WEIGHT", " WT"],
            "exclude": [],
        },
        "bp": {
            "include": ["BLOOD PRESSURE", " BP", "BP ", "SYSTOLIC", "DIASTOLIC"],
            "exclude": [],
        },
        "height": {
            "include": ["HEIGHT", " HT"],
            "exclude": [],
        },
    }

    counters = {
        vital_type: Counter({
            "total_rows": 0,
            "null_count": 0,
            "blank_count": 0,
            "text_missing_count": 0,
            "missing_count": 0,
        })
        for vital_type in vital_patterns
    }

    matched_names = {vital_type: Counter() for vital_type in vital_patterns}

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=[vital_name_col, vital_value_col],
            use_threads=True
        ),
        desc="Counting BMI/weight/BP/height missing values in vitals"
    ):
        chunk = batch.to_pandas()

        names = chunk[vital_name_col].astype("string").str.upper().str.strip()

        for vital_type, pattern_info in vital_patterns.items():
            mask = pd.Series(False, index=chunk.index)

            for term in pattern_info["include"]:
                mask = mask | names.str.contains(term, na=False, regex=False)

            for term in pattern_info["exclude"]:
                mask = mask & ~names.str.contains(term, na=False, regex=False)

            matched = chunk.loc[mask]
            counters[vital_type]["total_rows"] += int(len(matched))

            if len(matched) == 0:
                continue

            counts = null_blank_missing_counts(matched[vital_value_col])
            for key, value in counts.items():
                counters[vital_type][key] += int(value)

            update_category_counter(matched_names[vital_type], matched[vital_name_col])

    result = {}

    for vital_type, counts in counters.items():
        total_rows = int(counts.get("total_rows", 0))
        missing_count = int(counts.get("missing_count", 0))
        non_missing_count = int(total_rows - missing_count)
        missing_percent = (missing_count / total_rows * 100) if total_rows > 0 else 0.0

        result[vital_type] = {
            "vital_name_column": vital_name_col,
            "vital_value_column": vital_value_col,
            "matched_rows": total_rows,
            "null_count": int(counts.get("null_count", 0)),
            "blank_count": int(counts.get("blank_count", 0)),
            "text_missing_count": int(counts.get("text_missing_count", 0)),
            "missing_count": missing_count,
            "non_missing_count": non_missing_count,
            "missing_percent": round(float(missing_percent), 4),
            "matched_vital_names": counter_to_sorted_dict(matched_names[vital_type]),
        }

    rows = []
    for vital_type, info in result.items():
        rows.append({
            "vital_type": vital_type,
            "vital_name_column": info["vital_name_column"],
            "vital_value_column": info["vital_value_column"],
            "matched_rows": info["matched_rows"],
            "null_count": info["null_count"],
            "blank_count": info["blank_count"],
            "text_missing_count": info["text_missing_count"],
            "missing_count": info["missing_count"],
            "non_missing_count": info["non_missing_count"],
            "missing_percent": info["missing_percent"],
        })

    output_path = output_dir / "vitals_bmi_weight_bp_height_missing_counts.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print("\n--- VITALS BMI / WEIGHT / BP / HEIGHT MISSING VALUE COUNTS ---")
    print(json.dumps(result, indent=4))
    print(f"Vitals BMI/weight/BP/height missing counts saved to: {output_path}")

    return result


def analyze_comorbidities_parquet(parquet_path):
    """
    Analyze Comorbidities.parquet in batches.
    Adds missing/null counts for age, race, ethnicity, language;
    category counts for race, ethnicity, language; and rows where age = 0.
    """

    if not parquet_path.exists():
        print(f"WARNING: Comorbidities Parquet file not found: {parquet_path}")
        return {}

    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names

    print("\n--- COMORBIDITIES Shape from Parquet Metadata ---")
    print(f"Rows: {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.metadata.num_columns:,}")

    needed_cols = [
        "PAT_MRN_ID_ENCRYPT",
        "CURRENT_ICD10_LIST",
        "RACE",
        "ETHNICITY",
        "LANGUAGE",
        "SEX",
        "DX_NAME",
        "PATIENT_AGE",
        "DISCHARGE_DISPOSITION"
    ]

    existing_cols = [c for c in needed_cols if c in all_cols]
    missing_cols = [c for c in needed_cols if c not in all_cols]

    if missing_cols:
        print("\nWARNING: These expected columns are missing from Comorbidities:")
        print(missing_cols)

    unique_mrn = set()
    unique_icd10 = set()
    unique_dx = set()

    race_values = set()
    ethnicity_values = set()
    language_values = set()
    sex_values = set()

    race_counts = Counter()
    ethnicity_counts = Counter()
    language_counts = Counter()
    sex_counts = Counter()
    discharge_counts = Counter()


    race_sex_counts = defaultdict(Counter)
    ethnicity_sex_counts = defaultdict(Counter)
    language_sex_counts = defaultdict(Counter)

    icd10_nulls = 0
    icd10_samples = []

    age_min = None
    age_max = None
    age_sum = 0.0
    age_count = 0
    age_hist = Counter()

   
    demographic_missing_counts = {
        "PATIENT_AGE": Counter(),
        "RACE": Counter(),
        "ETHNICITY": Counter(),
        "LANGUAGE": Counter(),
    }
    patient_age_non_numeric_count = 0
    patient_age_zero_count = 0
    age_zero_rows_printed = 0

    age_zero_rows_path = output_dir / "patient_age_zero_rows.csv"
    age_zero_rows_file_written = False

    
    if age_zero_rows_path.exists():
        age_zero_rows_path.unlink()

    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=existing_cols,
            use_threads=True
        ),
        desc="Analyzing Comorbidities"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        if "PAT_MRN_ID_ENCRYPT" in chunk.columns:
            unique_mrn.update(
                chunk["PAT_MRN_ID_ENCRYPT"]
                .dropna()
                .unique()
            )

        if "CURRENT_ICD10_LIST" in chunk.columns:
            icd10_nulls += int(chunk["CURRENT_ICD10_LIST"].isna().sum())
            unique_icd10.update(
                chunk["CURRENT_ICD10_LIST"]
                .dropna()
                .unique()
            )

            if len(icd10_samples) < 10:
                needed = 10 - len(icd10_samples)
                icd10_samples.extend(
                    chunk["CURRENT_ICD10_LIST"]
                    .dropna()
                    .head(needed)
                    .tolist()
                )

        if "DX_NAME" in chunk.columns:
            unique_dx.update(
                chunk["DX_NAME"]
                .dropna()
                .unique()
            )

        if "RACE" in chunk.columns:
            update_missing_counter(demographic_missing_counts, "RACE", chunk["RACE"])
            update_category_counter(race_counts, chunk["RACE"])
            race_values.update(race_counts.keys())

            if "SEX" in chunk.columns:
                update_category_by_sex_counter(
                    race_sex_counts,
                    chunk["RACE"],
                    chunk["SEX"]
                )

        if "ETHNICITY" in chunk.columns:
            update_missing_counter(demographic_missing_counts, "ETHNICITY", chunk["ETHNICITY"])
            update_category_counter(ethnicity_counts, chunk["ETHNICITY"])
            ethnicity_values.update(ethnicity_counts.keys())

            if "SEX" in chunk.columns:
                update_category_by_sex_counter(
                    ethnicity_sex_counts,
                    chunk["ETHNICITY"],
                    chunk["SEX"]
                )

        if "LANGUAGE" in chunk.columns:
            update_missing_counter(demographic_missing_counts, "LANGUAGE", chunk["LANGUAGE"])
            update_category_counter(language_counts, chunk["LANGUAGE"])
            language_values.update(language_counts.keys())

            if "SEX" in chunk.columns:
                update_category_by_sex_counter(
                    language_sex_counts,
                    chunk["LANGUAGE"],
                    chunk["SEX"]
                )

        if "SEX" in chunk.columns:
            update_category_counter(sex_counts, chunk["SEX"])
            sex_values.update(sex_counts.keys())

        if "DISCHARGE_DISPOSITION" in chunk.columns:
            update_category_counter(discharge_counts, chunk["DISCHARGE_DISPOSITION"])

        if "PATIENT_AGE" in chunk.columns:
            age_raw = chunk["PATIENT_AGE"]
            update_missing_counter(demographic_missing_counts, "PATIENT_AGE", age_raw)

            age_missing_mask = missing_value_mask(age_raw)
            age = pd.to_numeric(age_raw, errors="coerce")

           
            patient_age_non_numeric_count += int(age.isna().sum() - age_missing_mask.sum())

            valid_age = age.dropna()

            if not valid_age.empty:
                chunk_min = valid_age.min()
                chunk_max = valid_age.max()

                age_min = chunk_min if age_min is None else min(age_min, chunk_min)
                age_max = chunk_max if age_max is None else max(age_max, chunk_max)

                age_sum += valid_age.sum()
                age_count += valid_age.count()

                age_hist.update(
                    valid_age.round().astype(int).tolist()
                )

         
            age_zero_mask = age.eq(0)
            age_zero_rows = chunk.loc[age_zero_mask].copy()
            patient_age_zero_count += int(age_zero_mask.sum())

            if not age_zero_rows.empty:
                
                age_zero_rows.to_csv(
                    age_zero_rows_path,
                    mode="a",
                    header=not age_zero_rows_file_written,
                    index=False
                )
                age_zero_rows_file_written = True

                
                if MAX_AGE_ZERO_ROWS_TO_PRINT is None:
                    print("\n--- ROWS WHERE PATIENT_AGE = 0 ---")
                    print(age_zero_rows.to_string(index=False))
                elif age_zero_rows_printed < MAX_AGE_ZERO_ROWS_TO_PRINT:
                    remaining_to_print = MAX_AGE_ZERO_ROWS_TO_PRINT - age_zero_rows_printed
                    rows_to_print = age_zero_rows.head(remaining_to_print)

                    print("\n--- ROWS WHERE PATIENT_AGE = 0 ---")
                    print(rows_to_print.to_string(index=False))

                    age_zero_rows_printed += len(rows_to_print)

    age_mean = age_sum / age_count if age_count > 0 else None
    age_median = weighted_median_from_counter(age_hist)

   
    demographic_missing_counts_json = {
        col: {
            "null_count": int(counts.get("null_count", 0)),
            "blank_count": int(counts.get("blank_count", 0)),
            "text_missing_count": int(counts.get("text_missing_count", 0)),
            "missing_count": int(counts.get("missing_count", 0)),
        }
        for col, counts in demographic_missing_counts.items()
    }

    race_value_counts = counter_to_sorted_dict(race_counts)
    ethnicity_value_counts = counter_to_sorted_dict(ethnicity_counts)
    language_value_counts = counter_to_sorted_dict(language_counts)
    sex_value_counts = counter_to_sorted_dict(sex_counts)

    race_sex_counts_json = nested_counter_to_sorted_dict(race_sex_counts)
    ethnicity_sex_counts_json = nested_counter_to_sorted_dict(ethnicity_sex_counts)
    language_sex_counts_json = nested_counter_to_sorted_dict(language_sex_counts)

    summary = {
        "comorbidities_rows_processed": int(total_rows),
        "comorbidities_unique_patients": int(len(unique_mrn)),
        "current_icd10_nulls": int(icd10_nulls),
        "current_icd10_sample_values": icd10_samples,
        "unique_icd10_count": int(len(unique_icd10)),
        "unique_dx_name_count": int(len(unique_dx)),

       
        "race_count": int(len(race_values)),
        "ethnicity_count": int(len(ethnicity_values)),
        "language_count": int(len(language_values)),
        "sex_values": sorted(list(sex_values)),
        "race_values": sorted(list(race_values)),
        "ethnicity_values": sorted(list(ethnicity_values)),
        "language_values": sorted(list(language_values)),

       
        "age_min": None if age_min is None else float(age_min),
        "age_max": None if age_max is None else float(age_max),
        "age_median": age_median,
        "age_mean": None if age_mean is None else float(age_mean),

        
        "demographic_missing_counts": demographic_missing_counts_json,
        "patient_age_null_count": demographic_missing_counts_json["PATIENT_AGE"]["null_count"],
        "patient_age_missing_count": demographic_missing_counts_json["PATIENT_AGE"]["missing_count"],
        "patient_age_non_numeric_count": int(patient_age_non_numeric_count),
        "patient_age_zero_count": int(patient_age_zero_count),
        "race_null_count": demographic_missing_counts_json["RACE"]["null_count"],
        "race_missing_count": demographic_missing_counts_json["RACE"]["missing_count"],
        "ethnicity_null_count": demographic_missing_counts_json["ETHNICITY"]["null_count"],
        "ethnicity_missing_count": demographic_missing_counts_json["ETHNICITY"]["missing_count"],
        "language_null_count": demographic_missing_counts_json["LANGUAGE"]["null_count"],
        "language_missing_count": demographic_missing_counts_json["LANGUAGE"]["missing_count"],

        
        "race_value_counts": race_value_counts,
        "ethnicity_value_counts": ethnicity_value_counts,
        "language_value_counts": language_value_counts,
        "sex_value_counts": sex_value_counts,

        "race_sex_counts": race_sex_counts_json,
        "ethnicity_sex_counts": ethnicity_sex_counts_json,
        "language_sex_counts": language_sex_counts_json,

        
        "patient_age_zero_rows_csv": str(age_zero_rows_path) if age_zero_rows_file_written else None,
    }

    print("\n--- COMORBIDITIES SUMMARY ---")
    print(json.dumps(summary, indent=4))

    print("\n--- RACE VALUE COUNTS ---")
    print(json.dumps(race_value_counts, indent=4))

    print("\n--- ETHNICITY VALUE COUNTS ---")
    print(json.dumps(ethnicity_value_counts, indent=4))

    print("\n--- LANGUAGE VALUE COUNTS ---")
    print(json.dumps(language_value_counts, indent=4))

    print("\n--- SEX VALUE COUNTS ---")
    print(json.dumps(sex_value_counts, indent=4))

    print("\n--- SEX COUNTS WITHIN EACH RACE CATEGORY ---")
    print(json.dumps(race_sex_counts_json, indent=4))

    print("\n--- SEX COUNTS WITHIN EACH ETHNICITY CATEGORY ---")
    print(json.dumps(ethnicity_sex_counts_json, indent=4))

    print("\n--- SEX COUNTS WITHIN EACH LANGUAGE CATEGORY ---")
    print(json.dumps(language_sex_counts_json, indent=4))

    print(f"\nRows where PATIENT_AGE = 0: {patient_age_zero_count:,}")
    if age_zero_rows_file_written:
        print(f"All PATIENT_AGE = 0 rows saved to: {age_zero_rows_path}")
    else:
        print("No PATIENT_AGE = 0 rows found.")

    summary_path = output_dir / "overview_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSummary saved to: {summary_path}")

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    pd.Series(sex_counts).sort_values(ascending=False).plot(
        kind="bar",
        ax=axes[0, 0],
        color="steelblue"
    )
    axes[0, 0].set_title("Sex Distribution")
    axes[0, 0].set_xlabel("")

    pd.Series(race_counts).sort_values(ascending=False).head(10).sort_values().plot(
        kind="barh",
        ax=axes[0, 1],
        color="steelblue"
    )
    axes[0, 1].set_title("Top 10 Race Categories")

    pd.Series(ethnicity_counts).sort_values(ascending=False).sort_values().plot(
        kind="barh",
        ax=axes[1, 0],
        color="steelblue"
    )
    axes[1, 0].set_title("Ethnicity Distribution")

    pd.Series(age_hist).sort_index().plot(
        kind="bar",
        ax=axes[1, 1],
        color="steelblue"
    )
    axes[1, 1].set_title("Age Distribution")
    axes[1, 1].set_xlabel("Age")

    pd.Series(language_counts).sort_values(ascending=False).head(10).sort_values().plot(
        kind="barh",
        ax=axes[2, 0],
        color="steelblue"
    )
    axes[2, 0].set_title("Top 10 Languages")

    pd.Series(discharge_counts).sort_values(ascending=False).head(10).sort_values().plot(
        kind="barh",
        ax=axes[2, 1],
        color="steelblue"
    )
    axes[2, 1].set_title("Discharge Disposition")

    plt.tight_layout()

    plot_path = output_dir / "dataset_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {plot_path}")

    return summary


PATIENT_ID_COL = "PAT_MRN_ID_ENCRYPT"


def valid_patient_id_mask(series):

    return ~missing_value_mask(series)


def clean_text_series(series):

    return series.astype("string").str.strip()


def most_common_value(counter):
    
    if not counter:
        return "Missing"

    return sorted(
        counter.items(),
        key=lambda item: (-item[1], str(item[0]))
    )[0][0]


def summarize_patient_missingness(patient_row_counts,
                                  patient_null_counts,
                                  patient_blank_counts,
                                  patient_text_missing_counts,
                                  patient_missing_counts,
                                  patient_non_missing_counts,
                                  total_patients=None):
    
    patients_seen = set(patient_row_counts.keys())
    if total_patients is None:
        total_patients = len(patients_seen)

    patients_with_any_null = {
        pid for pid, count in patient_null_counts.items()
        if count > 0
    }
    patients_with_all_null = {
        pid for pid in patients_seen
        if patient_null_counts.get(pid, 0) == patient_row_counts.get(pid, 0)
        and patient_row_counts.get(pid, 0) > 0
    }

    patients_with_any_blank = {
        pid for pid, count in patient_blank_counts.items()
        if count > 0
    }
    patients_with_all_blank = {
        pid for pid in patients_seen
        if patient_blank_counts.get(pid, 0) == patient_row_counts.get(pid, 0)
        and patient_row_counts.get(pid, 0) > 0
    }

    patients_with_any_text_missing = {
        pid for pid, count in patient_text_missing_counts.items()
        if count > 0
    }
    patients_with_all_text_missing = {
        pid for pid in patients_seen
        if patient_text_missing_counts.get(pid, 0) == patient_row_counts.get(pid, 0)
        and patient_row_counts.get(pid, 0) > 0
    }

    patients_with_any_missing = {
        pid for pid, count in patient_missing_counts.items()
        if count > 0
    }
    patients_with_all_missing = {
        pid for pid in patients_seen
        if patient_missing_counts.get(pid, 0) == patient_row_counts.get(pid, 0)
        and patient_row_counts.get(pid, 0) > 0
    }

    patients_with_non_missing = {
        pid for pid, count in patient_non_missing_counts.items()
        if count > 0
    }

    patients_missing_or_no_non_missing_value = total_patients - len(patients_with_non_missing)

    return {
        "total_patients": int(total_patients),
        "patients_seen_in_dataset": int(len(patients_seen)),
        "patients_with_any_null": int(len(patients_with_any_null)),
        "patients_with_all_null": int(len(patients_with_all_null)),
        "patients_with_any_blank": int(len(patients_with_any_blank)),
        "patients_with_all_blank": int(len(patients_with_all_blank)),
        "patients_with_any_text_missing": int(len(patients_with_any_text_missing)),
        "patients_with_all_text_missing": int(len(patients_with_all_text_missing)),
        "patients_with_any_missing": int(len(patients_with_any_missing)),
        "patients_with_all_missing": int(len(patients_with_all_missing)),
        "patients_with_non_missing": int(len(patients_with_non_missing)),
        "patients_missing_or_no_non_missing_value": int(patients_missing_or_no_non_missing_value),
        "missing_or_no_non_missing_percent": round(
            float(patients_missing_or_no_non_missing_value / total_patients * 100), 4
        ) if total_patients > 0 else 0.0,
    }


def get_patient_ids_from_parquet(parquet_path, dataset_name):
    
    if not parquet_path.exists():
        print(f"WARNING: {dataset_name} Parquet file not found: {parquet_path}")
        return set()

    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names

    if PATIENT_ID_COL not in columns:
        print(f"WARNING: {PATIENT_ID_COL} not found in {dataset_name}")
        return set()

    patient_ids = set()

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=[PATIENT_ID_COL],
            use_threads=True
        ),
        desc=f"Collecting patient IDs from {dataset_name}"
    ):
        chunk = batch.to_pandas()
        id_mask = valid_patient_id_mask(chunk[PATIENT_ID_COL])
        patient_ids.update(
            clean_text_series(chunk.loc[id_mask, PATIENT_ID_COL]).tolist()
        )

    return patient_ids


def analyze_comorbidities_patient_level(parquet_path):

    if not parquet_path.exists():
        print(f"WARNING: Comorbidities Parquet file not found: {parquet_path}")
        return {}, set()

    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names

    if PATIENT_ID_COL not in all_cols:
        print(f"WARNING: {PATIENT_ID_COL} not found in Comorbidities")
        return {}, set()

    demographic_cols = [
        "PATIENT_AGE",
        "RACE",
        "ETHNICITY",
        "LANGUAGE",
        "SEX",
        "DISCHARGE_DISPOSITION",
    ]
    demographic_cols = [col for col in demographic_cols if col in all_cols]

    existing_cols = [PATIENT_ID_COL] + demographic_cols

    patient_ids = set()

    
    patient_row_counts = {col: Counter() for col in demographic_cols}
    patient_null_counts = {col: Counter() for col in demographic_cols}
    patient_blank_counts = {col: Counter() for col in demographic_cols}
    patient_text_missing_counts = {col: Counter() for col in demographic_cols}
    patient_missing_counts = {col: Counter() for col in demographic_cols}
    patient_non_missing_counts = {col: Counter() for col in demographic_cols}

    
    category_cols = [col for col in ["RACE", "ETHNICITY", "LANGUAGE", "SEX", "DISCHARGE_DISPOSITION"] if col in demographic_cols]
    patient_value_counters = {
        col: defaultdict(Counter)
        for col in category_cols
    }

  
    patient_age_non_numeric_counts = Counter()
    patient_age_zero_counts = Counter()
    patient_age_numeric_counts = Counter()

    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=existing_cols,
            use_threads=True
        ),
        desc="Analyzing Comorbidities patient-level missingness"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        id_mask = valid_patient_id_mask(chunk[PATIENT_ID_COL])
        chunk = chunk.loc[id_mask].copy()
        if chunk.empty:
            continue

        chunk[PATIENT_ID_COL] = clean_text_series(chunk[PATIENT_ID_COL])
        patient_ids.update(chunk[PATIENT_ID_COL].tolist())

        for col in demographic_cols:
            series = chunk[col]
            ids = chunk[PATIENT_ID_COL]

            as_string = series.astype("string")
            stripped_lower = as_string.str.strip().str.lower()

            null_mask = as_string.isna()
            blank_mask = stripped_lower == ""
            text_missing_mask = stripped_lower.isin(MISSING_STRING_VALUES - {""})
            missing_mask = null_mask | blank_mask | text_missing_mask
            non_missing_mask = ~missing_mask

            
            patient_row_counts[col].update(ids.tolist())
            patient_null_counts[col].update(ids.loc[null_mask].tolist())
            patient_blank_counts[col].update(ids.loc[blank_mask].tolist())
            patient_text_missing_counts[col].update(ids.loc[text_missing_mask].tolist())
            patient_missing_counts[col].update(ids.loc[missing_mask].tolist())
            patient_non_missing_counts[col].update(ids.loc[non_missing_mask].tolist())

            if col in category_cols:
                valid_values = clean_text_series(series.loc[non_missing_mask])
                valid_ids = ids.loc[non_missing_mask]
                for pid, value in zip(valid_ids.tolist(), valid_values.tolist()):
                    patient_value_counters[col][pid][str(value)] += 1

        if "PATIENT_AGE" in demographic_cols:
            age_raw = chunk["PATIENT_AGE"]
            age_missing_mask = missing_value_mask(age_raw)
            age_numeric = pd.to_numeric(age_raw, errors="coerce")
            ids = chunk[PATIENT_ID_COL]

            non_numeric_mask = age_numeric.isna() & ~age_missing_mask
            zero_mask = age_numeric.eq(0)
            numeric_mask = age_numeric.notna()

            patient_age_non_numeric_counts.update(ids.loc[non_numeric_mask].tolist())
            patient_age_zero_counts.update(ids.loc[zero_mask].tolist())
            patient_age_numeric_counts.update(ids.loc[numeric_mask].tolist())

    total_patients = len(patient_ids)


    missing_summary = {}
    missing_rows = []

    for col in demographic_cols:
        col_summary = summarize_patient_missingness(
            patient_row_counts[col],
            patient_null_counts[col],
            patient_blank_counts[col],
            patient_text_missing_counts[col],
            patient_missing_counts[col],
            patient_non_missing_counts[col],
            total_patients=total_patients,
        )
        missing_summary[col] = col_summary

        row = {"column": col}
        row.update(col_summary)
        missing_rows.append(row)

    missing_output_path = output_dir / "comorbidities_patient_level_missing_counts.csv"
    pd.DataFrame(missing_rows).sort_values(
        by=["patients_missing_or_no_non_missing_value", "column"],
        ascending=[False, True]
    ).to_csv(missing_output_path, index=False)

    resolved_values = {
        col: {}
        for col in category_cols
    }

    for col in category_cols:
        for pid in patient_ids:
            resolved_values[col][pid] = most_common_value(
                patient_value_counters[col].get(pid, Counter())
            )

    category_count_rows = []
    category_counts_json = {}

    for col in category_cols:
        counts = Counter(resolved_values[col].values())
        category_counts_json[col] = counter_to_sorted_dict(counts)

        for category_value, patient_count in sorted(
            counts.items(),
            key=lambda item: (-item[1], str(item[0]))
        ):
            category_count_rows.append({
                "column": col,
                "category": category_value,
                "patient_count": int(patient_count),
                "patient_percent": round(
                    float(patient_count / total_patients * 100), 4
                ) if total_patients > 0 else 0.0,
            })

    category_output_path = output_dir / "comorbidities_patient_level_category_counts.csv"
    pd.DataFrame(category_count_rows).to_csv(category_output_path, index=False)

    
    category_by_sex_rows = []
    category_by_sex_json = {}

    if "SEX" in resolved_values:
        for col in [c for c in ["RACE", "ETHNICITY", "LANGUAGE", "DISCHARGE_DISPOSITION"] if c in resolved_values]:
            nested = defaultdict(Counter)
            for pid in patient_ids:
                category_value = resolved_values[col].get(pid, "Missing")
                sex_value = resolved_values["SEX"].get(pid, "Missing")
                nested[category_value][sex_value] += 1

            category_by_sex_json[col] = nested_counter_to_sorted_dict(nested)

            for category_value, sex_counter in sorted(
                nested.items(),
                key=lambda item: (-sum(item[1].values()), str(item[0]))
            ):
                category_total = sum(sex_counter.values())
                for sex_value, patient_count in sorted(
                    sex_counter.items(),
                    key=lambda item: (-item[1], str(item[0]))
                ):
                    category_by_sex_rows.append({
                        "column": col,
                        "category": category_value,
                        "sex": sex_value,
                        "patient_count": int(patient_count),
                        "percent_within_category": round(
                            float(patient_count / category_total * 100), 4
                        ) if category_total > 0 else 0.0,
                    })

    category_by_sex_output_path = output_dir / "comorbidities_patient_level_category_by_sex_counts.csv"
    pd.DataFrame(category_by_sex_rows).to_csv(category_by_sex_output_path, index=False)


    age_quality_summary = {}
    if "PATIENT_AGE" in demographic_cols:
        patients_with_numeric_age = {
            pid for pid, count in patient_age_numeric_counts.items()
            if count > 0
        }
        patients_with_age_zero = {
            pid for pid, count in patient_age_zero_counts.items()
            if count > 0
        }
        patients_with_non_numeric_age = {
            pid for pid, count in patient_age_non_numeric_counts.items()
            if count > 0
        }
        patients_with_only_age_zero_among_numeric = {
            pid for pid in patients_with_numeric_age
            if patient_age_zero_counts.get(pid, 0) == patient_age_numeric_counts.get(pid, 0)
        }

        age_quality_summary = {
            "total_patients": int(total_patients),
            "patients_with_numeric_age": int(len(patients_with_numeric_age)),
            "patients_with_no_numeric_age": int(total_patients - len(patients_with_numeric_age)),
            "patients_with_any_age_zero": int(len(patients_with_age_zero)),
            "patients_with_only_age_zero_among_numeric_age_rows": int(len(patients_with_only_age_zero_among_numeric)),
            "patients_with_any_non_numeric_age": int(len(patients_with_non_numeric_age)),
        }

        age_quality_output_path = output_dir / "comorbidities_patient_level_age_quality_counts.csv"
        pd.DataFrame([age_quality_summary]).to_csv(age_quality_output_path, index=False)
    else:
        age_quality_output_path = None

    summary = {
        "total_comorbidities_rows_processed": int(total_rows),
        "total_comorbidities_patients": int(total_patients),
        "patient_level_missing_counts_csv": str(missing_output_path),
        "patient_level_category_counts_csv": str(category_output_path),
        "patient_level_category_by_sex_counts_csv": str(category_by_sex_output_path),
        "patient_level_age_quality_counts_csv": str(age_quality_output_path) if age_quality_output_path is not None else None,
        "patient_level_missing_counts": missing_summary,
        "patient_level_category_counts": category_counts_json,
        "patient_level_category_by_sex_counts": category_by_sex_json,
        "patient_level_age_quality_counts": age_quality_summary,
    }

    print("\n--- COMORBIDITIES PATIENT-LEVEL MISSING COUNTS ---")
    print(pd.DataFrame(missing_rows).to_string(index=False))
    print(f"Comorbidities patient-level missing counts saved to: {missing_output_path}")

    print("\n--- COMORBIDITIES PATIENT-LEVEL CATEGORY COUNTS ---")
    print(pd.DataFrame(category_count_rows).to_string(index=False))
    print(f"Comorbidities patient-level category counts saved to: {category_output_path}")

    print("\n--- COMORBIDITIES PATIENT-LEVEL CATEGORY BY SEX COUNTS ---")
    if category_by_sex_rows:
        print(pd.DataFrame(category_by_sex_rows).to_string(index=False))
    else:
        print("No SEX column available for category-by-sex counts.")
    print(f"Comorbidities patient-level category-by-sex counts saved to: {category_by_sex_output_path}")

    if age_quality_summary:
        print("\n--- COMORBIDITIES PATIENT-LEVEL AGE QUALITY COUNTS ---")
        print(json.dumps(age_quality_summary, indent=4))
        print(f"Comorbidities patient-level age quality counts saved to: {age_quality_output_path}")

    return summary, patient_ids


def analyze_parquet_patient_level_missing_counts(parquet_path, dataset_name, cohort_patient_ids=None):
    
    if not parquet_path.exists():
        print(f"WARNING: {dataset_name} Parquet file not found: {parquet_path}")
        return {}

    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names

    if PATIENT_ID_COL not in columns:
        print(f"WARNING: {PATIENT_ID_COL} not found in {dataset_name}; skipping patient-level missing counts.")
        return {}

    value_cols = [col for col in columns if col != PATIENT_ID_COL]

    dataset_patient_ids = set()
    patient_null_sets = {col: set() for col in value_cols}
    patient_blank_sets = {col: set() for col in value_cols}
    patient_text_missing_sets = {col: set() for col in value_cols}
    patient_missing_sets = {col: set() for col in value_cols}
    patient_non_missing_sets = {col: set() for col in value_cols}

    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=columns,
            use_threads=True
        ),
        desc=f"Counting patient-level missing values in {dataset_name}"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        id_mask = valid_patient_id_mask(chunk[PATIENT_ID_COL])
        chunk = chunk.loc[id_mask].copy()
        if chunk.empty:
            continue

        chunk[PATIENT_ID_COL] = clean_text_series(chunk[PATIENT_ID_COL])
        ids = chunk[PATIENT_ID_COL]
        dataset_patient_ids.update(ids.tolist())

        for col in value_cols:
            series = chunk[col]
            as_string = series.astype("string")
            stripped_lower = as_string.str.strip().str.lower()

            null_mask = as_string.isna()
            blank_mask = stripped_lower == ""
            text_missing_mask = stripped_lower.isin(MISSING_STRING_VALUES - {""})
            missing_mask = null_mask | blank_mask | text_missing_mask
            non_missing_mask = ~missing_mask

            patient_null_sets[col].update(ids.loc[null_mask].unique().tolist())
            patient_blank_sets[col].update(ids.loc[blank_mask].unique().tolist())
            patient_text_missing_sets[col].update(ids.loc[text_missing_mask].unique().tolist())
            patient_missing_sets[col].update(ids.loc[missing_mask].unique().tolist())
            patient_non_missing_sets[col].update(ids.loc[non_missing_mask].unique().tolist())

    if cohort_patient_ids is not None and len(cohort_patient_ids) > 0:
        denominator_patients = set(cohort_patient_ids)
        denominator_name = "cohort_patients"
    else:
        denominator_patients = set(dataset_patient_ids)
        denominator_name = "dataset_patients"

    denominator = len(denominator_patients)

    result = {}
    rows = []

    for col in value_cols:
        patients_with_non_missing = patient_non_missing_sets[col]
        patients_missing_or_no_non_missing = denominator_patients - patients_with_non_missing

        col_result = {
            "denominator_type": denominator_name,
            "total_rows_processed": int(total_rows),
            "dataset_patients_with_rows": int(len(dataset_patient_ids)),
            "denominator_patients": int(denominator),
            "patients_with_any_null": int(len(patient_null_sets[col])),
            "patients_with_any_blank": int(len(patient_blank_sets[col])),
            "patients_with_any_text_missing": int(len(patient_text_missing_sets[col])),
            "patients_with_any_missing": int(len(patient_missing_sets[col])),
            "patients_with_non_missing": int(len(patients_with_non_missing)),
            "patients_missing_or_no_non_missing_value": int(len(patients_missing_or_no_non_missing)),
            "missing_or_no_non_missing_percent": round(
                float(len(patients_missing_or_no_non_missing) / denominator * 100), 4
            ) if denominator > 0 else 0.0,
        }

        result[col] = col_result
        row = {"column": col}
        row.update(col_result)
        rows.append(row)

    output_path = output_dir / f"{dataset_name}_patient_level_missing_counts.csv"
    pd.DataFrame(rows).sort_values(
        by=["patients_missing_or_no_non_missing_value", "column"],
        ascending=[False, True]
    ).to_csv(output_path, index=False)

    print(f"\n--- {dataset_name.upper()} PATIENT-LEVEL MISSING VALUE COUNTS ---")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"{dataset_name} patient-level missing counts saved to: {output_path}")

    return result


def analyze_vitals_measurement_patient_level_missing_counts(parquet_path, cohort_patient_ids=None):
    

    if not parquet_path.exists():
        print(f"WARNING: Vitals Parquet file not found: {parquet_path}")
        return {}

    pf = pq.ParquetFile(parquet_path)
    columns = pf.schema_arrow.names

    if PATIENT_ID_COL not in columns:
        print(f"WARNING: {PATIENT_ID_COL} not found in vitals; skipping patient-level vital missingness.")
        return {}

    vital_name_col = find_first_existing_column(
        columns,
        [
            "FLO_MEAS_NAME",
            "MEAS_NAME",
            "VITAL_NAME",
            "VITAL_SIGN",
            "COMPONENT_NAME",
            "DISPLAY_NAME",
            "FLO_MEAS_ID_DISP_NAME",
        ]
    )

    vital_value_col = find_first_existing_column(
        columns,
        [
            "MEAS_VALUE",
            "MEAS_VALUE_NUM",
            "VALUE",
            "RESULT_VALUE",
            "VITAL_VALUE",
            "OBS_VALUE",
        ]
    )

    if vital_name_col is None:
        print("WARNING: Could not detect the vitals name column, such as FLO_MEAS_NAME.")
        print("Available vitals columns:")
        print(columns)
        return {}

    if vital_value_col is None:
        print("WARNING: Could not detect the vitals value column, such as MEAS_VALUE.")
        print("Available vitals columns:")
        print(columns)
        return {}

    vital_patterns = {
        "bmi": {
            "include": ["BMI", "BODY MASS INDEX"],
            "exclude": [],
        },
        "weight": {
            "include": ["WEIGHT", " WT"],
            "exclude": [],
        },
        "bp": {
            "include": ["BLOOD PRESSURE", " BP", "BP ", "SYSTOLIC", "DIASTOLIC"],
            "exclude": [],
        },
        "height": {
            "include": ["HEIGHT", " HT"],
            "exclude": [],
        },
    }

    if cohort_patient_ids is not None and len(cohort_patient_ids) > 0:
        denominator_patients = set(cohort_patient_ids)
        denominator_name = "comorbidities_cohort_patients"
    else:
        denominator_patients = set()
        denominator_name = "vitals_dataset_patients"

    dataset_patient_ids = set()

    patient_row_counts = {vital_type: Counter() for vital_type in vital_patterns}
    patient_null_counts = {vital_type: Counter() for vital_type in vital_patterns}
    patient_blank_counts = {vital_type: Counter() for vital_type in vital_patterns}
    patient_text_missing_counts = {vital_type: Counter() for vital_type in vital_patterns}
    patient_missing_counts = {vital_type: Counter() for vital_type in vital_patterns}
    patient_non_missing_counts = {vital_type: Counter() for vital_type in vital_patterns}

    
    patient_name_sets = {
        vital_type: defaultdict(set)
        for vital_type in vital_patterns
    }

    total_rows = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=[PATIENT_ID_COL, vital_name_col, vital_value_col],
            use_threads=True
        ),
        desc="Counting BMI/weight/BP/height patient-level missingness in vitals"
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        id_mask = valid_patient_id_mask(chunk[PATIENT_ID_COL])
        chunk = chunk.loc[id_mask].copy()
        if chunk.empty:
            continue

        chunk[PATIENT_ID_COL] = clean_text_series(chunk[PATIENT_ID_COL])
        dataset_patient_ids.update(chunk[PATIENT_ID_COL].tolist())

        names_upper = chunk[vital_name_col].astype("string").str.upper().str.strip()

        for vital_type, pattern_info in vital_patterns.items():
            mask = pd.Series(False, index=chunk.index)

            for term in pattern_info["include"]:
                mask = mask | names_upper.str.contains(term, na=False, regex=False)

            for term in pattern_info["exclude"]:
                mask = mask & ~names_upper.str.contains(term, na=False, regex=False)

            matched = chunk.loc[mask]
            if matched.empty:
                continue

            ids = matched[PATIENT_ID_COL]
            values = matched[vital_value_col]
            as_string = values.astype("string")
            stripped_lower = as_string.str.strip().str.lower()

            null_mask = as_string.isna()
            blank_mask = stripped_lower == ""
            text_missing_mask = stripped_lower.isin(MISSING_STRING_VALUES - {""})
            missing_mask = null_mask | blank_mask | text_missing_mask
            non_missing_mask = ~missing_mask

            patient_row_counts[vital_type].update(ids.tolist())
            patient_null_counts[vital_type].update(ids.loc[null_mask].tolist())
            patient_blank_counts[vital_type].update(ids.loc[blank_mask].tolist())
            patient_text_missing_counts[vital_type].update(ids.loc[text_missing_mask].tolist())
            patient_missing_counts[vital_type].update(ids.loc[missing_mask].tolist())
            patient_non_missing_counts[vital_type].update(ids.loc[non_missing_mask].tolist())

            matched_names = clean_text_series(matched[vital_name_col])
            for pid, vital_name in zip(ids.tolist(), matched_names.tolist()):
                if pd.notna(vital_name):
                    patient_name_sets[vital_type][str(vital_name)].add(pid)

    if denominator_name == "vitals_dataset_patients":
        denominator_patients = set(dataset_patient_ids)

    denominator = len(denominator_patients)
    result = {}
    rows = []
    name_rows = []

    for vital_type in vital_patterns:
        patients_with_rows = set(patient_row_counts[vital_type].keys())
        patients_with_non_missing = {
            pid for pid, count in patient_non_missing_counts[vital_type].items()
            if count > 0
        }
        patients_with_no_measurement_rows = denominator_patients - patients_with_rows
        patients_with_all_measurement_rows_missing = {
            pid for pid in patients_with_rows
            if patient_missing_counts[vital_type].get(pid, 0) == patient_row_counts[vital_type].get(pid, 0)
            and patient_row_counts[vital_type].get(pid, 0) > 0
        }
        patients_missing_vital_overall = denominator_patients - patients_with_non_missing

        summary = {
            "denominator_type": denominator_name,
            "vital_name_column": vital_name_col,
            "vital_value_column": vital_value_col,
            "total_vitals_rows_processed": int(total_rows),
            "dataset_patients_with_any_vitals_rows": int(len(dataset_patient_ids)),
            "denominator_patients": int(denominator),
            "patients_with_measurement_rows": int(len(patients_with_rows)),
            "patients_with_no_measurement_rows": int(len(patients_with_no_measurement_rows)),
            "patients_with_any_null_measurement_row": int(len({pid for pid, count in patient_null_counts[vital_type].items() if count > 0})),
            "patients_with_any_blank_measurement_row": int(len({pid for pid, count in patient_blank_counts[vital_type].items() if count > 0})),
            "patients_with_any_text_missing_measurement_row": int(len({pid for pid, count in patient_text_missing_counts[vital_type].items() if count > 0})),
            "patients_with_any_missing_measurement_row": int(len({pid for pid, count in patient_missing_counts[vital_type].items() if count > 0})),
            "patients_with_all_measurement_rows_missing": int(len(patients_with_all_measurement_rows_missing)),
            "patients_with_non_missing_measurement": int(len(patients_with_non_missing)),
            "patients_missing_vital_overall": int(len(patients_missing_vital_overall)),
            "missing_vital_overall_percent": round(
                float(len(patients_missing_vital_overall) / denominator * 100), 4
            ) if denominator > 0 else 0.0,
        }

        result[vital_type] = summary
        row = {"vital_type": vital_type}
        row.update(summary)
        rows.append(row)

        for vital_name, patient_set in sorted(
            patient_name_sets[vital_type].items(),
            key=lambda item: (-len(item[1]), str(item[0]))
        ):
            name_rows.append({
                "vital_type": vital_type,
                "matched_vital_name": vital_name,
                "patient_count": int(len(patient_set)),
                "patient_percent_of_denominator": round(
                    float(len(patient_set) / denominator * 100), 4
                ) if denominator > 0 else 0.0,
            })

    output_path = output_dir / "vitals_bmi_weight_bp_height_patient_level_missing_counts.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    name_output_path = output_dir / "vitals_bmi_weight_bp_height_patient_level_name_counts.csv"
    pd.DataFrame(name_rows).to_csv(name_output_path, index=False)

    print("\n--- VITALS BMI / WEIGHT / BP / HEIGHT PATIENT-LEVEL MISSINGNESS ---")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Vitals BMI/weight/BP/height patient-level missing counts saved to: {output_path}")
    print(f"Vitals BMI/weight/BP/height patient-level name counts saved to: {name_output_path}")

    return result


def main():

    print("\n==============================")
    print("STEP 1: CONVERT CSV TO PARQUET")
    print("==============================")

    for name, info in files.items():
        convert_csv_to_parquet(
            csv_path=info["csv"],
            parquet_path=info["parquet"],
            sep=info["sep"],
            chunksize=CSV_CHUNKSIZE
        )

    print("\n==============================")
    print("STEP 2: DISPLAY VITALS HEAD AND COLUMNS")
    print("==============================")

    show_parquet_head_and_columns(
        files["vitals"]["parquet"],
        "vitals"
    )

    print("\n==============================")
    print("STEP 3: DISPLAY LABS HEAD AND COLUMNS")
    print("==============================")

    show_parquet_head_and_columns(
        files["labs"]["parquet"],
        "labs"
    )

    print("\n==============================")
    print("STEP 4: DISPLAY MEDS HEAD AND COLUMNS")
    print("==============================")

    show_parquet_head_and_columns(
        files["meds"]["parquet"],
        "meds"
    )

    print("\n==============================")
    print("STEP 5: ANALYZE COMORBIDITIES IN BATCHES - ROW LEVEL")
    print("==============================")

    summary = analyze_comorbidities_parquet(
        files["comorbidities"]["parquet"]
    )

    print("\n==============================")
    print("STEP 5B: ANALYZE COMORBIDITIES - PATIENT LEVEL")
    print("==============================")

    comorbidities_patient_level_summary, cohort_patient_ids = analyze_comorbidities_patient_level(
        files["comorbidities"]["parquet"]
    )

    summary["comorbidities_patient_level"] = comorbidities_patient_level_summary

    print("\n==============================")
    print("STEP 6: MISSING VALUE COUNTS FOR LABS, VITALS, MEDS - ROW LEVEL")
    print("==============================")

    labs_missing_counts = analyze_parquet_missing_counts(
        files["labs"]["parquet"],
        "labs"
    )

    vitals_missing_counts = analyze_parquet_missing_counts(
        files["vitals"]["parquet"],
        "vitals"
    )

    meds_missing_counts = analyze_parquet_missing_counts(
        files["meds"]["parquet"],
        "meds"
    )

    vitals_measurement_missing_counts = analyze_vitals_measurement_missing_counts(
        files["vitals"]["parquet"]
    )

    summary["labs_missing_counts"] = labs_missing_counts
    summary["vitals_missing_counts"] = vitals_missing_counts
    summary["meds_missing_counts"] = meds_missing_counts
    summary["vitals_bmi_weight_bp_height_missing_counts"] = vitals_measurement_missing_counts

    print("\n==============================")
    print("STEP 6B: MISSING VALUE COUNTS FOR LABS, VITALS, MEDS - PATIENT LEVEL")
    print("==============================")

    labs_patient_level_missing_counts = analyze_parquet_patient_level_missing_counts(
        files["labs"]["parquet"],
        "labs",
        cohort_patient_ids=cohort_patient_ids
    )

    vitals_patient_level_missing_counts = analyze_parquet_patient_level_missing_counts(
        files["vitals"]["parquet"],
        "vitals",
        cohort_patient_ids=cohort_patient_ids
    )

    meds_patient_level_missing_counts = analyze_parquet_patient_level_missing_counts(
        files["meds"]["parquet"],
        "meds",
        cohort_patient_ids=cohort_patient_ids
    )

    vitals_measurement_patient_level_missing_counts = analyze_vitals_measurement_patient_level_missing_counts(
        files["vitals"]["parquet"],
        cohort_patient_ids=cohort_patient_ids
    )

    summary["labs_patient_level_missing_counts"] = labs_patient_level_missing_counts
    summary["vitals_patient_level_missing_counts"] = vitals_patient_level_missing_counts
    summary["meds_patient_level_missing_counts"] = meds_patient_level_missing_counts
    summary["vitals_bmi_weight_bp_height_patient_level_missing_counts"] = vitals_measurement_patient_level_missing_counts

    summary_path = output_dir / "overview_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nUpdated summary with row-level and patient-level missing counts: {summary_path}")

    print("\n==============================")
    print("STEP 7: UNIQUE MRN COUNTS")
    print("==============================")

    unique_mrn_counts = {}

    for name, info in files.items():
        unique_mrn_counts[name] = get_unique_mrn_count_from_parquet(
            info["parquet"],
            name
        )

    print("\n--- UNIQUE MRN COUNTS ---")
    for name, count in unique_mrn_counts.items():
        print(f"{name}: {count:,}")

    unique_counts_path = output_dir / "unique_mrn_counts.json"

    with open(unique_counts_path, "w") as f:
        json.dump(unique_mrn_counts, f, indent=4)

    summary["unique_mrn_counts"] = {
        str(name): int(count)
        for name, count in unique_mrn_counts.items()
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nUnique MRN counts saved to: {unique_counts_path}")
    print(f"overview_summary.json updated with unique MRN counts: {summary_path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
