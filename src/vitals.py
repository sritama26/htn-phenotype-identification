#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

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

VITALS_CSV = DATA_DIR / "Hypertension Vitals 20250520.csv"
VITALS_PARQUET = PARQUET_DIR / "Hypertension Vitals 20250520.parquet"

VITALS_SEP = ","

CSV_CHUNKSIZE = 500_000
BATCH_ROWS = 100_000
PREVIEW_ROWS = 5

FORCE_RECONVERT = False

LONG_CSV_OUT = OUTPUT_DIR / "vitals_long_clean.csv"
LONG_PARQUET_OUT = OUTPUT_DIR / "vitals_long_clean.parquet"

BYPATIENT_CSV_OUT = OUTPUT_DIR / "vitals_by_patient.csv"
BYPATIENT_PARQUET_OUT = OUTPUT_DIR / "vitals_by_patient.parquet"


PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
DAY_COL = "MEAS_DAYS_SINCE_DX_FRST_OBSRVD"
FLO_COL = "FLO_MEAS_NAME"
DISP_COL = "DISP_NAME"
VALUE_COL = "MEAS_VALUE"
UNIT_COL = "UNITS"

ACCEPTED_ONLY = False
ACCEPTED_COL = "ISACCEPTED_YN"

MISSING_TOKENS = {
    "", "NAN", "NA", "N/A", "NULL", "NONE", ".", "?", "-",
    "SEE NOTE", "SEE COMMENT", "TNP", "PENDING"
}


VITAL_PATTERNS = {
    "height": (["HEIGHT"], []),
    "weight": (["WEIGHT/SCALE"], ["IDEAL", "DRY", "BIRTH", "CHANGE", "GOAL"]),
    "bmi":    (["BMI", "BODY MASS INDEX"], []),
    "bp":     (["BP", "BLOOD PRESSURE"], [
        "MAP",
        "MEAN ARTERIAL",
        "ARTERIAL LINE",
        "GOAL",
        "ORTHOSTATIC NOTE",
    ]),
}

BP_PARTS = ["bp_systolic", "bp_diastolic"]
VITAL_SERIES = ["height", "weight", "bmi", *BP_PARTS]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_FTIN_RE = re.compile(r"""(\d+(?:\.\d+)?)\s*'\s*(\d+(?:\.\d+)?)?\s*"?""")



def print_delimiter_check(csv_path):
    
    if not csv_path.exists():
        print(f"WARNING: file does not exist: {csv_path}")
        return

    with open(csv_path, "r", encoding="latin1", errors="replace") as f:
        first = f.readline()

    print(f"\nDelimiter check for: {csv_path.name}")
    print("First line preview:")
    print(first[:500])

    for sep in [",", "|", "\t", ";", "~"]:
        print(f"{repr(sep)} count = {first.count(sep)}")


def convert_csv_to_parquet(csv_path, parquet_path, sep=",", chunksize=500_000):
   

    if parquet_path.exists() and not FORCE_RECONVERT:
        print(f"Parquet already exists, skipping conversion: {parquet_path}")
        return

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if parquet_path.exists() and FORCE_RECONVERT:
        print(f"Removing old parquet file: {parquet_path}")
        parquet_path.unlink()

    print_delimiter_check(csv_path)

    print("\nConverting CSV to Parquet:")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"sep:     {repr(sep)}")

    reader = pd.read_csv(
        csv_path,
        sep=sep,
        quotechar='"',
        encoding="latin1",
        dtype=str,
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

       
        if list(chunk.columns) != schema.names:
            raise ValueError(
                "Column mismatch between chunks.\n"
                f"Expected: {schema.names}\n"
                f"Found:    {list(chunk.columns)}"
            )

        table = pa.Table.from_pandas(
            chunk,
            schema=schema,
            preserve_index=False,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                parquet_path,
                schema,
                compression="snappy",
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()

    print(f"Finished converting {csv_path.name}")
    print(f"Total rows written: {total_rows:,}")


def show_parquet_head_and_columns(parquet_path):
    pf = pq.ParquetFile(parquet_path)

    print("\n--- VITALS Parquet Metadata ---")
    print(f"Rows: {pf.metadata.num_rows:,}")
    print(f"Columns: {pf.metadata.num_columns:,}")

    print("\n--- VITALS Columns ---")
    print(pf.schema_arrow.names)

    print("\n--- VITALS Head ---")
    first_batch = next(pf.iter_batches(batch_size=PREVIEW_ROWS, use_threads=True))
    print(first_batch.to_pandas().head(PREVIEW_ROWS).to_string())


def json_safe(value):
    return json.dumps(value, ensure_ascii=False)


def save_by_patient_outputs(df):
    """
    Save patient-level arrays as both:
      - Parquet: list columns remain list-like
      - CSV: list columns are JSON strings
    """
    df.to_parquet(BYPATIENT_PARQUET_OUT, index=False)

    out_csv = df.copy()
    array_cols = [c for c in out_csv.columns if c.endswith(("_days", "_values"))]

    for col in array_cols:
        out_csv[col] = out_csv[col].apply(json_safe)

    out_csv.to_csv(BYPATIENT_CSV_OUT, index=False, encoding="utf-8")

    print(f"Saved patient-level vitals Parquet -> {BYPATIENT_PARQUET_OUT}")
    print(f"Saved patient-level vitals CSV     -> {BYPATIENT_CSV_OUT}")


def to_numeric(v) -> float:
   
    s = str(v).strip().replace(",", "")

    if s.upper() in MISSING_TOKENS:
        return np.nan

    m = _NUM_RE.search(s)
    return float(m.group()) if m else np.nan


def parse_height(v) -> float:
 
    s = str(v).strip()

    if s.upper() in MISSING_TOKENS:
        return np.nan

    if "'" in s:
        m = _FTIN_RE.search(s)
        if m:
            feet = float(m.group(1))
            inches = float(m.group(2)) if m.group(2) else 0.0
            return feet * 12 + inches

    return to_numeric(s)


def parse_bp(v):
    
    s = str(v).strip()

    if s.upper() in MISSING_TOKENS:
        return (np.nan, np.nan)

    if "/" in s:
        sys_s, _, dia_s = s.partition("/")
        return (to_numeric(sys_s), to_numeric(dia_s))

    return (to_numeric(s), np.nan)


def vital_for_name(name, cache, conflicts):
    
    if name in cache:
        return cache[name]

    u = str(name).upper()

    hits = [
        vital
        for vital, (includes, excludes) in VITAL_PATTERNS.items()
        if any(term in u for term in includes)
        and not any(term in u for term in excludes)
    ]

    vital = hits[0] if hits else None
    cache[name] = vital

    if len(hits) > 1:
        conflicts[name] = hits

    return vital


def mode_from_counter(counter: Counter) -> str:
    if not counter:
        return ""

    top = max(counter.values())
    return min(k for k, v in counter.items() if v == top)


def value_for_base(base, raw):
    return parse_height(raw) if base == "height" else to_numeric(raw)



class LongVitalsWriter:
   
    def __init__(self, csv_path, parquet_path):
        self.csv_path = Path(csv_path)
        self.parquet_path = Path(parquet_path)

        if self.csv_path.exists():
            self.csv_path.unlink()
        if self.parquet_path.exists():
            self.parquet_path.unlink()

        self.n_rows = 0
        self._csv_header_written = False

        self.schema = pa.schema([
            pa.field(PATIENT_COL, pa.string()),
            pa.field("vital", pa.string()),
            pa.field("day", pa.int64()),
            pa.field("value", pa.float64()),
            pa.field("unit", pa.string()),
            pa.field("disp_name", pa.string()),
        ])

        self._parquet_writer = pq.ParquetWriter(
            self.parquet_path,
            self.schema,
            compression="snappy",
        )

    def write(self, df):
        if df.empty:
            return

        out = df[[PATIENT_COL, "vital", "day", "value", "unit", "disp_name"]].copy()

        out[PATIENT_COL] = out[PATIENT_COL].astype(str)
        out["vital"] = out["vital"].astype(str)
        out["day"] = pd.to_numeric(out["day"], errors="coerce").astype("int64")
        out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")
        out["unit"] = out["unit"].fillna("").astype(str)
        out["disp_name"] = out["disp_name"].fillna("").astype(str)

        out.to_csv(
            self.csv_path,
            mode="a",
            header=not self._csv_header_written,
            index=False,
            encoding="utf-8",
        )
        self._csv_header_written = True

        table = pa.Table.from_pandas(
            out,
            schema=self.schema,
            preserve_index=False,
        )
        self._parquet_writer.write_table(table)

        self.n_rows += len(out)

    def close(self):
        self._parquet_writer.close()



def process_vitals_batches():
   
    pf = pq.ParquetFile(VITALS_PARQUET)
    all_cols = pf.schema_arrow.names

    requested_cols = [
        PATIENT_COL,
        DAY_COL,
        FLO_COL,
        DISP_COL,
        VALUE_COL,
        UNIT_COL,
        ACCEPTED_COL,
    ]

    columns = [c for c in requested_cols if c in all_cols]
    missing = [c for c in requested_cols if c not in all_cols]

    if missing:
        print("\nWARNING: These expected vitals columns are missing:")
        print(missing)

    required = [PATIENT_COL, DAY_COL, FLO_COL, VALUE_COL]
    missing_required = [c for c in required if c not in columns]

    if missing_required:
        raise KeyError(
            f"Required vitals columns missing: {missing_required}\n"
            f"Found columns: {all_cols}"
        )

    name_cache = {}
    conflicts = {}

    flo_counts = defaultdict(Counter)
    vital_n = Counter()

    
    series_days = defaultdict(list)
    series_vals = defaultdict(list)
    series_unit = defaultdict(Counter)
    patients = set()

    long_writer = LongVitalsWriter(LONG_CSV_OUT, LONG_PARQUET_OUT)

    n_seen = 0
    drop_val = 0
    drop_day = 0

    for batch in tqdm(
        pf.iter_batches(
            batch_size=BATCH_ROWS,
            columns=columns,
            use_threads=True,
        ),
        desc="Building vitals time-series",
    ):
        chunk = batch.to_pandas()

        if ACCEPTED_ONLY and ACCEPTED_COL in chunk.columns:
            chunk = chunk[chunk[ACCEPTED_COL].fillna("").astype(str).str.upper().eq("Y")]
            if chunk.empty:
                continue

        base = chunk[FLO_COL].map(lambda n: vital_for_name(n, name_cache, conflicts))
        keep = chunk[base.notna()].copy()

        if keep.empty:
            continue

        keep["base"] = base[base.notna()]
        keep["day"] = pd.to_numeric(keep[DAY_COL], errors="coerce")

        if UNIT_COL in keep.columns:
            keep["unit"] = keep[UNIT_COL].fillna("").astype(str)
        else:
            keep["unit"] = ""

        if DISP_COL in keep.columns:
            keep["disp_name"] = keep[DISP_COL].fillna("").astype(str)
        else:
            keep["disp_name"] = ""

       
        others = keep[keep["base"].ne("bp")].copy()

        if not others.empty:
            others["vital"] = others["base"]
            others["value"] = [
                value_for_base(b, v)
                for b, v in zip(others["base"], others[VALUE_COL])
            ]

      
        bp = keep[keep["base"].eq("bp")]
        bp_rows = []

        if not bp.empty:
            parsed = bp[VALUE_COL].map(parse_bp)

            for part_idx, vital_name in ((0, "bp_systolic"), (1, "bp_diastolic")):
                r = bp.copy()
                r["vital"] = vital_name
                r["value"] = [p[part_idx] for p in parsed]
                bp_rows.append(r)

        if bp_rows:
            bp_long = pd.concat(bp_rows, ignore_index=True)
        else:
            bp_long = keep.iloc[0:0].assign(vital="", value=np.nan)

        out_cols = [PATIENT_COL, "vital", "day", "value", "unit", "disp_name"]
        parts = [df[out_cols] for df in (others, bp_long) if not df.empty]

        if not parts:
            continue

        long_chunk = pd.concat(parts, ignore_index=True)

        n_seen += len(long_chunk)
        drop_val += int(long_chunk["value"].isna().sum())
        drop_day += int(long_chunk["day"].isna().sum())

        long_chunk = long_chunk.dropna(subset=["value", "day"]).copy()

        if long_chunk.empty:
            continue

        long_chunk["day"] = long_chunk["day"].round(0).astype(int)

       
        for base_name, sub in keep.groupby("base"):
            flo_counts[base_name].update(sub[FLO_COL].fillna("").astype(str))

        for vital_name, sub in long_chunk.groupby("vital"):
            vital_n[vital_name] += len(sub)

        long_chunk = long_chunk.sort_values([PATIENT_COL, "vital", "day"])
        long_writer.write(long_chunk)

      
        for (pid, vital_name), sub in long_chunk.groupby([PATIENT_COL, "vital"]):
            pid = str(pid)
            series_days[(pid, vital_name)].extend(sub["day"].tolist())
            series_vals[(pid, vital_name)].extend(sub["value"].tolist())
            series_unit[(pid, vital_name)].update(sub["unit"].fillna("").astype(str).tolist())
            patients.add(pid)

    long_writer.close()

    if conflicts:
        print("\nWARNING: FLO_MEAS_NAMEs matching >1 vital; assigned to first match:")
        for name, hits in list(conflicts.items())[:20]:
            print(f"  {name!r} -> {hits}")

    print(
        f"\nKept-vital rows: {n_seen:,} | "
        f"dropped non-numeric value: {drop_val:,} | "
        f"missing day: {drop_day:,} | "
        f"usable: {long_writer.n_rows:,}"
    )

    print("\nMatched FLO_MEAS_NAMEs per vital:")
    for base_name in VITAL_PATTERNS:
        names = flo_counts[base_name]
        tag = (
            "  ".join(f"{nm} ({cnt})" for nm, cnt in names.most_common(4))
            if names
            else "*** none ***"
        )
        print(f"  {base_name:<8} {tag}")

    by_patient = build_by_patient(
        patients=sorted(patients),
        series_days=series_days,
        series_vals=series_vals,
        series_unit=series_unit,
    )

    save_by_patient_outputs(by_patient)

    n_pat = by_patient.shape[0]

    print(f"\nPatients with >=1 target vital: {n_pat:,}")
    print("Coverage (% of these patients with >=1 measurement):")

    for vital_name in VITAL_SERIES:
        cov = (
            by_patient[f"{vital_name}_values"].apply(len).gt(0).mean() * 100
            if n_pat
            else 0.0
        )
        print(f"  {vital_name:<13} n={vital_n[vital_name]:<10} {cov:5.1f}%")

    print(f"\nSaved tidy long vitals CSV     -> {LONG_CSV_OUT}")
    print(f"Saved tidy long vitals Parquet -> {LONG_PARQUET_OUT}")

    return by_patient


def build_by_patient(patients, series_days, series_vals, series_unit):
   
    out = pd.DataFrame({PATIENT_COL: patients})

    for vital_name in VITAL_SERIES:
        days_map = {}
        vals_map = {}
        unit_map = {}

        for pid in patients:
            key = (pid, vital_name)

            if key not in series_days:
                continue

            pairs = sorted(
                zip(series_days[key], series_vals[key]),
                key=lambda t: t[0],
            )

            days_map[pid] = [int(d) for d, _ in pairs]
            vals_map[pid] = [float(v) for _, v in pairs]
            unit_map[pid] = mode_from_counter(series_unit[key])

        out[f"{vital_name}_days"] = out[PATIENT_COL].map(days_map).apply(
            lambda x: x if isinstance(x, list) else []
        )
        out[f"{vital_name}_values"] = out[PATIENT_COL].map(vals_map).apply(
            lambda x: x if isinstance(x, list) else []
        )
        out[f"{vital_name}_unit"] = out[PATIENT_COL].map(unit_map).fillna("")

    return out


def main():
    print("\n==============================")
    print("STEP 1: CONVERT VITALS CSV TO PARQUET")
    print("==============================")

    convert_csv_to_parquet(
        csv_path=VITALS_CSV,
        parquet_path=VITALS_PARQUET,
        sep=VITALS_SEP,
        chunksize=CSV_CHUNKSIZE,
    )

    print("\n==============================")
    print("STEP 2: DISPLAY VITALS HEAD AND COLUMNS")
    print("==============================")

    show_parquet_head_and_columns(VITALS_PARQUET)

    print("\n==============================")
    print("STEP 3: BUILD VITALS TIME-SERIES IN BATCHES")
    print("==============================")

    process_vitals_batches()

    print("\nDONE.")


if __name__ == "__main__":
    main()
