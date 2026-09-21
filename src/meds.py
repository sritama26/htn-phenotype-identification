#!/usr/bin/env python3

import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


MODE = "curated"  # "curated" or "subclass"
DATA_DIR = Path("/projects/f_miarc_1/Hypertension")
SCRIPT_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama")
OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
PARQUET_DIR = OUTPUT_DIR / "parquet_files"

MEDS_PATH = DATA_DIR / "Hypertension Medications 20250417.csv"  
MEDS_PARQUET = PARQUET_DIR / "Hypertension_Medications_20250417.parquet"
LONG_OUT = OUTPUT_DIR / "meds_long_clean_new.csv"       
GROUPS_OUT = OUTPUT_DIR / "patient_med_groups_new.csv"  
TOP_MED_FREQ_CSV = OUTPUT_DIR / "med_patient_frequencies_new.csv"

ENCODING = "latin1"
SEP = ","
CSV_CHUNKSIZE = 500_000
BATCH_ROWS = 100_000
FORCE_RECONVERT = False

SUBCLASS_COL = "PHARMACEUTICAL_SUBCLASS"  
DROP_STATUSES = []  
PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
GENERIC_COL = "SIMPLE_GENERIC"
MEDNAME_COL = "MEDICATION_NAME"
STATUS_COL = "ORDER_STATUS"
KEEP_COLS = [
    PATIENT_COL, "ORD_DAYS_SINCE_DX_FRST_OBSRVD", "STRT_DAYS_SINCE_DX_FRST_OBSRVD",
    "END_DAYS_SINCE_DX_FRST_OBSRVD", MEDNAME_COL, GENERIC_COL, "MED_ADMIN_ROUTE",
    "THERAPY_CLASS", "PHARMACUETICAL_CLASS", "PHARMACEUTICAL_SUBCLASS",
    STATUS_COL, "ACTIVE_ORDER_STATUS",
]
MISSING_TOKENS = {"", "NAN", "NA", "N/A", "NULL", "NONE", "."}

TOP_MED_MIN_PCT = 20
INCLUDE_STANDALONE_TOP_MEDS = True


MED_GROUPS = {
    "med_diuretic":               ["furosemide", "bumetanide", "torsemide", "ethacrynic", "amiloride",
                                    "hydrochlorothiazide", "hctz", "indapamide", "triamterene",
                                    "chlorthalidone", "chlorothiazide", "metolazone",
                                    "spironolactone", "eplerenone"],
    "med_beta_blocker":           ["metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol",
                                    "labetalol", "nebivolol", "nadolol", "sotalol", "esmolol",
                                    "betaxolol", "acebutolol", "pindolol", "timolol", "penbutolol"],
    "med_ace_inhibitor":          ["lisinopril", "enalapril", "ramipril", "benazepril", "captopril",
                                    "fosinopril", "quinapril", "perindopril", "trandolapril", "moexipril"],
    "med_arb":                    ["losartan", "valsartan", "olmesartan", "telmisartan", "candesartan",
                                    "irbesartan", "azilsartan", "eprosartan"],
    "med_ccb":                    ["amlodipine", "nifedipine", "felodipine", "nicardipine", "isradipine",
                                    "nisoldipine", "clevidipine", "diltiazem", "verapamil"],
    "med_alpha_blocker":          ["doxazosin", "prazosin", "terazosin"],
    "med_central_alpha2_agonist": ["clonidine", "methyldopa", "guanfacine", "guanabenz", "reserpine"],
    "med_vasodilator":            ["hydralazine", "minoxidil", "nitroprusside"],
    "med_statin":                 ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
                                    "lovastatin", "pitavastatin", "fluvastatin"],
    "med_other_lipid_lowering":   ["ezetimibe", "evolocumab", "alirocumab", "fenofibrate", "fenofibric",
                                    "gemfibrozil", "niacin", "cholestyramine", "colesevelam",
                                    "bempedoic", "icosapent"],
    "med_antiplatelet":           ["aspirin", "clopidogrel", "prasugrel", "ticagrelor", "dipyridamole",
                                    "cilostazol"],
    "med_anticoagulant":          ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
                                    "enoxaparin", "fondaparinux", "heparin"],  
    "med_nitrate":                ["nitroglycerin", "isosorbide"],
    "med_antiarrhythmic":         ["amiodarone", "dronedarone", "digoxin", "dofetilide", "flecainide",
                                    "propafenone", "mexiletine", "quinidine", "disopyramide", "ibutilide"],
    "med_antidiabetic_oral":      ["metformin", "glipizide", "glyburide", "glibenclamide", "gliclazide",
                                    "glimepiride", "pioglitazone", "rosiglitazone", "sitagliptin",
                                    "linagliptin", "saxagliptin", "alogliptin", "repaglinide",
                                    "nateglinide", "acarbose"],
    "med_sglt2_inhibitor":        ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"],
    "med_glp1_agonist":           ["liraglutide", "semaglutide", "dulaglutide", "exenatide",
                                    "lixisenatide", "tirzepatide"],
    "med_insulin":                ["insulin"],
}


def save_table(df, path):
    """Save a complete (patient-level or frequency) table as CSV or Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Output must end in .csv or .parquet: {path}")
    return path


def convert_csv_to_parquet(csv_path=MEDS_PATH, parquet_path=MEDS_PARQUET,
                           sep=SEP, chunksize=CSV_CHUNKSIZE):
    """Convert CSV incrementally, retaining the columns used by this script."""
    csv_path, parquet_path = Path(csv_path), Path(parquet_path)
    if parquet_path.exists() and not FORCE_RECONVERT:
        print(f"Parquet already exists; skipping conversion: {parquet_path}")
        return parquet_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Medication CSV not found: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = parquet_path.with_name(parquet_path.name + ".partial")
    temp_path.unlink(missing_ok=True)
    needed = set(KEEP_COLS) | {SUBCLASS_COL}
    reader = pd.read_csv(
        csv_path, sep=sep, quotechar='"', encoding=ENCODING,
        dtype=str, keep_default_na=False, chunksize=chunksize,
        engine="c", low_memory=False,
        usecols=lambda col: col.strip() in needed,
    )
    writer = None
    total_rows = 0
    completed = False
    try:
        for chunk in tqdm(reader, desc=f"Converting {csv_path.name}"):
            chunk.columns = chunk.columns.astype(str).str.strip()
            if writer is None:
                if PATIENT_COL not in chunk.columns:
                    raise KeyError(f"Missing patient column: {PATIENT_COL}")
                if MODE == "curated" and not ({GENERIC_COL, MEDNAME_COL} & set(chunk.columns)):
                    raise KeyError(f"Need {GENERIC_COL} or {MEDNAME_COL} for curated mode")
                if MODE == "subclass" and SUBCLASS_COL not in chunk.columns:
                    raise KeyError(f"Missing subclass column: {SUBCLASS_COL}")
                if DROP_STATUSES and STATUS_COL not in chunk.columns:
                    raise KeyError(f"Missing {STATUS_COL} required by DROP_STATUSES")
                columns = [col for col in KEEP_COLS if col in chunk.columns]
                columns += [col for col in chunk.columns if col not in columns]
                schema = pa.schema([pa.field(col, pa.string()) for col in columns])
                writer = pq.ParquetWriter(temp_path, schema, compression="snappy")
            table = pa.Table.from_pandas(
                chunk.reindex(columns=columns), schema=schema, preserve_index=False
            )
            writer.write_table(table)
            total_rows += len(chunk)
        if writer is None:
            raise ValueError(f"No rows in source CSV: {csv_path}")
        writer.close()
        writer = None
        os.replace(temp_path, parquet_path)
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed:
            temp_path.unlink(missing_ok=True)
    print(f"Converted {total_rows:,} rows to {parquet_path}")
    return parquet_path


def iter_parquet_batches(parquet_path, columns=None, batch_size=BATCH_ROWS):
    """Read selected Parquet columns, never the complete dataset at once."""
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)
    if columns is not None:
        missing = set(columns) - available
        if missing:
            raise KeyError(f"Missing columns in {parquet_path}: {sorted(missing)}")
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
        yield batch.to_pandas()


def _selected_columns(parquet_path, wanted, required=()):
    available = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = set(required) - available
    if missing:
        raise KeyError(f"Missing columns in {parquet_path}: {sorted(missing)}")
    return [c for c in wanted if c in available]


def _filtered_chunk(chunk):
    if DROP_STATUSES:
        if STATUS_COL not in chunk.columns:
            raise KeyError(f"{STATUS_COL} is required for DROP_STATUSES")
        chunk = chunk.loc[~chunk[STATUS_COL].isin(DROP_STATUSES)]
    return chunk


def valid_patients(df):
    pat = df[PATIENT_COL].fillna("").astype(str).str.strip()
    valid = pat.str.upper().ne("") & ~pat.str.upper().isin(MISSING_TOKENS)
    return pat, valid, sorted(pat[valid].unique())


def _valid_med_name(value):
    value = str(value).strip()
    return bool(value) and value.upper() not in MISSING_TOKENS


def _finalize(flag_patients, all_ids, group_names):
    """Merge flags across *all* batches, keeping patients with all-zero flags."""
    ids = sorted(all_ids)
    idx = pd.Index(ids)
    data = {PATIENT_COL: ids}
    for group in group_names:
        data[group] = idx.isin(flag_patients.get(group, set())).astype("int8")
    return pd.DataFrame(data)


def _sanitize(name):
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def _med_in_group(generic_lower, med_groups):
    return any(d in generic_lower for drugs in med_groups.values() for d in drugs)


def standalone_top_meds(freq, med_groups, min_pct=TOP_MED_MIN_PCT):
    """Standalone top meds not already covered by a curated group (original logic)."""
    top = freq[freq["pct_of_patients"].round() >= min_pct]
    groups, seen = {}, set()
    for generic in top["generic"]:
        gl = str(generic).strip().lower()
        if not gl or _med_in_group(gl, med_groups):
            continue
        slug = _sanitize(gl)
        if not slug:
            continue
        key = f"med_{slug}"
        if key not in med_groups and key not in seen:
            groups[key] = [gl]
            seen.add(key)
    return groups



def compute_med_frequencies(meds_parquet=MEDS_PARQUET):
    """Count distinct patients per generic across batches, not per-row orders.

    Returns (frequency_dataframe, all_valid_patient_ids). Patient sets are
    intentionally retained in memory, following comorbidity_flag_batch.py.
    """
    columns = _selected_columns(
        meds_parquet, [PATIENT_COL, GENERIC_COL, MEDNAME_COL, STATUS_COL],
        required=[PATIENT_COL] + ([STATUS_COL] if DROP_STATUSES else []),
    )
    if GENERIC_COL not in columns and MEDNAME_COL not in columns:
        raise KeyError(f"Need {GENERIC_COL} or {MEDNAME_COL} in {meds_parquet}")
    generic_col = GENERIC_COL if GENERIC_COL in columns else MEDNAME_COL
    all_ids = set()
    generic_patients = defaultdict(set)

    for chunk in tqdm(iter_parquet_batches(meds_parquet, columns),
                      desc="Medication-frequency batches"):
        chunk = _filtered_chunk(chunk)
        pat, valid, _ = valid_patients(chunk)
        all_ids.update(pat[valid].unique())
        gen = chunk.loc[valid, generic_col].fillna("").astype(str).str.strip()
       
        for patient, generic in zip(pat[valid], gen):
            if _valid_med_name(generic):
                generic_patients[generic].add(patient)

    counts = {generic: len(patients) for generic, patients in generic_patients.items()}
    freq = pd.DataFrame(
        [(generic, count) for generic, count in counts.items()],
        columns=["generic", "n_patients"],
    )
    freq = freq.sort_values(["n_patients", "generic"], ascending=[False, True])
    freq = freq.reset_index(drop=True)
    freq["pct_of_patients"] = (
        freq["n_patients"] / max(len(all_ids), 1) * 100
    ).round(2)
    freq.insert(0, "rank", range(1, len(freq) + 1))
    return freq, all_ids


def build_groups_curated(meds_path=MEDS_PARQUET, med_groups=None):
    """Build curated group flags with two Parquet passes (frequency + flags)."""
    meds_path = Path(meds_path)
    if meds_path.suffix.lower() == ".csv":
        meds_path = convert_csv_to_parquet(meds_path, MEDS_PARQUET)
    freq, all_ids = compute_med_frequencies(meds_path)
    save_table(freq, TOP_MED_FREQ_CSV)
    print(f"Med frequency list ({len(freq):,} generics) -> {TOP_MED_FREQ_CSV}")

    if med_groups is None:
        med_groups = dict(MED_GROUPS)
        if INCLUDE_STANDALONE_TOP_MEDS:
            extra = standalone_top_meds(freq, MED_GROUPS, TOP_MED_MIN_PCT)
            med_groups.update(extra)
            print(f"Standalone top meds (rounded pct >= {TOP_MED_MIN_PCT}%, "
                  f"not in any group): {len(extra)} added"
                  + (f" -> {', '.join(sorted(extra))}" if extra else ""))

    columns = _selected_columns(
        meds_path, [PATIENT_COL, GENERIC_COL, MEDNAME_COL, STATUS_COL],
        required=[PATIENT_COL] + ([STATUS_COL] if DROP_STATUSES else []),
    )
    terms = {group: tuple(d.lower() for d in drugs)
             for group, drugs in med_groups.items()}
    key_to_groups = {}
    flag_patients = {group: set() for group in med_groups}

    print("\nBuilding curated medication flags from Parquet batches...")
    for chunk in tqdm(iter_parquet_batches(meds_path, columns),
                      desc="Curated medication batches"):
        chunk = _filtered_chunk(chunk)
        pat, valid, _ = valid_patients(chunk)
        active = chunk.loc[valid]
        generic = (active[GENERIC_COL].fillna("").astype(str)
                   if GENERIC_COL in active.columns
                   else pd.Series("", index=active.index))
        medname = (active[MEDNAME_COL].fillna("").astype(str)
                   if MEDNAME_COL in active.columns
                   else pd.Series("", index=active.index))
        keys = (generic + " " + medname).str.lower()
        for patient, key in zip(pat[valid], keys):
            if key not in key_to_groups:
                key_to_groups[key] = tuple(
                    group for group, drugs in terms.items()
                    if any(drug in key for drug in drugs)
                )
            for group in key_to_groups[key]:
                flag_patients[group].add(patient)
    return _finalize(flag_patients, all_ids, med_groups)



def build_groups_subclass(meds_path=MEDS_PARQUET, subclass_col=SUBCLASS_COL):
    """Build one column per distinct subclass across all Parquet batches."""
    meds_path = Path(meds_path)
    if meds_path.suffix.lower() == ".csv":
        meds_path = convert_csv_to_parquet(meds_path, MEDS_PARQUET)
    columns = _selected_columns(
        meds_path, [PATIENT_COL, subclass_col, STATUS_COL],
        required=[PATIENT_COL, subclass_col] + ([STATUS_COL] if DROP_STATUSES else []),
    )
    all_ids = set()
    subclass_patients = defaultdict(set)
    print("\nBuilding pharmaceutical subclass flags from Parquet batches...")
    for chunk in tqdm(iter_parquet_batches(meds_path, columns),
                      desc="Subclass medication batches"):
        chunk = _filtered_chunk(chunk)
        pat, valid, _ = valid_patients(chunk)
        all_ids.update(pat[valid].unique())
        sub = chunk.loc[valid, subclass_col].fillna("").astype(str).str.strip()
        for patient, subclass in zip(pat[valid], sub):
            if _valid_med_name(subclass):
                subclass_patients[subclass].add(patient)
    group_names = sorted(subclass_patients)
    print(f"{len(group_names)} distinct '{subclass_col}' values -> binary columns")
    return _finalize(subclass_patients, all_ids, group_names)



def write_long_clean(meds_parquet=MEDS_PARQUET, output_path=LONG_OUT):
    """Write only KEEP_COLS, filtered by status, incrementally (CSV or Parquet)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _selected_columns(
        meds_parquet, KEEP_COLS, required=[PATIENT_COL]
        + ([STATUS_COL] if DROP_STATUSES else []),
    )
    if not columns:
        raise ValueError("No KEEP_COLS available in medication Parquet")
    temp_path = output_path.with_name(output_path.name + ".partial")
    temp_path.unlink(missing_ok=True)
    writer = None
    total_rows = 0
    completed = False
    try:
        if output_path.suffix.lower() == ".parquet":
            schema = pa.schema([pa.field(col, pa.string()) for col in columns])
            writer = pq.ParquetWriter(temp_path, schema, compression="snappy")
        elif output_path.suffix.lower() != ".csv":
            raise ValueError("LONG_OUT must end in .csv or .parquet")

        first = True
        for chunk in tqdm(iter_parquet_batches(meds_parquet, columns),
                          desc="Writing cleaned long-table batches"):
            chunk = _filtered_chunk(chunk)[columns]
            if writer is not None:
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
            else:
                chunk.to_csv(temp_path, mode="w" if first else "a",
                             header=first, index=False)
                first = False
            total_rows += len(chunk)
        if writer is not None:
            writer.close()
            writer = None
        elif first:
            pd.DataFrame(columns=columns).to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed:
            temp_path.unlink(missing_ok=True)
    print(f"Saved {total_rows:,} medication rows (kept columns) -> {output_path}")
    return output_path



def main():
    if MODE not in {"curated", "subclass"}:
        raise ValueError("MODE must be 'curated' or 'subclass'")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = convert_csv_to_parquet(MEDS_PATH, MEDS_PARQUET)
    if MODE == "curated":
        groups = build_groups_curated(parquet_path)
    else:
        groups = build_groups_subclass(parquet_path)

    write_long_clean(parquet_path, LONG_OUT)
    flag_cols = [c for c in groups.columns if c != PATIENT_COL]
    prev = groups[flag_cols].mean().mul(100).round(1).sort_values(ascending=False)
    print(f"\nGroups built for {len(groups):,} patients ({len(flag_cols)} groups). Prevalence (%):")
    print(prev.head(30).to_string())

    written = save_table(groups, GROUPS_OUT)
    print(f"\nSaved kept-columns long table -> {LONG_OUT}")
    print(f"Saved patient x med-group binary table -> {written}")


if __name__ == "__main__":
    main()
