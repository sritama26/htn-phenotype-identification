#!/usr/bin/env python3

import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MODE = "prefix"  
DATA_DIR = Path("/projects/f_miarc_1/Hypertension")
SCRIPT_DIR = Path("/projects/f_miarc_1/Hypertension/Sritama")
OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = OUTPUT_DIR / "parquet_files"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

COMORB_PATH = DATA_DIR / "Hypertension With Comorbidities 20250404.csv"
COMORB_PARQUET = PARQUET_DIR / "Hypertension_With_Comorbidities_20250404.parquet"
CCSR_PATH = SCRIPT_DIR / "DXCCSR_v2025-1.csv"
OUTPUT_PATH = OUTPUT_DIR / "patient_comorbidity_flags.csv"

ENCODING = "latin1"
SEP = ","
CSV_CHUNKSIZE = 500_000
BATCH_ROWS = 100_000
FORCE_RECONVERT = False

PATIENT_COL = "PAT_MRN_ID_ENCRYPT"
ICD10_COL = "CURRENT_ICD10_LIST"

MISSING_TOKENS = {
    "", "NAN", "NA", "N/A", "NULL", "NONE", ".", "..", "...", "?", "-",
    "UNKNOWN", "UNSPECIFIED", "*UNSPECIFIED", "*UNK", "*DELETED"
}
SPLIT_PATTERN = re.compile(r"[;,|/]+")

FLAG_PREFIXES = {
    
    "has_heart_failure": ["I50", "I11.0", "I13.0", "I13.2"],
    "has_diabetes": ["E08-E13"],
    "has_ckd": ["N18", "I12", "I13"],
    "has_ischemic_heart_disease": ["I20-I25"],
    "has_atrial_fibrillation": ["I48"],
    "has_hyperlipidemia": ["E78"],
    "has_cerebrovascular": ["I60-I69", "G45"],
    "has_pvd": ["I70", "I73"],
    "has_valvular": ["I05-I08", "I34-I39"],
    "has_cardiomyopathy": ["I42"],
    "has_obesity": ["E66"],
    "has_copd": ["J40-J47"],
    "has_osa": ["G47.3"],
    "has_tobacco_use": ["F17", "Z87.891"],
}

TOP_FREQ_CSV = OUTPUT_DIR / "icd10_3digit_patient_frequencies.csv"
TOP_MIN_PCT = 20
INCLUDE_CURATED_PREFIXES = True
TRIM_DUPLICATES = True
DUP_CORR_THRESHOLD = 0.95

FLAG_KEYWORDS = {
    "has_hypertension": ["hypertension"],
    "has_heart_failure": ["heart failure"],
    "has_diabetes": ["diabetes"],
    "has_ckd": ["chronic kidney disease"],
    "has_ischemic_heart_disease": ["coronary atherosclerosis", "myocardial infarction", "angina"],
    "has_arrhythmia_afib": ["dysrhythmia"],
    "has_hyperlipidemia": ["lipid"],
    "has_cerebrovascular": ["cerebrovascular", "transient cerebral ischemia"],
    "has_pvd": ["peripheral and visceral atherosclerosis", "aortic"],
    "has_valvular": ["heart valve", "valvular"],
    "has_cardiomyopathy": ["cardiomyopathy"],
    "has_obesity": ["obesity", "overweight"],
    "has_copd": ["chronic obstructive", "bronchiectasis"],
    "has_osa": ["sleep"],
}

_RANGE_RE = re.compile(r"^([A-Z])(\d{2})-([A-Z])(\d{2})$")


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


def save_table(df, path):
    path = Path(path)
    if str(path).lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def icd_codes_nodot(cell):
    text = str(cell).strip()
    if text.upper() in MISSING_TOKENS:
        return []
    out = []
    for tok in SPLIT_PATTERN.split(text):
        c = tok.strip().upper().replace(" ", "").replace(".", "")
        if c and c not in MISSING_TOKENS:
            out.append(c)
    return list(dict.fromkeys(out))


def expand_prefixes(specs):
    out = []
    for s in specs:
        s = str(s).strip().upper().replace(" ", "")
        m = _RANGE_RE.match(s)
        if m:
            l1, n1, _, n2 = m.groups()
            out += [f"{l1}{n:02d}" for n in range(int(n1), int(n2) + 1)]
        else:
            out.append(s.replace(".", ""))
    return set(out)


def load_top_prefixes(freq_csv=TOP_FREQ_CSV, min_pct=TOP_MIN_PCT):
    if not freq_csv or not Path(freq_csv).exists():
        print(f"  (top-prefix file '{freq_csv}' not found -> curated prefixes only)")
        return {}
    f = pd.read_csv(freq_csv, dtype=str, keep_default_na=False)
    pct = pd.to_numeric(f["pct_of_patients"], errors="coerce").round()
    code = f["icd10_code"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    keep = code[(pct >= min_pct) & code.str.match(r"^[A-Z][0-9A-Z]")]
    top = {f"has_{c}": [c] for c in dict.fromkeys(keep) if c}
    print(f"  top 3-digit comorbidities (rounded pct >= {min_pct}%): {len(top)} added"
          + (f" -> {', '.join(sorted(top))}" if top else ""))
    return top


def effective_prefixes():
    curated = dict(FLAG_PREFIXES) if INCLUDE_CURATED_PREFIXES else {}
    top = load_top_prefixes()
    merged = dict(curated)
    merged.update(top)
    return merged, set(top), set(curated)


def _trim_duplicate_curated(flags, curated_keys, auto_keys, thr=DUP_CORR_THRESHOLD):
    cur = [c for c in curated_keys if c in flags.columns]
    aut = [a for a in auto_keys if a in flags.columns]
    drops = []
    for c in cur:
        cc = flags[c].astype(float)
        for a in aut:
            aa = flags[a].astype(float)
            if cc.std() == 0 or aa.std() == 0:
                corr = 1.0 if cc.equals(aa) else 0.0
            else:
                corr = abs(cc.corr(aa))
            if corr >= thr:
                drops.append((c, a, corr))
                break
    if drops:
        print(f"Trimmed {len(drops)} curated flag(s) duplicating a top code (>= {thr} corr):")
        for c, a, corr in drops:
            print(f"  drop {c:<26} (corr {corr:.3f} with {a})")
        flags = flags.drop(columns=[c for c, _, _ in drops])
    return flags


def build_flags_prefix(comorb_parquet=COMORB_PARQUET, flag_prefixes=None):
    auto_keys, curated_keys = set(), set()
    if flag_prefixes is None:
        flag_prefixes, auto_keys, curated_keys = effective_prefixes()

    expanded = {flag: tuple(sorted(expand_prefixes(specs))) for flag, specs in flag_prefixes.items()}
    code_flag_cache = {}
    all_ids = set()
    flag_patients = {flag: set() for flag in flag_prefixes}

    print("\nBuilding comorbidity flags from Parquet batches...")
    for chunk in tqdm(iter_parquet_batches(comorb_parquet, [PATIENT_COL, ICD10_COL]), desc="Comorbidity batches"):
        pat = chunk[PATIENT_COL].fillna("").astype(str).str.strip()
        valid = pat.str.upper().ne("") & ~pat.str.upper().isin(MISSING_TOKENS)
        all_ids.update(pat[valid].unique())

        codes_series = chunk[ICD10_COL].apply(icd_codes_nodot)

        for patient, is_valid, codes in zip(pat, valid, codes_series):
            if not is_valid:
                continue
            hit_flags = set()
            for code in codes:
                if code not in code_flag_cache:
                    code_flag_cache[code] = [flag for flag, prefs in expanded.items() if code.startswith(prefs)]
                hit_flags.update(code_flag_cache[code])
            for flag in hit_flags:
                flag_patients[flag].add(patient)

    all_ids = sorted(all_ids)
    idx = pd.Index(all_ids)
    out = pd.DataFrame({PATIENT_COL: all_ids})
    for flag in flag_prefixes:
        out[flag] = idx.isin(flag_patients[flag]).astype("int8")

    if TRIM_DUPLICATES and auto_keys and curated_keys:
        out = _trim_duplicate_curated(out, curated_keys, auto_keys)

    return out


def _clean(tok):
    return str(tok).strip().strip("'").strip('"').strip()


def load_ccsr_mapping(path):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, quotechar="'", skipinitialspace=True)
    raw.columns = [_clean(c) for c in raw.columns]
    icd_col = next((c for c in raw.columns if c.upper().replace("-", "").replace(" ", "").startswith("ICD10CMCODE")), raw.columns[0])
    cat_cols = [c for c in raw.columns if re.fullmatch(r"CCSR CATEGORY \d+", c, flags=re.I)] or [c for c in raw.columns if re.fullmatch(r"CCSR CATEGORY", c, flags=re.I)]
    desc_for = {c: next((d for d in raw.columns if d.upper() == f"{c} DESCRIPTION".upper()), None) for c in cat_cols}
    raw[icd_col] = raw[icd_col].map(_clean).str.upper().str.replace(".", "", regex=False)
    frames = []
    for c in cat_cols:
        cat = raw[c].map(_clean)
        desc = raw[desc_for[c]].map(_clean) if desc_for[c] else ""
        part = pd.DataFrame({"icd10": raw[icd_col], "ccsr_category": cat, "ccsr_description": desc})
        part = part[part["ccsr_category"].str.upper().ne("") & ~part["ccsr_category"].str.upper().isin(MISSING_TOKENS)]
        frames.append(part)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def build_flags_ccsr(comorb_parquet=COMORB_PARQUET, ccsr_path=CCSR_PATH, flag_keywords=FLAG_KEYWORDS):
    mapping = load_ccsr_mapping(ccsr_path)
    cat_desc = mapping.drop_duplicates("ccsr_category")[["ccsr_category", "ccsr_description"]]

    print("Resolving flags against CCSR category descriptions:")
    flag_to_cats = {}
    for flag, kws in flag_keywords.items():
        kws_l = [k.lower() for k in kws]
        mask = cat_desc["ccsr_description"].str.lower().apply(lambda d: any(k in d for k in kws_l))
        cats = cat_desc.loc[mask, "ccsr_category"].tolist()
        flag_to_cats[flag] = set(cats)
        print(f"  {flag:<28} {', '.join(cats) if cats else '*** NO MATCH -- adjust keywords ***'}")

    code_to_cats = mapping.groupby("icd10")["ccsr_category"].apply(set).to_dict()
    all_ids = set()
    flag_patients = {flag: set() for flag in flag_keywords}

    for chunk in tqdm(iter_parquet_batches(comorb_parquet, [PATIENT_COL, ICD10_COL]), desc="CCSR flag batches"):
        pat = chunk[PATIENT_COL].fillna("").astype(str).str.strip()
        valid = pat.str.upper().ne("") & ~pat.str.upper().isin(MISSING_TOKENS)
        all_ids.update(pat[valid].unique())
        codes_series = chunk[ICD10_COL].apply(icd_codes_nodot)

        for patient, is_valid, codes in zip(pat, valid, codes_series):
            if not is_valid:
                continue
            patient_cats = set()
            for code in codes:
                patient_cats.update(code_to_cats.get(code, set()))
            for flag, cats in flag_to_cats.items():
                if patient_cats & cats:
                    flag_patients[flag].add(patient)

    all_ids = sorted(all_ids)
    idx = pd.Index(all_ids)
    out = pd.DataFrame({PATIENT_COL: all_ids})
    for flag in flag_keywords:
        out[flag] = idx.isin(flag_patients[flag]).astype("int8")
    return out


def main():
    convert_csv_to_parquet(COMORB_PATH, COMORB_PARQUET, sep=SEP, chunksize=CSV_CHUNKSIZE)

    if MODE == "prefix":
        flags = build_flags_prefix()
    elif MODE == "ccsr":
        flags = build_flags_ccsr()
    else:
        raise ValueError("MODE must be 'prefix' or 'ccsr'")

    flag_cols = [c for c in flags.columns if c.startswith("has_")]
    prev = flags[flag_cols].mean().mul(100).round(1).sort_values(ascending=False)
    print(f"\nFlags built for {len(flags):,} patients ({len(flag_cols)} flags). Prevalence (%):")
    print(prev.to_string())

    written = save_table(flags, OUTPUT_PATH)
    print(f"\nSaved -> {written}")


if __name__ == "__main__":
    main()
