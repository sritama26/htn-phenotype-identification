# Hypertension Patient Clustering Pipeline



```
src/            Python scripts (data prep, clustering, modelling)
batch_scripts/  Slurm .sh wrappers, one per script (or group of scripts)
```

## `src/`

### Data extraction (raw files -> per-patient tables)

| Script | What it does |
|---|---|
| `diagnoses.py` | Counts patients per 3-digit ICD-10 code (`icd10_3digit_patient_frequencies.csv`). |
| `comorbidity_flag.py` | Builds per-patient comorbidity flags (`patient_comorbidity_flags.csv`). |
| `meds.py` | Processes the medications file into per-patient features. |
| `lab_timeseries.py` | Cleans the labs file into long and per-patient tables. |
| `vitals.py` | Cleans the vitals file into per-patient tables. |

### Linking labs/vitals to diagnosis date

| Script | What it does |
|---|---|
| `labs_x_diag.py` | Keeps labs within 15 days of diagnosis (`lab_within_15_days_of_diagnosis.csv`). |
| `vitals_x_diag.py` | Keeps vitals within 15 days of diagnosis (`vitals_within_15_days_of_diagnosis.csv`). |
| `lab_filtered.py` | Converts lab units, pivots to one row per patient, median-imputes. Output in `filtered_labs/`. |
| `vital_filtered.py` | Cleans vitals, pivots to one row per patient, median-imputes. Output in `filtered_vitals/`. |

### Master file and feature prep

| Script | What it does |
|---|---|
| `build_patient_master_file.py` | Merges all per-patient tables into `master_file/master.csv`. |
| `features_cluster.py` | Turns the master file into `master_clustering_ready.csv`. |
| `correlation/point_bis_correlation.py` | Point-biserial correlation between features (continuous vs binary). |
| `correlation/heterogeneous_correlation.py` | Correlation across mixed feature types. |
| `pca.py` | PCA on the clustering-ready file |

### Clustering (`clustering/`)

All of these read `master_clustering_ready.csv`. Run with `--help` for arguments.

| Script | Method |
|---|---|
| `run_kmeans_pam.py` | K-means and PAM (Gower distance), over a range of `k`. |
| `run_hdbscan_hierarchical.py` | HDBSCAN and hierarchical clustering, with UMAP. |
| `run_leiden.py` | Scanpy Leiden |
| `run_leiden_with_metrics.py` | Same as `run_leiden.py`, plus stability and evaluation metrics. |
| `run_affinity_propagation.py` | Affinity propagation. |

### Modelling (`model_training/`, `hf_prevalance.py`)

| Script | What it does |
|---|---|
| `model_training/join_clusters.py` | Joins cluster labels onto the master file to make the training data. |
| `hf_prevalance.py` | Heart-failure prevalence per cluster. |
| `model_training/train_xgb_metrics.py` | Trains XGBoost to predict cluster labels. Writes confusion matrix, per-class AUC and feature importance. |

### Exploration

`data_info.py` and `data_overview.py` produce summaries of the raw files (missingness, category counts, age checks).

## `batch_scripts/`

Each script sets up Slurm, activates the environment, and runs the Python script:

- Partition `p_njacts_1`, 16 CPUs, 128 GB memory, 3 to 12 hours.
- `conda activate hyperten`, then `cd` into the project directory.
- Logs go to `logs/complete_files_analysis/...`.

Submit with `sbatch batch_scripts/<script>.sh`.

| Batch script | Runs |
|---|---|
| `run_script_diagnoses.sh` | `diagnoses.py` |
| `run_script_comorb.sh` | `comorbidity_flag.py` |
| `run_script_meds.sh` | `meds.py` |
| `run_script_labs.sh` | `lab_timeseries.py` |
| `run_script_vitals.sh` | `vitals.py` |
| `run_script_labsxdiag.sh` | `labs_x_diag.py` |
| `run_script_vitalsxdiag.sh` | `vitals_x_diag.py` |
| `run_script_labs_filter.sh` | `lab_filtered.py` |
| `run_script_vitals_filter.sh` | `vital_filtered.py` |
| `run_script_master.sh` | `build_patient_master_file.py` |
| `run_script_features_cluster.sh` | `features_cluster.py` |
| `run_script_experiements.sh` | `clustering/run_affinity_propagation.py` (other clustering ) |
| `run_script.sh` | `model_training/train_xgb_metrics.py` (clustering, `hf_prevalance.py` and `join_clusters.py`) |

Several scripts contain commented-out `python` lines for earlier runs. Uncomment the one you want and comment out the others.

## Typical run order

1. `diagnoses`, `comorb`, `meds`, `labs`, `vitals`
2. `labsxdiag`, `vitalsxdiag`, then `labs_filter`, `vitals_filter`
3. `master`
4. `features_cluster`
5. Clustering (`run_script_experiements.sh`)
6. `join_clusters`, then model training (`run_script.sh`)
