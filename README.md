# Hypertension Patient Clustering Pipeline

## How to reproduce the results (start here)

The project has many scripts, but only **one script per step** is needed to get the final results.
Run the steps below **in order**. Each step reads the files written by the step before it, so do not skip ahead.
Every step is started with one `sbatch` command (a job submitted to the Amarel Slurm cluster).

### Before you start (Step 0)

1. **Environment:** a conda environment named `hyperten` with `pandas`, `numpy`, `pyarrow`, `scikit-learn`,
   `scanpy`, `leidenalg`, `umap-learn`, `xgboost` and `matplotlib`.
2. **Raw input files**, placed in `/projects/f_miarc_1/Hypertension/`:
   - `Hypertension With Comorbidities 20250404.csv`
   - `Hypertension Labs 20250417.csv`
   - `Hypertension Medications 20250417.csv`
   - `Hypertension Vitals 20250520.csv`
3. **ICD-10 to CCSR mapping file** `DXCCSR_v2025-1.csv`, placed in `/projects/f_miarc_1/Hypertension/Sritama/`.
4. **Paths:** input and output paths are written at the top of each script (and in each `batch_scripts/*.sh`).
   If your copy of the data lives somewhere else, update those paths first.
   All outputs go to `/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/`.

### The steps

| Step | What it does (in plain words) | Command | Main output (in `batch_outputs/`) |
|---|---|---|---|
| **1** | Count diagnosis codes per patient | `sbatch batch_scripts/run_script_diagnoses.sh` | `icd10_3digit_patient_frequencies.csv` |
| **2** | Flag each patient's comorbidities (needs Step 1) | `sbatch batch_scripts/run_script_comorb.sh` | `patient_comorbidity_flags.csv` |
| **3** | Group each patient's medications | `sbatch batch_scripts/run_script_meds.sh` | `patient_med_groups_new.csv` (see note A) |
| **4** | Clean lab results | `sbatch batch_scripts/run_script_labs.sh` | `labs_by_patient.csv` |
| **5** | Clean vital signs | `sbatch batch_scripts/run_script_vitals.sh` | `vitals_by_patient.csv` |
| **6** | Keep only labs taken within 15 days of the hypertension diagnosis | `sbatch batch_scripts/run_script_labsxdiag.sh` | `lab_within_15_days_of_diagnosis.csv` |
| **7** | Keep only vitals taken within 15 days of the hypertension diagnosis | `sbatch batch_scripts/run_script_vitalsxdiag.sh` | `vitals_within_15_days_of_diagnosis.csv` |
| **8** | Convert lab units, one row per patient, fill missing values with the median | `sbatch batch_scripts/run_script_labs_filter.sh` | `filtered_labs/patient_lab_values_wide_median_imputed.csv` |
| **9** | Same as Step 8, for vitals | `sbatch batch_scripts/run_script_vitals_filter.sh` | `filtered_vitals/patient_vital_values_wide_median_imputed.csv` |
| **10** | Merge everything into one table (one row per patient) | `sbatch batch_scripts/run_script_master.sh` | `master_file/master.csv` |
| **11** | Prepare the merged table for clustering (scaling, encoding) | `sbatch batch_scripts/run_script_features_cluster.sh` | `master_file/master_clustering_ready.csv` |
| **12** | Group patients into clusters (Leiden, see note B) | `sbatch batch_scripts/run_script_experiements.sh` | `clustering_median_strdzn/phenograph/scanpy_leiden_k30_res0p4_cosine/patient_clusters.csv` |
| **13** | Attach each patient's cluster to their features (see note C) | `sbatch batch_scripts/run_script.sh` | `model_training/training_data.csv` |
| **14** | Heart-failure rate in each cluster (see note C) | `sbatch batch_scripts/run_script.sh` | `model_training/hf_prevalence_by_cluster.csv` |
| **15** | Train XGBoost to predict the cluster, and report accuracy and important features | `sbatch batch_scripts/run_script.sh` | `model_training/confusion_matrix.csv`, `per_class_auc.csv`, `feature_importance.csv` |

Steps 1 to 5 do not depend on each other except that **Step 2 needs Step 1**, so Steps 1, 3, 4 and 5 can be submitted together.
Wait for each job to finish (check with `squeue -u $USER`, and the log files in `logs/`) before starting a step that needs its output.

### Notes (please read before running)

- **A. Medication file name.** `meds.py` writes `patient_med_groups_new.csv`, but Step 10 reads
  `patient_med_groups.csv`. After Step 3, rename the file:
  `mv batch_outputs/patient_med_groups_new.csv batch_outputs/patient_med_groups.csv`
- **B. Step 12 runs the final clustering method.** `run_script_experiements.sh` holds one line per clustering
  method that was tried. To reproduce the reported clusters, make the `run_leiden_with_metrics.py` line
  (`--method scanpy_leiden --n-pcs 19 --n-neighbors 30 --resolution 0.4 --metric cosine ...`) the only
  uncommented `python` line, and comment out the `run_affinity_propagation.py` line.
- **C. Steps 13 to 15 share one batch script.** `run_script.sh` holds the commands for all three steps.
  Uncomment one `python` line at a time, in this order, and submit once for each:
  1. Step 13: `python src/model_training/join_clusters.py --out-name training_data.csv`
     (by default it writes `training_data_pam.csv`, but Steps 14 and 15 read `training_data.csv`).
  2. Step 14: `python src/hf_prevalance.py --data batch_outputs/model_training/training_data.csv --out batch_outputs/model_training/hf_prevalence_by_cluster.csv`
     (`--out` must be a file name, not a folder).
  3. Step 15: the `train_xgb_metrics.py` line, which is already uncommented.
- **Optional scripts** are not needed for the main results: `data_info.py` and `data_overview.py` (raw data summaries),
  `pca.py` and `correlation/` (feature exploration), `all_labs_x_days_diagnosis.py` (an older version of Step 6),
  and the other clustering methods in `clustering/` (methods we compared against).

## Project layout

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
