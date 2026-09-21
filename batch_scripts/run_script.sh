#!/bin/bash
#SBATCH --job-name=complete_files_analysis/training/training_data_job
#SBATCH --partition=p_njacts_1
#SBATCH --output=logs/complete_files_analysis/training/training_data_output_%j.txt
#SBATCH --error=logs/complete_files_analysis/training/training_data_error_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

unset LUA_PATH
unset LUA_CPATH

module purge 
source "$HOME/miniconda3/etc/profile.d/conda.sh" 

conda activate hyperten

cd /projectsp/f_miarc_1/Hypertension/Sritama

# pip install xgboost scikit-learn pandas matplotlib

# pip install sklearn_extra
# python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_kmeans_pam.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --output_dir /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median(strdzn) --k_values 2 3 4 5 6 7 8 9 10 --pam_sample_size 10000 --silhouette_sample 10000

# python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_hdbscan_hierarchical.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --output-root "/projects/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median(strdzn)/9July" --hdbscan-min-cluster-size 500 --hdbscan-min-samples 100 --hier-sample-size 5000 --hier-k-min 2 --hier-k-max 10 --umap-sample 10000 --metric-sample 10000 --jobs 16

#python /projectsp/f_miarc_1/Hypertension/Sritama/src/hf_prevalance.py --data /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training/training_data.csv --out /projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training

#python /projectsp/f_miarc_1/Hypertension/Sritama/src/model_training/join_clusters.py

python /projectsp/f_miarc_1/Hypertension/Sritama/src/model_training/train_xgb_metrics.py --data /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training/training_data.csv --out /projectsp/f_miarc_1/Hypertension/Sritama/batch_outputs/model_training