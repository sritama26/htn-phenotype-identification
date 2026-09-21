#!/bin/bash
#SBATCH --job-name=complete_files_analysis/clustering/affinity_metrics_job
#SBATCH --partition=p_njacts_1
#SBATCH --output=logs/complete_files_analysis/clustering/affinity__metrics_output_%j.txt
#SBATCH --error=logs/complete_files_analysis/clustering/affinity__metrics_error_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

unset LUA_PATH
unset LUA_CPATH

module purge 
source "$HOME/miniconda3/etc/profile.d/conda.sh" 

conda activate hyperten


cd /projectsp/f_miarc_1/Hypertension/Sritama

# python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_kmeans_pam.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --output_dir /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median_strdzn/forced_k/k7 --selected_k 7 --k_values 2 3 4 5 6 7 8 9 10 --pam_sample_size 10000 --silhouette_sample 10000 --umap_sample 30000 --umap_n_neighbors 50 --umap_min_dist 0.10


# python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_hdbscan_hierarchical.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --output-root /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/clustering_median_strdzn/forced_k/hdbscan_mcs500_ms5 --only hdbscan --selected-k 5 --hdbscan-min-cluster-size 500 --hdbscan-min-samples 5 --umap-sample 10000


#python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_leiden.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --method phenograph --n-pcs 19 --n-neighbors 30 --resolution 0.4 --metric euclidean --continuous-already-scaled

# python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_leiden_with_metrics.py --input /projects/f_miarc_1/Hypertension/Sritama/batch_outputs/master_file/master_clustering_ready.csv --method scanpy_leiden --n-pcs 19 --n-neighbors 30 --resolution 0.4 --metric cosine --continuous-already-scaled --run-umap --no-drop-procedural --stability-runs 5 --evaluation-sample-size 10000

python /projectsp/f_miarc_1/Hypertension/Sritama/src/clustering/run_affinity_propagation.py