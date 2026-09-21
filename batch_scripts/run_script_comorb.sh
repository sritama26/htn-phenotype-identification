#!/bin/bash
#SBATCH --job-name=complete_files_analysis/comorbflag_job
#SBATCH --partition=p_njacts_1
#SBATCH --output=logs/complete_files_analysis/comorbflag_job_output_%j.txt
#SBATCH --error=logs/complete_files_analysis/comorbflag_job_error_%j.txt
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

unset LUA_PATH
unset LUA_CPATH

module purge 
source "$HOME/miniconda3/etc/profile.d/conda.sh" 

conda activate hyperten

cd /projectsp/f_miarc_1/Hypertension/Sritama


python /projectsp/f_miarc_1/Hypertension/Sritama/src/comorbidity_flag.py
