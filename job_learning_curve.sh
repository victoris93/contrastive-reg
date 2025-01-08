#!/bin/bash 
#SBATCH --job-name=LCurve
#SBATCH -o ./logs/LCurve-%j.out
#SBATCH -p gpu
#SBATCH --gpus-per-node 1

conda activate neuro
echo the job id is $SLURM_JOB_ID

export HYDRA_FULL_ERROR=1
python3 -u lc_multivariate_abcd.py
