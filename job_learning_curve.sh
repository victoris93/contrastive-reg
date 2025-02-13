#!/bin/bash 
#SBATCH --job-name=LCurve
#SBATCH -o ./logs/LCurve-%j.out
#SBATCH -p gpu_long
#SBATCH --gpus-per-node 1

module load Python/3.11.3-GCCcore-12.3.0
source /well/margulies/users/cpy397/python/neuro/bin/activate

echo the job id is $SLURM_JOB_ID
export HYDRA_FULL_ERROR=1

python3 -u lc_multivariate_abcd.py
