#!/bin/bash 
#SBATCH --job-name=ContrastiveMLP
#SBATCH -o ./logs/ContrastiveMLP-%j.out
#SBATCH -p gpu_short
#SBATCH --gpus-per-node 1
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate

echo the job id is $SLURM_JOB_ID
python3 -u train_test_cont_mlp.py