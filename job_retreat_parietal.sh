#!/bin/bash 
#SBATCH --job-name=retreatPar
#SBATCH -o ./logs/retreatPar-%j.out
#SBATCH -p gpu_short
#SBATCH --constraint="skl-compat"
#SBATCH --gres=gpu:1
#SBATCH --array 1-5:1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
### source /well/margulies/users/mnk884/python/corrmats-skylake/bin/activate
echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 


RATIOS=./train_ratios.txt
train_ratio=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $RATIOS)

python -u retreat_parietal.py $train_ratio
