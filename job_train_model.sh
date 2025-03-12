#!/bin/bash 
#SBATCH --job-name=TrainModel
#SBATCH -o ./logs/TrainModel-%j.out
#SBATCH -p gpu_short
#SBATCH --gpus-per-node 1

module load Python/3.11.3-GCCcore-12.3.0
source /well/margulies/users/cpy397/python/neuro/bin/activate

echo the job id is $SLURM_JOB_ID
export HYDRA_FULL_ERROR=1

MODE=$1
MODEL=$2

echo "Mode is $MODE, model is $MODEL"

if [ "$MODE" = "cv" ] && [ "$MODEL" = "main" ]; then
    python3 -u cv_main_model.py
elif [ "$MODE" = "cv" ] && [ "$MODEL" = "mat_ae" ]; then
    python3 -u cv_mat_ae.py
elif [ "$MODE" = "shuffle" ] && [ "$MODEL" = "mat_ae" ]; then
    python3 -u shuffle_mat_ae.py
elif [ "$MODE" = "shuffle" ] && [ "$MODEL" = "main" ]; then
    python3 -u shuffle_main_model.py
fi
