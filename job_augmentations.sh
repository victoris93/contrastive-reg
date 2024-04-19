#!/bin/bash
#SBATCH --job-name=aug_eval
#SBATCH --output=./res_test/job_output_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --partition=parietal,normal,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --error ./errors/error_%A_%a.out
#SBATCH --array=1-936:1

module load conda

conda activate

index=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /data/parietal/store2/work/mrenaudi/contrastive-reg-1/subjects.txt)


srun python augmentation_evaluation.py

