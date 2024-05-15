#!/bin/bash
#
#SBATCH --job-name=test_job
#SBATCH --output=res_test_job_%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=parietal,normal,gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --error error_%A_%a.out
#
#SBATCH --array=0-935
# Load necessary modules
module load python/3.8  # adjust based on your environment
module load cuda/11.1  # adjust based on your environment

# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

# Run the Python script with the current array index
python job_graph.py $SLURM_ARRAY_TASK_ID