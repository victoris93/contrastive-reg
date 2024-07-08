#!/bin/bash 
#SBATCH --job-name=LCurve
#SBATCH -o ./logs/LCurve-%j.out
#SBATCH --account=ftj@a100
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --time=60


module purge
conda deactivate

module load cpuarch/amd
module load pytorch-gpu/py3/2.0.0
set -x

python -u lc_multivariate_2.py
