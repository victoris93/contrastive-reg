#!/bin/bash 
#SBATCH --job-name=LCurve
#SBATCH -o ./logs/LCurve-%j.out
#SBATCH -p gpu_long
#SBATCH --gpus-per-node 1

module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/cpy397/env/ClinicalGrads/bin/activate
echo the job id is $SLURM_JOB_ID

<<<<<<< HEAD
python3 -u lc_multivariate_abcd.py
=======
python3 -u lc_gcn.py
>>>>>>> 81ae6870e953708552f37f31ae8d8538af419b48
