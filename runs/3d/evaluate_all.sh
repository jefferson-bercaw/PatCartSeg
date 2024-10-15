#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -p common
#SBATCH --array=0-62

cd 3d

python evaluate.py