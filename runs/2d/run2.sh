#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-yaolab-gpu-08

cd 2d

python multiclass_segment.py --dataset=cHTO5 --tissue=c --learningrate=0.001 --batch=16 --depth=5 --dropout=0.4 --kernel=9 --epochs=1000
python evaluate.py