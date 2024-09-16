#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-majoroslab-gpu-01

python multiclass_segment.py --dataset=cHTCO-Group5 --tissue=p --batch=2 --epochs=500
python evaluate.py --tissue=p

python multiclass_segment.py --dataset=cHTCO-Group5 --tissue=p --batch=2 --epochs=500
python evaluate.py --tissue=c