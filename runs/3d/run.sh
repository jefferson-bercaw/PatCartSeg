#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-10

cd 3d

python multiclass_segment.py --dataset=cHTCO-Group5 --tissue=c
python evaluate.py --tissue=c

python multiclass_segment.py --dataset=cHTCO-Group5Z --tissue=c
python evaluate.py --tissue=c