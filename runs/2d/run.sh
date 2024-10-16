#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-01

cd 2d

python multiclass_segment.py --dataset=cHTO-Group5 --tissue=p --learningrate=0.01 --batch=32 --depth=5 --dropout=0.1 --kernel=5 --epochs=1000
python evaluate.py
