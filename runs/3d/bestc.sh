#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-03

cd 3d

python multiclass_segment.py --dataset=cHTCO-Group5Z --tissue=c --learningrate=0.00001 --batch=1 --depth=3 --dropout=0.1 --kernel=9 --epochs=1000
python evaluate.py --tissue=c

