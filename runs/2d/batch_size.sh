#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-04

cd 2d

python multiclass_segment.py --tissue=p --batch=16
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --batch=32
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --batch=64
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --batch=16
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --batch=32
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --batch=64
python evaluate.py --tissue=c
