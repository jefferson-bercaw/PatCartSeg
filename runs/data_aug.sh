#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-01

python multiclass_segment.py --tissue=p --batch=1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --batch=2
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --batch=4
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --batch=1
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --batch=2
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --batch=4
python evaluate.py --tissue=c
