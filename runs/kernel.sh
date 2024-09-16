#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-11

python multiclass_segment.py --tissue=p --kernel=3 --batch=1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --kernel=5 --batch=1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --kernel=7 --batch=1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --kernel=9 --batch=1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --kernel=3 --batch=1
python evaluate.py --tissue=c
