#!/bin/sh

NODE_NAME=$1

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=$NODE_NAME

python multiclass_segment.py --tissue=p --epochs=500
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --epochs=1000
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --epochs=2000
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --epochs=500
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --epochs=1000
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --epochs=2000
python evaluate.py --tissue=c