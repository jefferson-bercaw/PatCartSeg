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

python multiclass_segment.py --tissue=p --dropout=0.1
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --dropout=0.2
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --dropout=0.3
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --dropout=0.4
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --dropout=0.1
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --dropout=0.2
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --dropout=0.3
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --dropout=0.4
python evaluate.py --tissue=c