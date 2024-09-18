#!/bin/sh

#SBATCH --output=./outputs/%A_%a_slurm.out
#SBATCH --error=./errors/%A_%a_slurm.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodelist=dcc-allenlab-gpu-08

cd 2d

python multiclass_segment.py --tissue=p --depth=3
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --depth=4
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --depth=5
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=p --depth=6
python evaluate.py --tissue=p

python multiclass_segment.py --tissue=c --depth=3
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --depth=4
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --depth=5
python evaluate.py --tissue=c

python multiclass_segment.py --tissue=c --depth=6
python evaluate.py --tissue=c