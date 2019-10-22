#!/bin/sh
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --account=m1759
#SBATCH --job-name=ensemble_batch
#SBATCH --out=ensemble_batch.out
#SBATCH --error=ensemble_batch.error

export PYTHONPATH=$PYTHONPATH:$HOME/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 2 python assess_ensemble.py 0 &
srun -N 1 -G 2 python d20_assess_ensemble.py 1 &
wait