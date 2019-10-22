#!/bin/sh
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --account=m1759
#SBATCH --job-name=more_dropouts_batch
#SBATCH --out=ensemble_batch.out
#SBATCH --error=ensemble_batch.error

# Attention:
# It is very important to add the directory where cgcnn is installed to the
# system path
export PYTHONPATH=$PYTHONPATH:/global/u2/q/qingyanz/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 2 python d15_assess_ensemble.py 0 &
srun -N 1 -G 2 python d25_assess_ensemble.py 1 &
wait