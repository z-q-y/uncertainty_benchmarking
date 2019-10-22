#!/bin/sh
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --account=m1759
#SBATCH --job-name=pfgp_batch
#SBATCH --out=pfgp_batch.out
#SBATCH --error=pfgp_batch.error

export PYTHONPATH=$PYTHONPATH:/global/u2/q/qingyanz/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 1 python pfgp.py 0 &
srun -N 1 -G 1 python d_pfgp.py 1 &
wait