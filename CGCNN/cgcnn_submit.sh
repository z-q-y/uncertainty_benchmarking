#!/bin/sh
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --account=m1759
#SBATCH --job-name=cgcnn_batch
#SBATCH --out=cgcnn_batch.out
#SBATCH --error=cgcnn_batch.error

# Attention
# 1. v_cgcnn.py was once named cgcnn.py
#    This caused a naming conflict with the cgcnn module invoked in that file
# 2. It is very important to add the directory where cgcnn is installed to the
#    system path

export PYTHONPATH=$PYTHONPATH:$HOME/cgcnn

source /global/homes/k/ktran/miniconda3/bin/activate
conda activate gaspy

srun -N 1 -G 1 python v_cgcnn.py 0 &
srun -N 1 -G 1 python d20_cgcnn.py 1 &
wait