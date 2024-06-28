#!/bin/bash
source ~/.bashrc
module purge
module load gcc/11.1.0
module load cuda/11.8
module load cudnn/8.6.0_cuda11.x
module load openmpi/4.1.1_gcc11.1.0_cuda11.8
module load anaconda/2020.11
module list
source activate py39
export PYTHONUNBUFFERED=1

date

SH_DIR=$(dirname "$0")
RUN_FILE=$SH_DIR/../ddp_darcy_train_strong.py

torchrun --standalone --nnodes=1 --nproc-per-node=7 $RUN_FILE
