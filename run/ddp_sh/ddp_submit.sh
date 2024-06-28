#!/bin/bash

sbatch --gpus=8 ddp_strong_gpu8.sh
sbatch --gpus=7 ddp_strong_gpu7.sh
sbatch --gpus=6 ddp_strong_gpu6.sh
sbatch --gpus=5 ddp_strong_gpu5.sh
sbatch --gpus=4 ddp_strong_gpu4.sh
sbatch --gpus=3 ddp_strong_gpu3.sh
sbatch --gpus=2 ddp_strong_gpu2.sh

sbatch --gpus=8 ddp_weak_gpu8.sh
sbatch --gpus=7 ddp_weak_gpu7.sh
sbatch --gpus=6 ddp_weak_gpu6.sh
sbatch --gpus=5 ddp_weak_gpu5.sh
sbatch --gpus=4 ddp_weak_gpu4.sh
sbatch --gpus=3 ddp_weak_gpu3.sh
sbatch --gpus=2 ddp_weak_gpu2.sh