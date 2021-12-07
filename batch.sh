#!/bin/bash

#SBATCH -n 15
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --job-name=RefSeg

module load python/3.6.8
module load cuda/11.0
module load cudnn/8-cuda-11.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONUNBUFFERED=TRUE

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --data_root /home/kanishk/smai_project/KNN_Networks/datasets/BSDS500/data/rgb/
