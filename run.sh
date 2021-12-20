#!/bin/bash

#SBATCH --job-name="Segmentation"
#SBATCH -D .
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=72:00:00
#SBATCH --nodelist=g[13]

source activate reinforce

python train.py
