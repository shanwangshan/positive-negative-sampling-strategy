#!/bin/bash
#SBATCH --job-name=TAU_tt_lin
#SBATCH --account=project_2003370
#SBATCH --output=./err_out/out_tt_lin.txt
#SBATCH --error=./err_out/err_tt_lin.txt

#SBATCH --partition=gpusmall
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --gres=gpu:a100:1


source activate torch_1.11

#module load pytorch/1.11

python test_batch.py config.yaml --model_type 'audio' --lin_prob
#python test_batch.py config.yaml --model_type 'audio'

#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py config.yaml --model_type 'audio'
#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py
