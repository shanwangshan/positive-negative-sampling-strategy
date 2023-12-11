#!/bin/bash
#SBATCH --job-name=random_norm_indoamin_ft
#SBATCH --output=./err_out/out_tr_v_ft.txt
#SBATCH --error=./err_out/err_tr_v_ft.txt

#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --gres=gpu:2
echo $PATH

export PATH="/home/wang9/anaconda3/bin:$PATH"
source activate torch_1.11
echo $PATH
#module load pytorch/1.11
#python train.py config.yaml --model_type 'audio' --lin_prob
#python train.py config.yaml --model_type 'audio'

#python train.py config.yaml --model_type 'video' --lin_prob
python train.py config.yaml --model_type 'video'

#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py config.yaml --model_type 'audio'
#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py
