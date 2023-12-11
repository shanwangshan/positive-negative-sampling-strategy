#!/bin/bash
#SBATCH --job-name=random_norm_fsd_ft
#SBATCH --output=./err_out/out_tr_clean_lin.txt
#SBATCH --error=./err_out/err_tr_clean_lin.txt

#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
echo $PATH

export PATH="/home/wang9/anaconda3/bin:$PATH"
source activate torch_1.11
echo $PATH
#module load pytorch/1.11
python train.py config.yaml --model_type 'audio' --training_type 'clean' --lin_prob
#python train.py config.yaml --model_type 'audio' --training_type 'clean'
