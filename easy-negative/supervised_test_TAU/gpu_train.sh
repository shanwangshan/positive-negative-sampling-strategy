#!/bin/bash
#SBATCH --job-name=supervised_outdomain
#SBATCH --account=project_2003370
#SBATCH --output=./err_out/out_tr.txt
#SBATCH --error=./err_out/err.txt

#SBATCH --partition=gpusmall
#SBATCH --time=0-36:00:00

##SBATCH --partition=gputest
##SBATCH --time=0-00:15:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:a100:1

#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:a100:4


#tar -xf ../../../TAU-urban-audio-visual-scenes-2021-development.tar -C $LOCAL_SCRATCH/


#echo ${LOCAL_SCRATCH}

source activate torch_1.11
#module load pytorch/1.11
python train.py config.yaml
#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py
