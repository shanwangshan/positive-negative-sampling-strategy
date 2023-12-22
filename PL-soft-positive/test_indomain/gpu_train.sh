#!/bin/bash
#SBATCH --job-name=real_ssl_classi_norm_indoamin_lin
#SBATCH --account=project_2003370
#SBATCH --output=./err_out/out_lin.txt
#SBATCH --error=./err_out/err_lin.txt

#SBATCH --partition=gpusmall
#SBATCH --time=0-36:00:00

##SBATCH --partition=gputest
##SBATCH --time=0-00:15:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:a100:4

#SBATCH --gres=gpu:a100:1,nvme:400
##SBATCH --gres=gpu:a100:1,nvme:400
##SBATCH --gres=gpu:a100:4


### to tar a data folder, use "tar -cvf vgg-sound.tar ./vgg-sound"

tar -xf ../../../vgg-sound.tar -C $LOCAL_SCRATCH/


echo ${LOCAL_SCRATCH}

source activate torch_1.11


python train.py config.yaml --model_type 'audio' --lin_prob
#python train.py config.yaml --model_type 'audio'
