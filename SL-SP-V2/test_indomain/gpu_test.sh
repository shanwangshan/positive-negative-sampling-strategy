#!/bin/bash
#SBATCH --job-name=real_ssl_classi_norm_indoamin_tt_ft
#SBATCH --account=project_2003370
#SBATCH --output=./err_out/tt_out_ft.txt
#SBATCH --error=./err_out/tt_err_ft.txt

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


#module load pytorch/1.11
#python test_batch.py config.yaml --model_type 'audio' --lin_prob

python test_batch.py config.yaml --model_type 'audio'

#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py config.yaml --model_type 'audio'
#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py
