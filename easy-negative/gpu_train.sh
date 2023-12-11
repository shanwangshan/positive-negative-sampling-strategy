#!/bin/bash
#SBATCH --job-name=negative_norm
#SBATCH --account=project_2003370
#SBATCH --output=./err_out/out.txt
#SBATCH --error=./err_out/err.txt

#SBATCH --partition=gpumedium
#SBATCH --time=0-20:00:00

##SBATCH --partition=gputest
##SBATCH --time=0-00:15:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:a100:4

#SBATCH --gres=gpu:a100:4,nvme:400
##SBATCH --gres=gpu:a100:4


### to tar a data folder, use "tar -cvf vgg-sound.tar ./vgg-sound"

tar -xf ../../vgg-sound.tar -C $LOCAL_SCRATCH/


echo ${LOCAL_SCRATCH}

source activate torch_1.11
#module load pytorch/1.11
srun python train.py config.yaml
#/scratch/project_2003370/shanshan/anaconda/envs/torch_1.11/bin/python train.py
