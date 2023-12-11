import sys
from data_loader_mean_std import TAU
from torch.utils.data import DataLoader
sys.path.insert(0,'../')
from model import ClassificationWrapper

import models
import torch
from models.av_wrapper import av_wrapper
from criterions.contrastive import ContrastiveLoss
from criterions.amc import Amc
import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
import argparse

# python train.py config.yaml
parser = argparse.ArgumentParser(description='audio/video test self-supervised learning on vgg-sound')
parser.add_argument('cfg', help='training config file')
parser.add_argument('--model_type', type= str, required = True, help='train audio subnetwork or video')
parser.add_argument('--lin_prob', default = False, action="store_true", help='linear prob')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
print(cfg)
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

if cfg['debug']:
  cfg['batch_size'] = 1
  cfg['num_workers'] = 0



tr_Dataset = TAU(cfg,data_type="tr")
training_generator = DataLoader(tr_Dataset, batch_size = cfg['batch_size'], shuffle = True, num_workers = cfg['num_workers'], drop_last =True, pin_memory=True)

audio_all = []
loader = training_generator
for batch_idx, data in tqdm(enumerate(loader)):

    #embed()

  #import pdb; pdb.set_trace()
  batch_embed = data[0].to(device)
  audio_all.append(batch_embed.cpu().numpy())


import pdb; pdb.set_trace()
audio = np.concatenate(audio_all, 0)
audio = audio.reshape(-1, audio.shape[-1])
mu = audio.mean(0)
std = audio.std(0)
fn = './mean_std_TAU.npz'
np.savez(fn, mean=mu, std=std)
