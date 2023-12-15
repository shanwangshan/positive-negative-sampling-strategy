import sys
from torch.utils.data import DataLoader
from model import ClassificationWrapper
sys.path.insert(0,'../')
import models
import torch
from data_loader_test import FSD
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
import pandas as pd
from sklearn.metrics import accuracy_score
#import torchaudio.transforms as T
import librosa
parser = argparse.ArgumentParser(description='audio/video test self-supervised learning on vgg-sound')
parser.add_argument('cfg', help='training config file')
parser.add_argument('--model_type', type= str, required = True, help='train audio subnetwork or video')
parser.add_argument('--training_type', type= str, required = True, help='train all noisy noisy_small clean')
parser.add_argument('--lin_prob', default = False, action="store_true", help='linear prob')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
cfg['use_dropout'] = False
cfg['batch_size']= 1
print(cfg)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

tt_Dataset = FSD(path_features=cfg['data_path'], training_type=args.training_type)
tt_generator = DataLoader(tt_Dataset, batch_size = cfg['batch_size'], shuffle = False, num_workers = cfg['num_workers'], drop_last = False, pin_memory=True)

pretrained_net = av_wrapper(proj_dim=None)
if args.model_type=='video':
  pretrained_vnet = pretrained_net.video_model
  if args.lin_prob:
    fpath = './video_model_lin/model.pt'
  else:
    fpath = './video_model_ft/model.pt'


else:
  pretrained_vnet = pretrained_net.audio_model
  if args.lin_prob:
    fpath = './audio_model_lin_'+args.training_type+'/'+'model.pt'
  else:
    fpath = './audio_model_ft_'+args.training_type+'/'+'model.pt'


model = ClassificationWrapper(feature_extractor=pretrained_vnet, n_classes = cfg['num_classes'],feat_name = cfg['feat_name'],feat_dim=cfg['feat_dim'],pooling_op=cfg['pooling_op'],use_dropout=cfg['use_dropout'],dropout=cfg['dropout'])

model.load_state_dict(torch.load(fpath,map_location='cpu'))
print(model)
model.to(device)


fn_softmax = torch.nn.Softmax(dim=1)
ground_tr_list = []
esti_list=[]

model.eval()
for batch_idx, data in tqdm(enumerate(tt_generator)):
  #import pdb; pdb.set_trace()

  batch_embed = data[0].to(device)
  batch_label = data[1].to(device)
  ground_tr_list.append(batch_label.cpu().numpy())
  with torch.no_grad():
    esti_label = model(batch_embed)
    esti_label = fn_softmax(esti_label)

  es_class = torch.argmax(esti_label,dim=1)
  esti_list.append(es_class.cpu().numpy())
#import pdb; pdb.set_trace()

y_true = np.reshape(np.array(ground_tr_list),(-1))

y_pred = np.reshape(np.array(esti_list),(-1))

acc = accuracy_score(y_true, y_pred)
acc = np.round(acc,decimals = 3)
print('model type is', args.model_type, 'linear prob is ', args.lin_prob,'training_type is', args.training_type ,'test accuracy is',acc)
