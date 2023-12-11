import sys
from torch.utils.data import DataLoader
from model import ClassificationWrapper
sys.path.insert(0,'../')
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
import pandas as pd
from sklearn.metrics import accuracy_score
import torchaudio.transforms as T
import librosa
parser = argparse.ArgumentParser(description='audio/video test self-supervised learning on vgg-sound')
parser.add_argument('cfg', help='training config file')

args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
print(cfg)

pretrained_net = av_wrapper(proj_dim=None)
if cfg['model_type']=='video':
  pretrained_vnet = pretrained_net.video_model
else:
  pretrained_vnet = pretrained_net.audio_model


model = ClassificationWrapper(feature_extractor=pretrained_vnet, n_classes = cfg['num_classes'],feat_name = cfg['feat_name'],feat_dim=cfg['feat_dim'],pooling_op=cfg['pooling_op'],use_dropout=cfg['use_dropout'],dropout=cfg['dropout'])

model_path = './audio_model/model.pt'
#model_path = '../../AV-ACL1/test/audio_model/model.pt'
model.load_state_dict(torch.load(model_path))
print(model)

########### use GPU ##########
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
########### use GPU ##########
model.to(device)
audio_mean_std = np.load('./mean_std_TAU.npz')
audio_mean = audio_mean_std['mean']
audio_std = audio_mean_std['std']



# audio_transforms1 =T.MelSpectrogram(sample_rate=cfg['audio_fps'],n_fft = cfg['n_fft'], hop_length = int(1/cfg['spectrogram_fps']*cfg['audio_fps']) , n_mels = cfg['n_mels'])
# audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

fn_softmax = torch.nn.Softmax(dim=1)
ground_tr_list = []
esti_list=[]

#import pdb; pdb.set_trace()
df = pd.read_csv(cfg['data_path']+ 'evaluation_setup/fold1_evaluate.csv',sep = '\t')
classes = sorted(list(set(df['scene_label'].values.tolist())))

all_files = df['filename_audio'].values.tolist()
#all_files = np.random.choice(df['filename_audio'].values.tolist(),size=100, replace=False,)

for i in tqdm(range(len(all_files))):
    ground_tr = classes.index(all_files[i].split('/')[1].split('-')[0])
    ground_tr_list.append(ground_tr)
    filename = os.path.join(cfg['data_path'],all_files[i])

    audio_dt,fs = librosa.load(filename,offset = 0,sr = cfg['audio_fps'],mono = True,duration = cfg['audio_dur'] )
    mels = librosa.feature.melspectrogram(y=audio_dt, sr=fs, n_fft=cfg['n_fft'], hop_length=int(1/cfg['spectrogram_fps']*cfg['audio_fps']), power=1.0,n_mels=cfg['n_mels'])
    mels = librosa.core.power_to_db(mels, top_db=100)
    mels = torch.tensor(mels)
    mels =mels[None,:,:]

    mels = (mels-audio_mean[:,np.newaxis])/(audio_std[:,np.newaxis]+1e-5)
    mels = torch.transpose(mels, 1, 2).to(device)

    audio_fea = mels[None,:,:,:]
    model.eval()
    with torch.no_grad():
        esti_label = model(audio_fea)
        esti_label = fn_softmax(esti_label)

    es_class = torch.argmax(esti_label)
    esti_list.append(es_class.cpu().numpy())

y_true = np.array(ground_tr_list)
y_pred = np.array(esti_list)
acc = accuracy_score(y_true, y_pred)
acc = np.round(acc,decimals = 3)
print('test accuracy is',acc)
