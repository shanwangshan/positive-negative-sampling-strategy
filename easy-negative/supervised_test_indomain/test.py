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
from utils.ioutils import av_video_loader, av_audio_loader
import av
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
#model = torch.nn.DataParallel(model)
fpath = './audio_model/model.pt'

checkpoint = torch.load(fpath,map_location='cpu')
model.load_state_dict(checkpoint)
#model_path = '../../AV-ACL1/test/audio_model/model.pt'

# ckp = torch.load(checkpoint_fn, map_location='cpu')
# model_ckp = ckp['state_dict'] if 'state_dict' in ckp else ckp['model']
# model.load_state_dict({k.replace('module.', ''): model_ckp[k] for k in model_ckp})
#model.load_state_dict(torch.load(model_path))
print(model)

########### use GPU ##########
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
########### use GPU ##########
model.to(device)


# audio_transforms1 =T.MelSpectrogram(sample_rate=cfg['audio_fps'],n_fft = cfg['n_fft'], hop_length = int(1/cfg['spectrogram_fps']*cfg['audio_fps']) , n_mels = cfg['n_mels'])
# audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

fn_softmax = torch.nn.Softmax(dim=1)
ground_tr_list = []
esti_list=[]
audio_mean_std = np.load('./mean_std_VGGsound.npz')
audio_mean = audio_mean_std['mean']
audio_std = audio_mean_std['std']

#import pdb; pdb.set_trace()
path = os.environ["LOCAL_SCRATCH"]+ cfg['vgg_path']
vggsound =pd.read_csv(path + 'vggsound-1.csv',header=None)
unwanted = pd.read_csv(cfg['unwanted_files_path'],header = None)
unwanted = unwanted[0].values.tolist()

vggsound = vggsound[~vggsound[0].isin(unwanted)]

train_files = vggsound.loc[vggsound[3] == 'test']
classes = sorted(list(set(train_files[2].values.tolist())))

all_files = train_files.values.tolist()

#all_files = np.random.choice(df['filename_audio'].values.tolist(),size=100, replace=False,)

for i in tqdm(range(len(all_files))):


        filename = os.path.join(path,'video/'+all_files[i][0]+'_'+str(all_files[i][1]*1000)+'_'+str(all_files[i][1]*1000+10000)+'.mp4')
        #filename = os.path.join(self.path,self.train_files[index])

        video_ctr = av.open(filename)

        # video_stream = video_ctr.streams.video[0]
        # dur = video_stream.duration * video_stream.time_base


        audio_dt, audio_fps = av_audio_loader(
            container=video_ctr,
            rate=cfg['audio_fps'],
            start_time=0,
            duration=10
        )
        audio = audio_dt.mean(axis=0)
        mels = librosa.feature.melspectrogram(y=audio, sr=audio_fps, n_fft=cfg['n_fft'], hop_length=int(1/cfg['spectrogram_fps']*cfg['audio_fps']), power=1.0)
        mels = librosa.core.power_to_db(mels, top_db=100)
        mels = torch.tensor(mels)
        mels = mels[None,:,:]
        audio_tensor = (mels-audio_mean[:,np.newaxis])/(audio_std[:,np.newaxis]+1e-5)
        audio_fea = torch.transpose(audio_tensor, 1, 2).to(device)
        audio_fea = audio_fea[None,:,:,:]

        ground_tr = classes.index(all_files[i][2])
        ground_tr_list.append(ground_tr)

        model.eval()
        with torch.no_grad():
          esti_label = model(audio_fea)
          esti_label = fn_softmax(esti_label)

        es_class = torch.argmax(esti_label)
        esti_list.append(es_class.cpu().numpy())

y_true = np.array(ground_tr_list)
y_pred = np.array(esti_list)

# from sklearn.metrics import classification_report
# print(classification_report(y_true, y_pred, labels= classes))
acc = accuracy_score(y_true, y_pred)
acc = np.round(acc,decimals = 3)
print('test accuracy is',acc)

# from sklearn.metrics import ConfusionMatrixDisplay

# IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
# cm=ConfusionMatrixDisplay.from_estimator(IC, y_pred, y_true, normalize='true',  values_format='.2%')
# cm.figure_.savefig('confusion_matrix.png')
