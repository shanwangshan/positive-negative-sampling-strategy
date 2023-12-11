import sys
import path
directory = path.Path(__name__).abspath()

# setting path
sys.path.append(directory.parent.parent)
#sys.path.insert(0,'../')
#sys.path.append('../')
from utils.ioutils import av_video_loader, av_audio_loader
import av
import pandas as pd
import os
from tqdm import tqdm
from torchvision.transforms import transforms
import librosa
import numpy as np
import torch
from PIL import Image
import random
from torch.utils import data
import torchaudio
import torchaudio.transforms as T

class TAU(data.Dataset):

    def __init__(self,cfg,model_type):
        super(TAU,self).__init__()
        self.cfg = cfg
        self.model_type = model_type
        self.path =  self.cfg['data_path']

        df =pd.read_csv(self.path+ 'evaluation_setup/fold1_evaluate.csv',sep = '\t')
        self.classes = sorted(list(set(df['scene_label'].values.tolist())))

        audio_mean_std = np.load('./mean_std_TAU.npz')
        self.audio_mean = audio_mean_std['mean']
        self.audio_std = audio_mean_std['std']


        if self.model_type=='audio':
            self.all_files = df['filename_audio'].values.tolist()
            # self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
            # self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)
        else:
            self.all_files = df_tau['filename_video'].values.tolist()
            self.video_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
                                           transforms.ToTensor()])



    def __len__(self):

        return len(self.all_files)



    def __getitem__(self,index):

        filename = os.path.join(self.path,self.all_files[index])

        if self.model_type=='video':
            video_ctr = av.open(filename)
            video_stream = video_ctr.streams.video[0]
            (video_dt, video_fps), _ = av_video_loader(
                container=video_ctr,
                rate=self.cfg['video_fps'],
                start_time=0,
                duration=self.cfg['video_clip_duration']
            )

            images_tensor = [self.data_transforms(i) for i in video_dt]
            images_tensor = torch.transpose(torch.stack(images_tensor),1,0)
            video_ctr.close()
            label = self.classes.index(self.all_files[index].split('/')[1].split('-')[0])
            ground_tr_tensor=torch.tensor(label)

            return images_tensor, ground_tr_tensor
        else:

        #     audio_dt, audio_fps = av_audio_loader(
        #     container=video_ctr,
        #     rate=self.cfg['audio_fps'],
        #     start_time=0,
        #     duration=self.cfg['audio_dur']
        # )
        #     import pdb; pdb.set_trace()
            #st =round(random.uniform(0.25, 10-0.75), 2)


            audio_dt,fs = librosa.load(filename,offset = 0,sr = self.cfg['audio_fps'],mono = True,duration = 10 )

            mels = librosa.feature.melspectrogram(y=audio_dt, sr=fs, n_fft=self.cfg['n_fft'], hop_length=int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']), power=1.0,n_mels=self.cfg['n_mels'])
            mels = librosa.core.power_to_db(mels, top_db=100)
            mels = torch.tensor(mels)
            mels =mels[None,:,:]

            mels = (mels-self.audio_mean[:,np.newaxis])/(self.audio_std[:,np.newaxis]+1e-5)
            mels = torch.transpose(mels, 1, 2)

            #video_ctr.close()
            label = self.classes.index(self.all_files[index].split('/')[1].split('-')[0])
            ground_tr_tensor=torch.tensor(label)


            return  mels,ground_tr_tensor
