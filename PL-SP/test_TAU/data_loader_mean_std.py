import sys
import path
#directory = path.Path(__name__).abspath()

# setting path
#sys.path.append(directory.parent.parent)
sys.path.insert(0,'../')
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
import soundfile as sf
import torchaudio.transforms as T

class TAU(data.Dataset):

    def __init__(self,cfg,data_type):
        super(TAU,self).__init__()
        self.cfg = cfg
        self.data_type = data_type

        self.path =  self.cfg['data_path']
        self.model_type = self.cfg['model_type']
        #df =pd.read_csv(os.environ["LOCAL_SCRATCH"] + 'evaluation_setup/train.csv',sep = ',')
        df =pd.read_csv(self.path+ 'evaluation_setup/train.csv',sep = ',')
        self.classes = sorted(list(set(df['scene_label'].values.tolist())))

        if self.data_type=='tr':
            df_tau =pd.read_csv(self.path + 'evaluation_setup/fold1_train.csv',sep = '\t')

            if self.model_type=='audio':
                self.all_files = df_tau['filename_audio'].values.tolist()

                self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
                self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

            else:
                self.all_files = df_tau['filename_video'].values.tolist()
                self.video_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        elif self.data_type=='val':

            df_tau =pd.read_csv(self.path + 'evaluation_setup/val.csv',sep = ',')
            #df_tau =pd.read_csv(os.environ["LOCAL_SCRATCH"] + 'evaluation_setup/val.csv',sep = ',')
            #self.all_files = df_tau['filename_video'].values.tolist()
            if self.model_type=='audio':
                self.all_files = df_tau['filename_audio'].values.tolist()
                self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
                self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)
            else:
                self.all_files = df_tau['filename_video'].values.tolist()
                self.video_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
                                              transforms.ToTensor()])



    def __len__(self):

        return len(self.all_files)



    def __getitem__(self,index):

        filename = os.path.join(self.path,self.all_files[index])
        #video_ctr = av.open(filename)
        #filename = os.path.join(self.path,self.train_files[index])
        #import pdb; pdb.set_trace()
        #dur = video_stream.duration * video_stream.time_base
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
            st = round(random.uniform(0,8), 2)
            audio_dt,fs = librosa.load(filename,offset = st,sr = self.cfg['audio_fps'],mono = True,duration = 2 )
            #import pdb; pdb.set_trace()
            #audio_dt,fs = sf.read(filename )
            #audio_dt, audio_fps = torchaudio.load(filename)
            #import pdb; pdb.set_trace()

            #import pdb; pdb.set_trace()
            mels = librosa.feature.melspectrogram(y=audio_dt, sr=fs, n_fft=self.cfg['n_fft'], hop_length=int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']), power=1.0,n_mels=self.cfg['n_mels'])
            mels = librosa.core.power_to_db(mels, top_db=100)
            mels = torch.tensor(mels)
            mels =mels[None,:,:]
            mels = torch.transpose(mels, 1, 2)
            #audio_tensor = self.audio_transforms1(audio)
            #audio_tensor = self.audio_transforms2(audio_tensor)
            #audio_fea = torch.transpose(audio_tensor, 1, 2)
            #video_ctr.close()
            # label = self.classes.index(self.all_files[index].split('/')[1].split('-')[0])
            # ground_tr_tensor=torch.tensor(label)


            return  mels
