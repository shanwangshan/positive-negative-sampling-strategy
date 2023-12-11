import sys
sys.path.insert(0,'../')
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

class Vgg_Sound(data.Dataset):

    def __init__(self,cfg,data_type,model_type):
        super(Vgg_Sound,self).__init__()
        self.cfg = cfg
        self.data_type = data_type
        self.model_type = model_type
        self.path = os.environ["LOCAL_SCRATCH"]+ self.cfg['vgg_path']
        #self.path = '../../../vgg-sound/'

        vggsound = pd.read_csv(self.cfg['filepath'],sep= '\t',usecols=[1,2,3])
        print('all the training files is', len(vggsound))
        val = vggsound.sample(frac=0.2, random_state=1)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        audio_mean_std = np.load('../mean_std_VGGsound.npz')
        self.audio_mean = audio_mean_std['mean']
        self.audio_std = audio_mean_std['std']


        if self.data_type=='val':
            self.train_files = val.values.tolist()
            print('validation has ',len(self.train_files))
        else:
            df = pd.concat([vggsound, val])
            df = df.drop_duplicates(keep=False)
            self.train_files = df.values.tolist()
            print('training has ', len(self.train_files))

        if self.model_type=='video':
            if self.data_type=='tr':
                self.data_transforms_tr = transforms.Compose([transforms.Resize(size=(112,112)),
                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),self.normalize])
            else:
                self.data_transforms_val = transforms.Compose([transforms.Resize(size=(112,112)),
                                                               transforms.ToTensor(),self.normalize])


        # else:
        #     self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
        #     self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

    def __len__(self):

        #print('total number of training files is',len(self.train_files))

        return len(self.train_files)



    def __getitem__(self,index):

        filename = os.path.join(self.path,'video/'+self.train_files[index][0]+'_'+str(self.train_files[index][1]*1000)+'_'+str(self.train_files[index][1]*1000+10000)+'.mp4')
        #filename = os.path.join(self.path,self.train_files[index])

        video_ctr = av.open(filename)

        video_stream = video_ctr.streams.video[0]
        dur = video_stream.duration * video_stream.time_base

        #st = np.random.randint(1,int(dur),1)
        st = round(random.uniform(0.25, dur-0.75), 2)
        if self.model_type=='video':
            if self.data_type=='tr':
                (video_dt, video_fps), _ = av_video_loader(
                container=video_ctr,
                rate=self.cfg['video_fps'],
                start_time=st-0.25,
                duration=0.1#self.cfg['video_clip_duration']
            )

                images_tensor = [self.data_transforms_tr(i) for i in video_dt]
                images_tensor = torch.transpose(torch.stack(images_tensor),1,0)
                video_ctr.close()
                #print(filename,torch.tensor(self.train_files[index][2])
                return images_tensor, torch.tensor(self.train_files[index][2])
            else:
                (video_dt, video_fps), _ = av_video_loader(
                container=video_ctr,
                rate=self.cfg['video_fps'],
                start_time=5,
                duration=0.1
            )
                images_tensor = [self.data_transforms_val(i) for i in video_dt]
                images_tensor = torch.transpose(torch.stack(images_tensor),1,0)
                video_ctr.close()
                #print(filename,torch.tensor(self.train_files[index][2])
                return images_tensor, torch.tensor(self.train_files[index][2])


        else:
            if self.data_type=='tr':
                audio_dt, audio_fps = av_audio_loader(
                container=video_ctr,
                rate=self.cfg['audio_fps'],
                start_time=st-0.25,
                duration=self.cfg['audio_dur']
                )
                audio = audio_dt.mean(axis=0)
                mels = librosa.feature.melspectrogram(y=audio, sr=audio_fps, n_fft=self.cfg['n_fft'], hop_length=int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']), power=1.0)
                mels = librosa.core.power_to_db(mels, top_db=100)
                mels = torch.tensor(mels)
                mels = mels[None,:,:]
                mels = mels[:,:,:100]

                audio_tensor = (mels-self.audio_mean[:,np.newaxis])/(self.audio_std[:,np.newaxis]+1e-5)
                audio_fea = torch.transpose(audio_tensor, 1, 2)
                video_ctr.close()
                return audio_fea, torch.tensor(self.train_files[index][2])
            else:
               audio_dt, audio_fps = av_audio_loader(
               container=video_ctr,
               rate=self.cfg['audio_fps'],
               start_time=0,
               duration=10
           )
               audio = audio_dt.mean(axis=0)
               mels = librosa.feature.melspectrogram(y=audio, sr=audio_fps, n_fft=self.cfg['n_fft'], hop_length=int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']), power=1.0)
               mels = librosa.core.power_to_db(mels, top_db=100)
               mels = torch.tensor(mels)
               mels = mels[None,:,:]
               #mels = mels[:,:,:100]

               audio_tensor = (mels-self.audio_mean[:,np.newaxis])/(self.audio_std[:,np.newaxis]+1e-5)
               audio_fea = torch.transpose(audio_tensor, 1, 2)
               video_ctr.close()
               return audio_fea, torch.tensor(self.train_files[index][2])
