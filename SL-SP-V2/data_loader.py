
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
from sklearn import preprocessing
class Vgg_Sound(data.Dataset):

    def __init__(self,cfg):
        super(Vgg_Sound,self).__init__()
        self.cfg = cfg
        self.path = os.environ["LOCAL_SCRATCH"]+ self.cfg['vgg_path']

        vggsound =pd.read_csv('./sudo_label.csv', sep = ',')

        self.train_files =  vggsound.values.tolist()


        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        audio_mean_std = np.load('./mean_std_VGGsound.npz')
        self.audio_mean = audio_mean_std['mean']
        self.audio_std = audio_mean_std['std']


        self.data_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
                                              transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),self.normalize])
        #self.data_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
         #
         #                                          transforms.ToTensor(),normalize])


        # self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
        # self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

    def __len__(self):

        print('total number of training files is',len(self.train_files))

        return len(self.train_files)



    def __getitem__(self,index):
        filename = self.train_files[index][0].split('/')[2].split('.jpg')[0]+'.mp4'

        filename_path = os.path.join(self.path,'video/'+filename)
        #filename = os.path.join(self.path,self.train_files[index])

        video_ctr = av.open(filename_path)

        video_stream = video_ctr.streams.video[0]
        dur = video_stream.duration * video_stream.time_base

        #st = np.random.randint(1,int(dur),1)
        st = round(random.uniform(0.25, dur-0.75), 2)

        sudo_label = self.train_files[index][1]
        all_lists = [i for i in self.train_files if sudo_label==i[1]]
        filename_1 = random.choice(all_lists)
        filename_1= filename_1[0].split('/')[2].split('.jpg')[0]+'.mp4'
        filename_1 = os.path.join(self.path,'video/'+filename_1)
        video_ctr_1 = av.open(filename_1)

        (video_dt, video_fps), _ = av_video_loader(
            container=video_ctr_1,
            rate=self.cfg['video_fps'],
            start_time=st,
            duration=self.cfg['video_clip_duration']
        )


        images_tensor = [self.data_transforms(i) for i in video_dt]
        images_tensor = torch.transpose(torch.stack(images_tensor),1,0)

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
        return images_tensor, audio_fea
