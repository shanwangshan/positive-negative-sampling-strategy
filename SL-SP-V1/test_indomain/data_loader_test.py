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

    def __init__(self,cfg,model_type):
        super(Vgg_Sound,self).__init__()
        self.cfg = cfg

        self.model_type = model_type
        self.path = os.environ["LOCAL_SCRATCH"]+ self.cfg['vgg_path']
        #vggsound =pd.read_csv(self.path + 'vggsound-1.csv',header=None)
        #self.path = '../../../vgg-sound/'
        vggsound =pd.read_csv(self.path + 'vggsound-1.csv',header=None)


        unwanted = pd.read_csv(cfg['unwanted_files_path'],header = None)
        unwanted = unwanted[0].values.tolist()
        vggsound = vggsound[~vggsound[0].isin(unwanted)]
        test_files = vggsound.loc[vggsound[3] == 'test']
        self.classes = sorted(list(set(test_files[2].values.tolist())))
        self.all_test_files = test_files.values.tolist()

        print('all the test files is', len(self.all_test_files))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        audio_mean_std = np.load('../mean_std_VGGsound.npz')
        self.audio_mean = audio_mean_std['mean']
        self.audio_std = audio_mean_std['std']


        if self.model_type=='video':
            self.data_transforms = transforms.Compose([transforms.Resize(size=(112,112)),
                                                       transforms.ToTensor(),self.normalize])
        # else:
        #     self.audio_transforms1 =T.MelSpectrogram(sample_rate=self.cfg['audio_fps'],n_fft = self.cfg['n_fft'], hop_length = int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']) , n_mels = self.cfg['n_mels'])
        #     self.audio_transforms2 = T.AmplitudeToDB(stype="power", top_db=100)

    def __len__(self):

        #print('total number of training files is',len(self.train_files))

        return len(self.all_test_files)



    def __getitem__(self,index):

        filename = os.path.join(self.path,'video/'+self.all_test_files[index][0]+'_'+str(self.all_test_files[index][1]*1000)+'_'+str(self.all_test_files[index][1]*1000+10000)+'.mp4')
        #filename = os.path.join(self.path,self.train_files[index])

        video_ctr = av.open(filename)

        video_stream = video_ctr.streams.video[0]


        if self.model_type=='video':
            (video_dt, video_fps), _ = av_video_loader(
                container=video_ctr,
                rate=self.cfg['video_fps'],
                start_time=0,
                duration=10
            )
            images_tensor = [self.data_transforms(i) for i in video_dt]
            images_tensor = torch.transpose(torch.stack(images_tensor),1,0)
            video_ctr.close()
                #print(filename,torch.tensor(self.train_files[index][2])
            return images_tensor, self.classes.index(self.all_test_files[index][2])


        else:
            audio_dt, audio_fps = av_audio_loader(
               container=video_ctr,
               rate=self.cfg['audio_fps'],
               start_time=0,
               duration=10)

            audio = audio_dt.mean(axis=0)
            mels = librosa.feature.melspectrogram(y=audio, sr=audio_fps, n_fft=self.cfg['n_fft'], hop_length=int(1/self.cfg['spectrogram_fps']*self.cfg['audio_fps']), power=1.0)
            mels = librosa.core.power_to_db(mels, top_db=100)
            mels = torch.tensor(mels)
            mels = mels[None,:,:]
               #mels = mels[:,:,:100]

            audio_tensor = (mels-self.audio_mean[:,np.newaxis])/(self.audio_std[:,np.newaxis]+1e-5)
            audio_fea = torch.transpose(audio_tensor, 1, 2)


            # audio = torch.from_numpy(audio_dt)
            # audio = audio.mean(axis=0,keepdim=True)

            # audio -= audio.mean()
            # max_value = max(abs(audio))+0.005
            # audio/=max_value
            # #import pdb; pdb.set_trace()
            # audio_tensor = self.audio_transforms1(audio)
            # audio_tensor = self.audio_transforms2(audio_tensor)
            # audio_tensor = (audio_tensor-self.audio_mean[:,np.newaxis])/(self.audio_std[:,np.newaxis]+1e-5)
            # audio_fea = torch.transpose(audio_tensor, 1, 2)
            if audio_fea.shape[1]!=1001:
                #print(filename)
                m = torch.nn.ZeroPad2d((0, 0, int(1001-audio_fea.shape[1]), 0))
                audio_fea = m(audio_fea)


            video_ctr.close()
            return audio_fea, self.classes.index(self.all_test_files[index][2])
