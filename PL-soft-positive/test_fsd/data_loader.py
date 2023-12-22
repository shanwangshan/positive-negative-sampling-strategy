import torch
from torch.utils import data
import numpy as np
import os
import pandas
import h5py
import pickle
class FSD(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self,data_type, path_features,training_type):
        super(FSD, self).__init__()
        self.data_type = data_type
        self.path_features = path_features
        self.training_type = training_type


        if self.data_type == 'tr':
            self.path_input = self.path_features +'tr_'+self.training_type+'.hdf5'

        else:
            self.path_input =  self.path_features+'val_'+self.training_type+'.hdf5'

        self.all_files = []
        self.group = []
        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)
            elif isinstance(obj, h5py.Group):
                self.group.append(name)
        self.hf = h5py.File(self.path_input, 'r')
        self.hf.visititems(func)
        self.hf.close()
    def __len__(self):

        'Denotes the total number of samples'
        print('total number of files is',len(self.all_files))
        return len(self.all_files)
    def __getitem__(self,index):
        hf = h5py.File(self.path_input, 'r')
        emb = np.array(hf[self.all_files[index]])
        emb = torch.from_numpy(emb)
        emb=emb[None,:,:].float()
        ground_tr = int(self.all_files[index].split('/')[0])
        return emb, ground_tr
