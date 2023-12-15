import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

path = '/lustre/wang9/amc/FSDnoisy18k/features/'
name = 'clean'
hf = h5py.File(path+'tr_'+name+'.hdf5', 'r')

all_files = []
group = []
def func(name, obj):     # function to recursively store all the keys
    if isinstance(obj, h5py.Dataset):
        all_files.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

hf.visititems(func)


# data_0 = hf[all_files[0]] # 100x96
# label_0 = int(all_files[0].split('/')[0])
data = []
for i in range(len(all_files)):

    data.append(hf[all_files[i]])

data =np.vstack(np.array(data))



mu = np.mean(data)
std = np.std(data)
fn = './mean_std_'+name+'.npz'
np.savez(fn, mean=mu, std=std)
print('number of training files',len(all_files))
print('saved')

import pdb; pdb.set_trace()
#clean : 5794
#all:
#noisy:
#noisy_small: 6499
