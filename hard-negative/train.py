from data_loader import Vgg_Sound
from torch.utils.data import DataLoader,Sampler
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
import torch.utils.data.sampler as sampler
import yaml
import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

# python train.py config.yaml
parser = argparse.ArgumentParser(description='self-supervised learning on vgg-sound')
parser.add_argument('cfg', help='training config file')

args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
print(cfg)
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

if cfg['debug']:
  cfg['batch_size'] = 1
  cfg['num_workers'] = 0
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            #y = y.numpy()
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)


        self.y = y
        self.shuffle = shuffle
        self.class_indices = self._get_class_indices()
    def _get_class_indices(self):
        class_indices = {}
        for i in range(len(self.y)):
            label = self.y[i]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        #import pdb; pdb.set_trace()
        #print(class_indices)
        return class_indices


    def __iter__(self):
      for label_indices in self.class_indices.values():

        yield label_indices

    def __len__(self):
      return len(self.y)

tr_Dataset = Vgg_Sound(cfg)


# my_sampler = SameClassSampler(tr_Dataset, 123)
# training_generator = DataLoader(tr_Dataset, num_workers = cfg['num_workers'],batch_size = cfg['batch_size'], sampler=my_sampler)

# sequential_sampler = SequentialSampler(tr_Dataset)
# training_generator = DataLoader(tr_Dataset, num_workers = cfg['num_workers'],batch_size = cfg['batch_size'], sampler=sequential_sampler)

training_generator = DataLoader(tr_Dataset, num_workers =cfg['num_workers'],
                                batch_sampler=StratifiedBatchSampler(np.array(tr_Dataset.train_files)[:,2],batch_size =cfg['batch_size'] ))

model = av_wrapper(proj_dim=None)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = torch.nn.DataParallel(model)

model.to(device)


infonce = ContrastiveLoss(input_dim=512,proj_dim=[128],target='cross-modal',temperature=0.07, normalize=True,device=device)

optimizer = torch.optim.Adam(
            params=list(model.parameters()) + list(infonce.parameters()),
            lr=0.0001,
            weight_decay=0.00001
        )


start_epoch = 0
n_epochs = cfg['n_epochs']
training_loss = []
def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir+'checkpoint.pt'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    training_loss  = checkpoint['training_loss']
    return model, optimizer, checkpoint['epoch'],training_loss

ckp_path = cfg['ckp_path']+"checkpoint.pt"
if os.path.exists(ckp_path):
    model, optimizer, start_epoch, training_loss = load_ckp(ckp_path, model, optimizer)


def train(epoch):
    model.train()
    train_loss = 0.

    start_time = time.time()
    count  = training_generator.__len__()*(epoch-1)
    #loader = tqdm(training_generator)
    loader = training_generator

    for batch_idx, data in enumerate(loader):
        #embed()
        count = count + 1

        video = data[0].to(device=device, dtype=torch.float)
        audio = data[1].to(device=device, dtype=torch.float)
        label=  data[2].to(device=device, dtype=torch.float)

        #import pdb; pdb.set_trace()

        bs = video.shape[0]

        video_emb, audio_emb = model(video, audio)

        features = torch.cat([video_emb, audio_emb], dim=0)

        video_emb = video_emb.view(bs, 1, -1)
        audio_emb = audio_emb.view(bs, 1, -1)
        loss = infonce(video_emb,audio_emb)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()



        if (batch_idx+1) % 100 == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(training_generator),
                elapsed * 1000 / (batch_idx+1), loss ))

    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    print(torch.unique(label).shape)
    return train_loss



if not os.path.exists(cfg['ckp_path']):
    os.makedirs(cfg['ckp_path'])
    print("Directory " , cfg['ckp_path'] ,  " Created ")
else:
    print("Directory " , cfg['ckp_path'],  " already exists")


for epoch in range(start_epoch, n_epochs):

    print('this is epoch', epoch)
    training_loss.append(train(epoch)) # Call training


    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),

    'optimizer': optimizer.state_dict(),
    'training_loss': training_loss
}
    save_ckp(checkpoint, cfg['ckp_path'])
    print(training_loss)

plt.figure()
plt.plot(training_loss,'r')

plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(cfg['ckp_path']+'loss.png')
