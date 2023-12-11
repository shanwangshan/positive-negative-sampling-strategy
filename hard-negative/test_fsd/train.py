import sys
from data_loader import FSD
from torch.utils.data import DataLoader
sys.path.insert(0,'../')
from model import ClassificationWrapper

import models
import torch
from models.av_wrapper import av_wrapper
from criterions.contrastive import ContrastiveLoss

import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
import argparse

# python train.py config.yaml
parser = argparse.ArgumentParser(description='audio/video test self-supervised learning on vgg-sound')
parser.add_argument('cfg', help='training config file')
parser.add_argument('--model_type', type= str, required = True, help='train audio subnetwork or video')
parser.add_argument('--training_type', type= str, required = True, help='train all noisy noisy_small clean')
parser.add_argument('--lin_prob', default = False, action="store_true", help='linear prob')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))
print(cfg)
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

if cfg['debug']:
  cfg['batch_size'] = 1
  cfg['num_workers'] = 0



tr_Dataset = FSD(data_type="tr",path_features=cfg['data_path'], training_type=args.training_type)
training_generator = DataLoader(tr_Dataset, batch_size = cfg['batch_size'], shuffle = True, num_workers = cfg['num_workers'], drop_last = True, pin_memory=True)

val_Dataset = FSD(data_type="val",path_features=cfg['data_path'], training_type=args.training_type)
val_generator = DataLoader(val_Dataset, batch_size = cfg['batch_size'], shuffle = False, num_workers = cfg['num_workers'], drop_last = False, pin_memory=True)

# tt_Dataset = TAU(cfg,transform=False)
# tt_generator = DataLoader(tt_Dataset, batch_size = cfg['batch_size'], shuffle = False, num_workers = cfg['num_workers'], drop_last = False, pin_memory=True)




pretrained_net = av_wrapper(proj_dim=None)
checkpoint_fn = '../checkpoint/checkpoint.pt'
ckp = torch.load(checkpoint_fn, map_location='cpu')
model_ckp = ckp['state_dict'] if 'state_dict' in ckp else ckp['model']
pretrained_net.load_state_dict({k.replace('module.', ''): model_ckp[k] for k in model_ckp})
#pretrained_net.load_state_dict(torch.load(checkpoint_fn))
if args.model_type=='video':
  pretrained_vnet = pretrained_net.video_model
  if args.lin_prob:
    output_dir = './video_model_lin/'
  else:
    output_dir = './video_model_ft/'


else:
  pretrained_vnet = pretrained_net.audio_model
  if args.lin_prob:
    output_dir = './audio_model_lin_'+args.training_type+'/'
  else:
    output_dir = './audio_model_ft_'+args.training_type+'/'

print('model type is', args.model_type, 'linear prob is', args.lin_prob, 'training_type is', args.training_type)

model = ClassificationWrapper(feature_extractor=pretrained_vnet, n_classes = cfg['num_classes'],feat_name = cfg['feat_name'],feat_dim=cfg['feat_dim'],pooling_op=cfg['pooling_op'],use_dropout=cfg['use_dropout'],dropout=cfg['dropout'])


# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = torch.nn.DataParallel(model)

loss_fn = torch.nn.CrossEntropyLoss()
if args.lin_prob:
  parameters = []
  for name, p in model.named_parameters():
    if "classifier" in name:
      parameters.append(p)

  optimizer =optim.Adam(parameters, lr=0.0001,weight_decay=0.0001)
else:
  optimizer =optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directory " , output_dir ,  " Created ")
else:
    print("Directory " , output_dir ,  " already exists")

########### use GPU ##########
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
########### use GPU ##########
model.to(device)


#### define the loss function and the optimizer#########
#loss_fn = torch.nn.CrossEntropyLoss()
#optimizer =optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

start_epoch = 0
training_loss = []

print('-----------start training')
######define train function######
def train(epoch):
    model.train()
    train_loss = 0.
    #embed()
    start_time = time.time()
    count  = training_generator.__len__()*(epoch-1)
    #loader = tqdm(training_generator)
    loader = training_generator
    for batch_idx, data in enumerate(loader):
        #embed()
        count = count + 1
        #import pdb; pdb.set_trace()
        batch_embed = data[0].to(device)
        batch_label = data[1].to(device)

        # training
        optimizer.zero_grad()
       # embed()
        esti_label = model(batch_embed)
        loss = loss_fn(esti_label,batch_label)
        loss.backward()

        train_loss += loss.data.item()
        optimizer.step()
        #writer_tr.add_scalar('Loss/train', loss.data.item(),batch_idx*epoch)

        if (batch_idx+1) % 100 == 0:
            elapsed = time.time() - start_time

            #writer_tr.add_scalar('Loss/train', loss.data.item(),count)
            #writer_tr.add_scalar('Loss/train_avg', train_loss/(batch_idx+1),count)
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(training_generator),
                elapsed * 1000 / (batch_idx+1), loss ))


    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))

    return train_loss
######define train function######

######define validate function######
def validate(epoch):
    model.eval()
    validation_loss = 0.
    start_time = time.time()
    # data loading
    for batch_idx, data in enumerate(val_generator):
        batch_embed = data[0].to(device)
        batch_label = data[1].to(device)

        with torch.no_grad():
             esti_label = model(batch_embed)
             loss = loss_fn(esti_label,batch_label)
             validation_loss += loss.data.item()

    #print('the ',batch_idx,'iteration val_loss is ', validation_loss)
    validation_loss /= (batch_idx+1)
   # embed()
    #writer_val.add_scalar('Loss/val', loss.data.item(),batch_idx*epoch)
    #writer_val.add_scalar('Loss/val_avg', validation_loss,batch_idx*epoch)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)

    return validation_loss
######define validate function######


training_loss = []
validation_loss = []
decay_cnt = 0
#writer_tr = SummaryWriter(os.path.join(output_dir,'train'))
#writer_val = SummaryWriter(os.path.join(output_dir,'val'))
for epoch in range(1, cfg['n_epochs']):
    model.cuda()
    print('this is epoch', epoch)
    training_loss.append(train(epoch)) # Call training
    validation_loss.append(validate(epoch)) # call validation
    print('-' * 99)
    print('after epoch', epoch, 'training loss is ', training_loss, 'validation loss is ', validation_loss)
    if training_loss[-1] == np.min(training_loss):
        print(' Best training model found.')
        print('-' * 99)
    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(output_dir+'model.pt', 'wb') as f:
            torch.save(model.cpu().state_dict(), f)

            print(' Best validation model found and saved.')
            print('-' * 99)

        #torch.save(model, os.path.join(os.path.dirname(args.val_save), 'alt_' + os.path.basename(args.val_save))) # save in alternative format

    # decay_cnt += 1
    # #lr decay
    # #if np.min(validation_loss) not in validation_loss[-3:] and decay_cnt >= 3:
    # if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
    #     scheduler.step()
    #     decay_cnt = 0
    #     print('  Learning rate decreased.')
    #     print('-' * 99)

####### plot the loss and val loss curve####
minmum_val_index=np.argmin(validation_loss)
minmum_val=np.min(validation_loss)
plt.plot(training_loss,'r')
#plt.hold(True)
plt.plot(validation_loss,'b')
plt.axvline(x=minmum_val_index,color='k',linestyle='--')
plt.plot(minmum_val_index,minmum_val,'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(output_dir+'loss.png')
