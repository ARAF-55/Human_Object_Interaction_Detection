from __future__ import print_function, division
import torch
import torch.nn as nn
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader_vcoco import Rescale, ToTensor, vcoco_Dataset, vcoco_collate
from train_test import train_test
import model as rr
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

seed = 10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def _init_fn(worker_id):
    np.random.seed(int(seed))


number_of_epochs = 1
learning_rate = 0.01
breaking_point = 100000
saving_epoch = 1
first_word = "Checkpoints"
batch_size = 1
resume_model = 't'
infr = 't'
hyp = 'f'
visualize = 'test'
check = "best"
############################################

all_data_dir = "All_data/"

annotation_train = all_data_dir + 'Annotations_vcoco/train_annotations.json'
image_dir_train = all_data_dir + 'Data_vcoco/train2014/'

annotation_val = all_data_dir + 'Annotations_vcoco/val_annotations.json'
image_dir_val = all_data_dir + 'Data_vcoco/train2014/'

annotation_test = all_data_dir + 'Annotations_vcoco/test_annotations.json'
image_dir_test = all_data_dir + 'Data_vcoco/val2014/'

vcoco_train = vcoco_Dataset(annotation_train, image_dir_train, transform=transforms.Compose([Rescale((400, 400)), ToTensor()]))
vcoco_val = vcoco_Dataset(annotation_val, image_dir_val, transform=transforms.Compose([Rescale((400, 400)), ToTensor()]))
vcoco_test = vcoco_Dataset(annotation_test, image_dir_test, transform=transforms.Compose([Rescale((400, 400)), ToTensor()]))

dataloader_train = DataLoader(vcoco_train, batch_size, shuffle=True, collate_fn=vcoco_collate, worker_init_fn=_init_fn)
dataloader_val = DataLoader(vcoco_val, batch_size, shuffle=True, collate_fn=vcoco_collate, worker_init_fn=_init_fn)
dataloader_test = DataLoader(vcoco_test, batch_size, shuffle=False, collate_fn=vcoco_collate, worker_init_fn=_init_fn)
dataloader = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}

folder_name = '{}'.format(first_word)

### Loading Model ###
res = rr.HOI_Detector()

trainables = []
not_trainables = []
spmap = []
single = []

for name, p in res.named_parameters():
    if name.split('.')[0] == 'Conv_pretrain':
        p.requires_grad = False
        not_trainables.append(p)
    else:
        if name.split('.')[0] in ['conv_sp_map', 'spmap_up']:
            spmap.append(p)
        else:
            trainables.append(p)

optim1 = optim.SGD(
    [
        {"params": trainables, "lr": learning_rate},
        {"params": spmap, "lr": 0.001}
    ],
    momentum=0.9, weight_decay=0.0001
)

lambda1 = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 28 else 1)
lambda2 = lambda epoch: 1
scheduler = optim.lr_scheduler.LambdaLR(optim1, [lambda1, lambda2])

res.to(device)

epoch = 0
mean_best = 0

if resume_model == 't':
    try:
        file_name = folder_name + '/' + check + 'checkpoint.pth.tar'
        print(file_name)
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
        res.load_state_dict(checkpoint['state_dict'], strict=True)
        epoch = checkpoint['epoch']
        mean_best = checkpoint['mean_best']
        print(f"=> loaded checkpoint when best_prediction mAP {mean_best} and epoch {checkpoint['epoch']}")
    except:
        print('Failed to load checkpoint')

if hyp == 't':
    try:
        print('Loading previous Hyperparameters')
        optim1.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        print('Failed to load previous Hyperparameters')
        

    
train_test(res, optim1, scheduler, dataloader, number_of_epochs, breaking_point, saving_epoch, folder_name, batch_size, infr, epoch, mean_best, visualize)
