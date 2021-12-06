import os
import copy
import math
import tqdm
import torch
import random
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from torchvision.utils import save_image

import argparse

from utils import *
from dataset import *
from model import ResNet18
from augmentations import *

parser = argparse.ArgumentParser(description='Mixed Example Pytorch Implementation')

'''Seed'''
seed = 1

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

'''Data Preparation Arguments'''
parser.add_argument('--data_prep_from_scratch', type=bool, default=True, help='')
parser.add_argument('--dataset', type=str, default='cifar10', help='')
parser.add_argument('--path_dataset', type=str, default='./Data/CIFAR10/', help='')

'''Data Augmentation Method'''
parser.add_argument('--method', type=str, default='vhmixup', help='Either \'vhmixup\' or \'vhbcplus\'')

'''Optimization Arguments'''
parser.add_argument('--lr', default=0.01, help='')
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--momentum', default=0.9, help='')

parser.add_argument('--weight_decay', default=5e-4, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--train_epochs', type=int, default=225, help='')

# Increase LR from 0.01 to 1.
lr_scheduler_1_gamma = 10.0; milestones_1 = [400]
# Lower LR from 1. to 0.01
lr_scheduler_2_gamma = 0.1; milestones_2 = [32000, 48000, 70000]

def main():
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create dataloader
    if args.data_prep_from_scratch and args.dataset == 'cifar100':
        prep_cifar100(args.path_dataset, args.dataset)
        
    if args.data_prep_from_scratch and args.dataset == 'cifar10':
        prep_cifar10(args.path_dataset, args.dataset)
        
    d_tr, d_te, n_outputs = load_datasets(os.path.join(args.path_dataset, '{}.pt'.format(args.dataset)))
    
    train_datasets = CIFAR(d_tr, args.method, train=True)
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    train_iterator = iter(train_dataloaders)
    
    test_datasets = CIFAR(d_te, args.method)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
    
    # Create instance of model
    model = ResNet18(n_outputs).to(device)
    
    # Define the augmentation method to be used
    if args.method == 'vhmixup':
        augmentor = VHMixup
    else:
        augmentor = VHBCplus
        
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(opt, gamma=lr_scheduler_1_gamma, milestones=milestones_1)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(opt, gamma=lr_scheduler_2_gamma, milestones=milestones_2)
    
    # Loss Function for soft labels
    def softXEnt(output, target):
        logprobs = torch.nn.functional.log_softmax(output, dim = 1)
        return  -(target * logprobs).sum() / output.shape[0]
    
    # Summary Writer
    ROOT_DIR = './Results/'
    now =  '{}_ResNet_{}_{}'.format(args.dataset, args.method, seed)

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    if not os.path.exists(ROOT_DIR + now):
        os.makedirs(ROOT_DIR + now)

    LOG_DIR = ROOT_DIR + now + '/logs/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    else:
        import shutil
        shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR)

    MODEL_DIR = ROOT_DIR + now + '/models/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    summary_writer = SummaryWriter(LOG_DIR)
    
    max_acc = -1
    for epoch in range(args.train_epochs+1):
        model.train()
        for i, d in enumerate(tqdm.tqdm(train_dataloaders)):
            try:
                d1 = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloaders)
                d1 = next(train_iterator)

            try:
                d2 = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloaders)
                d2 = next(train_iterator)

            x1, y1 = d1[0], d1[1]
            x2, y2 = d2[0], d2[1]

            x, y = augmentBatch((x1,y1), (x2,y2), augmentor, n_outputs)
            x = x.float().to(device)
            y = y.float().to(device)

            opt.zero_grad()

            out = model(x)

            loss = softXEnt(out, y)
            loss.backward()

            opt.step()

            summary_writer.add_scalar('Loss', loss.item())

            # Scheduler is defined based on total number of iterations
            scheduler_1.step()
            scheduler_2.step()
        
        # Evaluate Model after every epoch
        total_acc = evaluate(model, test_dataloaders, device)

        summary_writer.add_scalar('Eval ACC', total_acc / len(test_datasets))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), MODEL_DIR+'model_after_{}.pth'.format(epoch))
            max_acc = total_acc / len(test_datasets)
            
    torch.save(model.state_dict(), MODEL_DIR+'final.pth')
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    