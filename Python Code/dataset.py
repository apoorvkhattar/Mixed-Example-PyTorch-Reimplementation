import os
import torch
import pickle

import numpy as np
import torchvision.transforms.functional as TF

from torchvision import transforms
from torchvision.utils import save_image

# Prepare CIFAR10 / CIFAR100 as .pt file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prep_cifar100(path_dataset, dataset):
    cifar100_train = unpickle(os.path.join(path_dataset, 'train'))
    cifar100_test = unpickle(os.path.join(path_dataset, 'test'))

    x_tr = torch.from_numpy(cifar100_train[b'data'].reshape((-1,32,32,3), order='F')).permute(0,2,1,3)
    y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
    x_te = torch.from_numpy(cifar100_test[b'data'].reshape((-1,32,32,3), order='F')).permute(0,2,1,3)
    y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

    torch.save((x_tr, y_tr, x_te, y_te, 100), os.path.join(path_dataset, '{}.pt'.format(dataset)))
    
def prep_cifar10(path_dataset, dataset):
    x_tr, y_tr = None, None
    for b in range(5):
        cifar10_train = unpickle(os.path.join(path_dataset, 'data_batch_{}'.format(b+1)))
        
        batch_img = torch.from_numpy(cifar10_train[b'data'].reshape((-1,32,32,3), order='F')).permute(0,2,1,3)
        batch_label = torch.LongTensor(cifar10_train[b'labels'])
        
        if x_tr is None:
            x_tr = batch_img
            y_tr = batch_label
        else:
            x_tr = torch.cat((x_tr, batch_img), dim=0)
            y_tr = torch.cat((y_tr, batch_label), dim=0)
    
    cifar10_test = unpickle(os.path.join(path_dataset, 'test_batch'))
    x_te = torch.from_numpy(cifar10_test[b'data'].reshape((-1,32,32,3), order='F')).permute(0,2,1,3)
    y_te = torch.LongTensor(cifar10_test[b'labels'])
    
    torch.save((x_tr, y_tr, x_te, y_te, 10), os.path.join(path_dataset, '{}.pt'.format(dataset)))
    

def load_datasets(path):
    d = torch.load(path)
    d_tr = (d[0], d[1])
    d_te = (d[2], d[3])
    n_outputs = d[4]
    return d_tr, d_te, n_outputs

# Custom CIFAR Dataloader with random cropping and horizontal flipping
class CIFAR(torch.utils.data.Dataset):
    def __init__(self, pack, method, train=False):
        self.x = pack[0]
        self.y = pack[1]
        self.img_size = (3,32,32)
        
        self.method = method
        self.train = train
    
    def __len__(self):
        return len(self.x)
    
    def transform(self, img):
        top = torch.randint(0,8,(1,))
        left = torch.randint(0,8,(1,))
        img = TF.crop(img, top=top, left=left, height=self.img_size[1], width=self.img_size[2])
        
        if torch.rand(1) > 0.5:
            img = TF.hflip(img)
            
        return img
    
    def __getitem__(self, item):
        x = self.x[item].float() / 255.0
        
        x = x.permute(2,0,1)
        
        if self.train:
            x = TF.pad(x, padding=4)
            x = self.transform(x)
        
        if 'bcplus' not in self.method:
            mean_image = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465])).float()
            std_image = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010])).float()
        else:
            x = x - torch.mean(x)
            mean_image = torch.from_numpy(np.array([0.21921569, 0.21058824, 0.22156863])).float()
            std_image = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010])).float()
        
        x = x.permute(1,2,0)
        x = x - mean_image
        x = x / std_image
        
        return x.permute(2,0,1), self.y[item]
    