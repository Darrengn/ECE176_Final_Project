import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

NUM_TRAIN = 49000
batch_size= 64

train_transform = transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

cifar10_train = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=train_transform)
loader_train = DataLoader(cifar10_train, batch_size=batch_size, num_workers=2,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=batch_size, num_workers=2, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets/cifar100', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=batch_size, num_workers=2)

