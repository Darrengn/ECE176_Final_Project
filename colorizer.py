import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

class Preprocess:
    def __call__(self, img):
        img = img.convert("LAB")  
        img = T.functional.to_tensor(img)

        img[0] = img[0] / 100 
        img[1] = (img[1] + 128) / 255
        img[2] = (img[2] + 128) / 255
        return img


NUM_TRAIN = 49000
batch_size= 64

if __name__ == '__main__':
    train_transform = transform = T.Compose([
        Preprocess()
    ])

    cifar10_train = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
        transform=train_transform)
    loader_train = DataLoader(cifar10_train, batch_size=batch_size, num_workers=2,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./datasets/cifar10', train=True, download=True,
        transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=batch_size, num_workers=2, 
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./datasets/cifar10', train=False, download=True, 
        transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=batch_size, num_workers=2)

    model = nn.Sequential(
        # conv1
        nn.Conv2d(1, 64, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, stride=2, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        # conv2
        nn.Conv2d(64, 128, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, stride=2, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        # conv3
        nn.Conv2d(128, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=2, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        # conv4
        nn.Conv2d(256, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, stride=2, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv5
        nn.Conv2d(512, 512, 3, dilation=2, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, stride=2, dilation=2, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv6
        nn.Conv2d(512, 512, 3, dilation=2, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, stride=2, dilation=2, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv7
        nn.Conv2d(512, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=2, padding = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(256, 128, 4, sride=2, padding=1),
        # conv8
        nn.Conv2d(128, 128, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, stride=2, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        #1x1 to ab color space
        nn.Conv2d(128, 313, 1),
        
        nn.Softmax(1),
        nn.Upsample(scale_factor=4, mode='bilinear')
        )