
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import kornia
from PIL import Image
import numpy as np

from resnet_classifier import ResNet

from ColorizationLoss import *
from colorizer import Preprocess

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def show_img(loader, model):
    gamut = torch.tensor([[-90, 50], [-90, 60], [-90, 70], [-90, 80], [-90, 90],
                     [-80, 20], [-80, 30], [-80, 40], [-80, 50], [-80, 60], [-80, 70], [-80, 80], [-80, 90],
                     [-70, 0], [-70, 10], [-70, 20], [-70, 30], [-70, 40], [-70, 50], [-70, 60], [-70, 70], [-70, 80], [-70, 90], 
                     [-60, -20], [-60, -10], [-60, 0], [-60, 10], [-60, 20], [-60, 30], [-60, 40], [-60, 50], [-60, 60], [-60, 70], [-60, 80], [-60, 90], 
                     [-50, -30], [-50, -20], [-50, -10], [-50, 0], [-50, 10], [-50, 20], [-50, 30], [-50, 40], [-50, 50], [-50, 60], [-50, 70], [-50, 80], [-50, 90], [-50, 100], 
                     [-40, -40], [-40, -30], [-40, -20], [-40, -10], [-40, 0], [-40, 10], [-40, 20], [-40, 30], [-40, 40], [-40, 50], [-40, 60], [-40, 70], [-40, 80], [-40, 90], [-40, 100], 
                     [-30, -50], [-30, -40], [-30, -30], [-30, -20], [-30, -10], [-30, 0], [-30, 10], [-30, 20], [-30, 30], [-30, 40], [-30, 50], [-30, 60], [-30, 70], [-30, 80], [-30, 90], [-30, 100], 
                     [-20, -50], [-20, -40], [-20, -30], [-20, -20], [-20, -10], [-20, 0], [-20, 10], [-20, 20], [-20, 30], [-20, 40], [-20, 50], [-20, 60], [-20, 70], [-20, 80], [-20, 90], [-20, 100], 
                     [-10, -60], [-10, -50], [-10, -40], [-10, -30], [-10, -20], [-10, -10], [-10, 0], [-10, 10], [-10, 20], [-10, 30], [-10, 40], [-10, 50], [-10, 60], [-10, 70], [-10, 80], [-10, 90], [-10, 100], 
                     [0, -70], [0, -60], [0, -50], [0, -40], [0, -30], [0, -20], [0, -10], [0, 0], [0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60], [0, 70], [0, 80], [0, 90], [0, 100], 
                     [10, -80], [10, -70], [10, -60], [10, -50], [10, -40], [10, -30], [10, -20], [10, -10], [10, 0], [10, 10], [10, 20], [10, 30], [10, 40], [10, 50], [10, 60], [10, 70], [10, 80], [10, 90], 
                     [20, -80], [20, -70], [20, -60], [20, -50], [20, -40], [20, -30], [20, -20], [20, -10], [20, 0], [20, 10], [20, 20], [20, 30], [20, 40], [20, 50], [20, 60], [20, 70], [20, 80], [20, 90], 
                     [30, -90], [30, -80], [30, -70], [30, -60], [30, -50], [30, -40], [30, -30], [30, -20], [30, -10], [30, 0], [30, 10], [30, 20], [30, 30], [30, 40], [30, 50], [30, 60], [30, 70], [30, 80], [30, 90], 
                     [40, -100], [40, -90], [40, -80], [40, -70], [40, -60], [40, -50], [40, -40], [40, -30], [40, -20], [40, -10], [40, 0], [40, 10], [40, 20], [40, 30], [40, 40], [40, 50], [40, 60], [40, 70], [40, 80], [40, 90], 
                     [50, -100], [50, -90], [50, -80], [50, -70], [50, -60], [50, -50], [50, -40], [50, -30], [50, -20], [50, -10], [50, 0], [50, 10], [50, 20], [50, 30], [50, 40], [50, 50], [50, 60], [50, 70], [50, 80], 
                     [60, -110], [60, -100], [60, -90], [60, -80], [60, -70], [60, -60], [60, -50], [60, -40], [60, -30], [60, -20], [60, -10], [60, 0], [60, 10], [60, 20], [60, 30], [60, 40], [60, 50], [60, 60], [60, 70], [60, 80], 
                     [70, -110], [70, -100], [70, -90], [70, -80], [70, -70], [70, -60], [70, -50], [70, -40], [70, -30], [70, -20], [70, -10], [70, 0], [70, 10], [70, 20], [70, 30], [70, 40], [70, 50], [70, 60], [70, 70], [70, 80], 
                     [80, -110], [80, -100], [80, -90], [80, -80], [80, -70], [80, -60], [80, -50], [80, -40], [80, -30], [80, -20], [80, -10], [80, 0], [80, 10], [80, 20], [80, 30], [80, 40], [80, 50], [80, 60], [80, 70], 
                     [90, -110], [90, -100], [90, -90], [90, -80], [90, -70], [90, -60], [90, -50], [90, -40], [90, -30], [90, -20], [90, -10], [90, 0], [90, 10], [90, 20], [90, 30], [90, 40], [90, 50], [90, 60], [90, 70], 
                     [100, -90], [100, -80], [100, -70], [100, -60], [100, -50], [100, -40], [100, -30], [100, -20], [100, -10], [100, 0]])
    gamut = gamut.to(device=device)
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            l = x[:, 0:1, :, :]
            scores = model(l)
            preds = prediction(scores)
            b, c, h, w = preds.shape
            preds = preds.view(b,h,w,c)
            a_b = torch.zeros(b,h,w,2)
            preds = preds.to(dtype=torch.int)
            a_b = gamut[preds[:,:,:,0]]
            a_b = a_b.view(b, 2, h, w)
            img = torch.zeros(b,3,h,w)
            img[:,0:1,:,:] = l*100
            img[:,1:3,:,:] = a_b*255-128
            img = kornia.color.lab_to_rgb(img)
            for batch in range(img.shape[0]):
                image = img[batch,:,:,:].view(h,w,3).cpu().numpy()
                rgb = Image.fromarray((image*255).astype(np.uint8))
                rgb.save(f"output/{batch}.png")

NUM_TRAIN = 49000
batch_size= 64




class_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

color_transform = T.Compose([
    Preprocess()
])


class_test = dset.CIFAR10('./datasets/cifar10', train=False, download=True, 
    transform=class_transform)
loader_class = DataLoader(class_test, batch_size=batch_size, num_workers=2)

color_test = dset.CIFAR10('./datasets/cifar10', train=False, download=True, 
    transform=color_transform)
loader_color = DataLoader(color_test, batch_size=batch_size, num_workers=2)

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 2, padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding = 1),
            ResNet(64,64,3,padding=1),
            ResNet(64,128,3,padding=1),
            ResNet(128,256,3,padding=1),
            ResNet(256,512,3,padding=1,downsample=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    model.load_state_dict(torch.load('classifier.pth'))
    model = model.to(device=device)
    model.eval()
    check_accuracy(loader_class, model)
    modelColor = nn.Sequential(
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
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        # conv4
        nn.Conv2d(256, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv5
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv6
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, dilation=2, padding = 2),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        # conv7
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
        # conv8
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        #1x1 to ab color space
        nn.Conv2d(256, 313, 1),
        
        nn.Softmax(1),
        nn.Upsample(scale_factor=4, mode='bilinear')
        )
    modelColor = modelColor.to(device=device)
    modelColor.load_state_dict(torch.load('colorizer.pth'))
    show_img(loader_color, modelColor)
