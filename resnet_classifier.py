import torch.nn as nn
from utils.data_preprocess import get_cifar10_data

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, batch_norm=False, downsample=False, padding = 0):
        super(ResNet, self).__init__()
        stride=1
        if downsample:
            stride=2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride=1, padding=padding)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.relu = nn.ReLU()
        if downsample:
            self.residual = nn.Conv2d(in_channel, out_channel, 1, stride=2)
        else:
            self.residual = nn.Conv2d(in_channel, out_channel, 1, stride=1)
        nn.init.kaiming_normal_(self.residual.weight)
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
    def forward(self,x):
        if self.batch_norm:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        x = self.residual(x)
        return self.relu(out+x)