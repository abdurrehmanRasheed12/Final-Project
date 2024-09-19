import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, kind, block=ResidualBlock, all_connections=[3,4,6,3]):
        super(ResNet, self).__init__()

        if kind=='RGB':
          C = 3
        elif kind=='SWIR':
          C = 2

        self.inputs = 16
        self.conv1 = nn.Sequential(
                        nn.Conv2d(C, 16, kernel_size = 4, stride = 2),
                        nn.BatchNorm2d(16),
                        nn.ReLU()) #16x250x250
        self.maxpool = nn.MaxPool2d(kernel_size = 4, stride = 2) #16x124x124


        self.layer0 = self._make_layer(block, 16, all_connections[0], stride = 1) #connections = 3, shape: 16x124x124
        self.layer1 = self._make_layer(block, 32, all_connections[1], stride = 2)#connections = 4, shape: 128x
        self.layer2 = self._make_layer(block, 64, all_connections[2], stride = 2)#connections = 6
        self.layer3 = self._make_layer(block, 128, all_connections[3], stride = 2)#connections = 3, shape: 512x10x10
        self.avgpool = nn.AvgPool2d(12, stride=2)
        self.fc = nn.Linear(128*3*3, 1)

    def _make_layer(self, block, outputs, connections, stride=1):
        downsample = None
        if stride != 1 or self.inputs != outputs:
            downsample = nn.Sequential(
                nn.Conv2d(self.inputs, outputs, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outputs),
            )
        layers = []
        layers.append(block(self.inputs, outputs, stride, downsample))
        self.inputs = outputs
        for i in range(1, connections):
            layers.append(block(self.inputs, outputs))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 128*3*3)
        x = self.fc(x).flatten()
        return x
