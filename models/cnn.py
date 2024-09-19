import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class SimpleCNN(nn.Module):
    def __init__(self, kind):
        super(SimpleCNN, self).__init__()
        if kind=='RGB':
          C = 3
        elif kind=='SWIR':
          C = 2
        self.conv_1 = DoubleConv(C, 8) # 8x502x502
        self.pool_1 = nn.MaxPool2d(kernel_size=4, stride=2) # 8x250x250

        self.conv_2 = DoubleConv(8, 16)  #16x250x250
        self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=2) #16x124x124

        self.conv_3 = DoubleConv(16, 32)  #32x124x124
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2) #32x62x62

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2) #32x31x31

        self.fc1 = nn.Linear(32 * 31 * 31, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        conv_1_out = self.conv_1(x)
        conv_2_out = self.conv_2(self.pool_1(conv_1_out))
        conv_3_out = self.conv_3(self.pool_2(conv_2_out))
        output = self.pool_4(self.pool_3(conv_3_out))

        output = output.view(-1, 32 * 31 * 31)
        output = self.relu(self.fc1(output))
        output = self.fc2(output).flatten()
        return output
