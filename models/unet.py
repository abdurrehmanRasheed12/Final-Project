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

class UNet(nn.Module):
    def __init__(self, kind, DoubleConv = DoubleConv):
      super().__init__()
      if kind=='RGB':
          C = 3
      elif kind=='SWIR':
          C = 2
      #ENCODER
      self.conv_1 = DoubleConv(C, 4) # 64x502x502
      self.pool_1 = nn.MaxPool2d(kernel_size=4, stride=2) # 64x250x250

      self.conv_2 = DoubleConv(4, 8)  #128x250x250
      self.pool_2 = nn.MaxPool2d(kernel_size=4, stride=2) #128x124x124

      self.conv_3 = DoubleConv(8, 16)  #256x124x124
      self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2) #256x62x62

      self.conv_4 = DoubleConv(16, 32)  #512x62x62
      self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2) #512x31x31

      self.conv_5 = DoubleConv(32, 64)  #1024x31x31

      #DECODER
      self.upconv_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) #512x62x62
      self.conv_6 = DoubleConv(64, 32) #512x62x62


      self.upconv_2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) #256x124x124
      self.conv_7 = DoubleConv(32, 16)  #256x124x124

      self.upconv_3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2) #128x250x250
      self.conv_8 = DoubleConv(16, 8)  #128x250x250

      self.upconv_4 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2) #64x502x502
      self.conv_9 = DoubleConv(8, 2)  #64x502x502

      self.output = nn.Conv2d(2, 1, kernel_size=1) #3x502x502

      self.fc1 = nn.Linear(502 * 502, 128)
      self.fc2 = nn.Linear(128, 1)

      self.relu = nn.ReLU()

    def forward(self, batch):

      conv_1_out = self.conv_1(batch)
      conv_2_out = self.conv_2(self.pool_1(conv_1_out))
      conv_3_out = self.conv_3(self.pool_2(conv_2_out))
      conv_4_out = self.conv_4(self.pool_3(conv_3_out))
      conv_5_out = self.conv_5(self.pool_4(conv_4_out))

      conv_6_out = self.conv_6(torch.cat([self.upconv_1(conv_5_out), conv_4_out], dim=1))
      conv_7_out = self.conv_7(torch.cat([self.upconv_2(conv_6_out), conv_3_out], dim=1))
      conv_8_out = self.conv_8(torch.cat([self.upconv_3(conv_7_out), conv_2_out], dim=1))
      conv_9_out = self.conv_9(torch.cat([self.upconv_4(conv_8_out), conv_1_out], dim=1))

      output = self.output(conv_9_out)
      output = output.view(-1, 502 * 502)

      output = self.relu(self.fc1(output))
      output = self.fc2(output).flatten()

      return output
