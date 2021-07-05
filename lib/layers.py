import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_groups=8, activation=nn.PReLU):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, in_channels)

        self.act = activation(inplace=True)

        # pointwise convolution
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, in_channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.norm1(res)
        
        res = self.act(res)

        res = self.conv1(res)
        res = self.norm2(res)
        
        return res + x

class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=2, padding=0):
        super(Upscale, self).__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self , x):
        return self.up(x)