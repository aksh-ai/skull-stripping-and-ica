# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import torch as th
import torch.nn as nn
from lib.layers import *
import torch.nn.functional as F

class ResidualUNET3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, blocks: list or tuple = [64, 128, 256, 512], kernel_size: int or list or tuple = 3, stride: int or list or tuple = 2, optional_skip: bool = False):
        '''
        Residualization based 3D segmentation architecture inspired by UNET

        Args:
          * in_channels: Input channels present in the image
          * out_channels: Output channels for the segmented image
          * blocks: List of units containing size of residual blocks
          * kernel_size: Kernel size to be sued for feature extraction
          * stride: Stride size for skipping windows/pixels when applying the filter
        
        Returns:
          * forward(x): Tensor containing segmented image's mask or classes
        '''
        super(ResidualUNET3D, self).__init__()

        # initial convolution with norm and activation
        self.conv1_1 = nn.Conv3d(in_channels, blocks[0], kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(blocks[0])
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # second stage initial pointwise convolution - optional skip (minimize parameters by skipping a residual op? norm and activation are applied in the beginning of every residual op)
        self.optional_skip = optional_skip
        
        self.conv1_2 = nn.Conv3d(blocks[0], blocks[0], kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv1_short = nn.Conv3d(in_channels, blocks[0], kernel_size=1, stride=1, padding=0, bias=False)

        # residual blocks for downsampling
        self.residual_block1 = ResidualBlock3D(blocks[0], blocks[1], kernel_size=kernel_size, stride=stride)
        self.residual_block2 = ResidualBlock3D(blocks[1], blocks[2], kernel_size=kernel_size, stride=stride)
        self.residual_block3 = ResidualBlock3D(blocks[2], blocks[3], kernel_size=kernel_size, stride=stride)

        # upscaling blocks with residualization
        self.upscale_block1 = Upscale3D(blocks[3], blocks[2])
        self.upscale_block2 = Upscale3D(blocks[2], blocks[1])
        self.upscale_block3 = Upscale3D(blocks[1], blocks[0])

        # output layer with sigmoid activation 
        # output 2 channels for foreground, background, N channels for class based segmentation, and 1 for mask
        self.out = nn.Conv3d(blocks[0], out_channels, kernel_size=1, padding=0, bias=False)
        self.act2 = nn.Sigmoid()

        # apply xaiver weights init
        self.apply(initialize_weights)

    def forward(self, x):
        # intial feature extraction
        res = self.conv1_1(x)
        res = self.norm1(res)
        res = self.act1(res)
        
        # add residual to featurized tensor
        if self.optional_skip:
            skip1 = self.conv1_2(res) + self.conv1_short(x)
        else:
            skip1 = x + res

        # downsample based on residualization - 3 levels
        skip2 = self.residual_block1(skip1)
        skip3 = self.residual_block2(skip2)
        skip4 = self.residual_block3(skip3)

        # upscale and add downsampled tensors at corresponding levels (3-1, 2-2, 1-3)
        up1 = self.upscale_block1(skip4, skip3)
        up2 = self.upscale_block2(up1, skip2)
        up3 = self.upscale_block3(up2, skip1)
        
        # apply output conv with activation
        out = self.out(up3)
        out = self.act2(out)

        return out

class DenseNet3D(nn.Module):
    def __init__(self, in_channels=1, num_features=64, depth_kernel_size=7, depth_stride=1, blocks=[6, 12, 24, 16], growth_rate=32, normalization_size=4, dropout=0.1, num_classes=1, add_top=False):
        '''
        DenseNet3D as backbone for the custom segmentation GAN
        '''
        super(DenseNet3D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=(depth_kernel_size, 7, 7), stride=(depth_stride, 2, 2), padding=(depth_kernel_size // 2, 3, 3), bias=False),
            nn.InstanceNorm3d(num_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        dt_blocks = []

        for i, layers in enumerate(blocks):
            dense_block = DenseBlock(num_features, layers, normalization_size, growth_rate, dropout)
            dt_blocks.append(dense_block)
            num_features = num_features + layers * growth_rate
            if i != len(blocks) - 1:
                transition_block = TransitionBlock(num_features, num_features//2)
                dt_blocks.append(transition_block)
                num_features = num_features // 2
        
        self.dt_blocks = nn.Sequential(*dt_blocks)

        self.bn = nn.InstanceNorm3d(num_features)

        if add_top: self.fc = nn.Linear(num_features, num_classes)
        else: self.fc = None

        self.apply(initialize_weights)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dt_blocks(x)
        x = self.bn(x)

        if self.fc != None:
            x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).view(x.shape[0], -1)
            x = self.fc(x)

        return x