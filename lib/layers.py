import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm3d(out_channels)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_short = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

        self.apply(initialize_weights)

    def forward(self, x):
        res = self.norm1(x)
        res = self.act1(res)

        res = self.conv1(res)
        res = self.norm2(res)
        res = self.act2(res)

        res = self.conv2(res)
        res = self.norm3(res)
        res = self.act3(res)

        x = res + self.conv_short(x)

        return x

class Upscale3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.residual = ResidualBlock3D(in_channels + out_channels, out_channels)

        self.apply(initialize_weights)

    def forward(self, x, skip):
        x = self.up(x)
        x = th.cat([x, skip], 1)
        x = self.residual(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, normalization=nn.GroupNorm, num_groups=8, activation=nn.PReLU):
        super(ResidualBlock, self).__init__()

        if type(normalization) == nn.GroupNorm: self.norm1 = normalization(num_groups, in_channels)
        else: self.norm1 = normalization(in_channels)

        self.act1 = activation(inplace=True)

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        
        if type(normalization) == nn.GroupNorm: self.norm2 = normalization(num_groups, in_channels)
        else: self.norm2 = normalization(in_channels)

        self.act2 = activation(inplace=True)

        # pointwise convolution
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, padding=1, bias=False)

        self.apply(initialize_weights)

    def forward(self, x, res):
        x = th.cat((x, res), 1)

        x = self.norm(x)
        x = self.act1(x)

        x = self.conv1(x)
        x = self.norm1(x)

        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        
        return F.pad(x, (1, 0, 1, 0, 1, 0))

class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, normalization=nn.GroupNorm, activation=nn.PReLU, num_groups=8, dropout=0.01):
        super(Upscale, self).__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if type(normalization) == nn.GroupNorm: self.norm = normalization(num_groups, out_channels)
        else: self.norm = normalization(out_channels)
        
        self.act = activation(inplace=True)

        self.dropout = nn.Dropout(dropout)

        self.apply(initialize_weights)
    
    def forward(self, x, res):
        x = th.cat((x, res), 1)

        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, activation=nn.PReLU, normalization=nn.GroupNorm, num_groups=8, dropout=0.01):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        if type(normalization) == nn.GroupNorm: self.norm = normalization(num_groups, out_channels)
        else: self.norm = normalization(out_channels)

        self.act = activation(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.apply(initialize_weights)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x
    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, normalization_size, growth_rate, dropout=0.01):
        super().__init__()

        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv3d(in_channels, normalization_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(normalization_size * growth_rate)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv3d(normalization_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        if dropout: self.dropout = nn.Dropout(dropout)
        else: self.dropout = None
    
    def forward(self, x):
        res = self.bn1(x)
        res = self.act1(res)
        res = self.conv1(res)
        res = self.bn2(res)
        res = self.act2(res)
        res = self.conv2(res)
        
        if self.dropout != None: res = self.dropout(res)

        return th.cat((x, res), 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, layers, normalization_size, growth_rate, dropout=0.0):
        super().__init__()

        dense_block = []

        for i in range(layers):
            dense_block.append(
                DenseLayer(in_channels + i * growth_rate, normalization_size, growth_rate, dropout)
            )
        
        self.net = nn.Sequential(*dense_block)

    def forward(self, x):
        return self.net(x)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()

        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        if pooling: self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        else: self.pool1 = None
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        if self.pool1 != None: self.pool1(x)

        return x

def initialize_weights(m):
    if type(m) in [nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d]:
        th.nn.init.xavier_uniform_(m.weight)
        if type(m.bias) == th.Tensor: th.nn.init.xavier_uniform_(m.bias)

    elif type(m) ==  nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        th.nn.init.constant_(m.bias, 0)