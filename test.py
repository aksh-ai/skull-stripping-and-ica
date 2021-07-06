import torch as th
from lib.models import *

inp = th.rand(1, 1, 128, 128, 128)

model = ResidualUNET3D()
out = model(inp)

print(out)

# model = DenseNet(add_top=True).to('cuda')
# out = model(inp)

# print(out)
# up = th.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
# print(up(inp).shape)

# up = th.nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
# print(up(inp).shape)