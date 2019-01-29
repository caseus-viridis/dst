import torch
from dst.models import (
    mnist_mlp,
    cifar10_wrn
)
from dst.modules import DSConv2d

# net = cifar10_wrn.net()

# net(torch.rand(4, 784))

layer = DSConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True)

# import ipdb; ipdb.set_trace()