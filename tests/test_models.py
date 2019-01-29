import torch
from dst.models import (
    mnist_mlp,
    cifar10_wrn
)
from dst.modules import DSConv2d

net = cifar10_wrn.net()
net(torch.rand(64, 3, 32, 32))

# import ipdb; ipdb.set_trace()