import torch
import torch.nn as nn
from dst.models import (
    mnist_mlp,
    cifar_wrn,
    cifar_resnet
)
from dst.modules import DSConv2d
from dst.reparameterization import param_count, get_sparse_param_stats
from dst.utils import param_count

# net = cifar10_wrn.net(width=2).cuda()
# print(net)
# n_total, n_dense, n_sparse, n_nonzero, breakdown = get_sparse_param_stats(net)
# print("Total parameter count = {}".format(n_total))
# print("Dense parameter count = {}".format(n_dense))
# print("Sparse parameter count = {}".format(n_sparse))
# print("Nonzero sparse parameter count = {}".format(n_nonzero))

# net(torch.rand(64, 3, 32, 32).cuda())

# ca = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)

# cb = nn.Sequential(
#     nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1, bias=False),
#     nn.ConvTranspose2d(
#         64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))

model = cifar_resnet.resnet34(spatial_bottleneck=False)
model_ = cifar_resnet.resnet34(spatial_bottleneck=True)
x = torch.rand(16, 3, 32, 32)

import ipdb; ipdb.set_trace()