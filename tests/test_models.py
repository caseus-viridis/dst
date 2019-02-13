import torch
import torch.nn as nn
import torch.nn.functional as F
from dst.models import (mnist_mlp, cifar_wrn, cifar_resnet)
from dst.modules import DSConv2d
from dst.reparameterization import DSModel
from dst.utils import param_count
from dst.activation_sparse import *

from dst.models.cresnet import *

# net = cifar10_wrn.net(width=2).cuda()
# print(net)
# n_total, n_dense, n_sparse, n_nonzero, breakdown = get_sparse_param_stats(net)
# print("Total parameter count = {}".format(n_total))
# print("Dense parameter count = {}".format(n_dense))
# print("Sparse parameter count = {}".format(n_sparse))
# print("Nonzero sparse parameter count = {}".format(n_nonzero))

# net(torch.rand(64, 3, 32, 32).cuda())

# ca = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
# cb = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0, bias=False)

# cb = nn.Sequential(
#     nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1, bias=False),
#     nn.ConvTranspose2d(
#         64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))

# model = cifar_resnet.resnet34(spatial_bottleneck=False)
# model_ = cifar_resnet.resnet34(spatial_bottleneck=True)

# ca = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)

# x = torch.rand(16, 3, 32, 32)

# conv = nn.Conv2d(1, 1, kernel_size=1, stride=2, padding=1, bias=False)
# x = torch.zeros(1, 1, 8, 8)
# x[:, :, 0::2, 0::2] = 1.
# y = torch.zeros(1, 1, 8, 8)
# y[:, :, 1::2, 1::2] = 1.
# z = x+y

# x = torch.rand(1, 1, 8, 8)
# k = torch.randn(1, 1, 3, 3)
# y = F.conv2d(x, k, stride=1, padding=1)
# z = F.conv2d(x, k, stride=2, padding=1)

# bn = SparseBatchNorm(16, Checker2d(4, quarters=2))
# x = torch.rand(1, 16, 4, 4)
# y = bn(x)

# TODO: Write some meaningful unit tests

