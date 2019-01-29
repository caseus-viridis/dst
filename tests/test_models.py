import torch
from dst.models import (
    mnist_mlp,
    cifar10_wrn
)
from dst.modules import DSConv2d
from dst.dynamics import param_count, get_sparse_param_stats

net = cifar10_wrn.net(width=2).cuda()
print(net)
n_total, n_dense, n_sparse, n_nonzero, breakdown = get_sparse_param_stats(net)
print("Total parameter count = {}".format(n_total))
print("Dense parameter count = {}".format(n_dense))
print("Sparse parameter count = {}".format(n_sparse))
print("Nonzero sparse parameter count = {}".format(n_nonzero))

net(torch.rand(64, 3, 32, 32).cuda())

# import ipdb; ipdb.set_trace()
