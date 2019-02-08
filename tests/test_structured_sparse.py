import pytest
import numpy as np
import torch
from dst.structured_dense import *
from dst.structured_sparse import *

# x = SparseParameter(torch.randn(size=(32, 32)))
# _, sp = x.prune_by_threshold(threshold=0.01)
# print("Sparsity = {}".format(sp))
# sp = x.prune_to_sparsity(sparsity=0.9)
# print("Sparsity = {}".format(sp))
# sp = x.grow_to_sparsity(sparsity=0.6)
# print("Sparsity = {}".format(sp))


# TODO: write some meaningful tests for sparse parameter tensors, 2D and 4D
