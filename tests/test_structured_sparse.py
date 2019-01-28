import pytest
import numpy as np
import torch
from dst.structured_dense import *
from dst.structured_sparse import *

x = SparseParameter(torch.Tensor(size=(3, 4)))

# TODO: write some meaningful tests for sparse parameter tensors, 2D and 4D
