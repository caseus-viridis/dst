import pytest
import numpy as np
import torch
from dst.structured_dense import *
from dst.structured_sparse import *

x = StructuredSparseParameter(
    dense=DenseParameter(
        torch.Tensor(size=(3, 4))
    ),
    grouping=ElementGrouping
)

