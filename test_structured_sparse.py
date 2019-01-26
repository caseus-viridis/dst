import pytest
import numpy as np
import torch
from structured_sparse import *

@pytest.mark.parametrize(
    "sz", (
        [16, 64],
        [32, 16, 3, 3]
    )
)
def test_shape_dof(sz):
    par = DenseParameter(torch.rand(*sz))
    assert par.shape==torch.Size(sz)
    assert par.dof==np.prod(sz)
    assert par().shape==par.shape


@pytest.mark.parametrize("sz", ([16, 64], [32, 4, 8, 16]))
@pytest.mark.parametrize("dof", (2, 16, 128))
def test_hashed(sz, dof):
    par = HashedParameter(shape=sz, bank=torch.rand(dof), seed=666)
    assert par.shape==torch.Size(sz)
    assert par.bank.shape==torch.Size((dof,))
    assert par().shape==par.shape