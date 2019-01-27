import pytest
import numpy as np
import torch 
from dst.utils import *

@pytest.mark.parametrize(
    "sz", (
        [16, 64], 
        [32, 4, 8, 16]
    )
)
@pytest.mark.parametrize(
    "dof", (
        2, 16, 128
    )
)
def test_rand_placement(sz, dof):
    pl = rand_placement(shape=sz, dof=dof, seed=7734)
    assert pl.shape==torch.Size(sz)
    assert pl.max() < dof
    assert pl.min() > -1
    pl_ = rand_placement(shape=sz, dof=dof, seed=666)
    assert not np.array_equal(pl, pl_)
    pl_ = rand_placement(shape=sz, dof=dof, seed=7734)
    assert np.array_equal(pl, pl_)
