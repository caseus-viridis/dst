import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import rand_placement


class StructuredDenseParameter(nn.Module):
    r"""
    A structured dense parameter tensor
        shape: size of the output dense tensor
        bank: contains true parameters
        transform: a function reparameterizing bank to a tensor of size size
    """
    def __init__(self, shape, bank, transform):
        super(StructuredDenseParameter, self).__init__()
        self.shape = torch.Size(shape)
        self.bank = nn.Parameter(data=bank)

    @property
    def dof(self):
        return self.bank.numel()

    def forward(self):
        dense = self.transform(self.bank)
        assert dense.shape==self.shape, "reparameterized dense tensor shape {} does not match specified shape {}".format(dense.shape, self.shape)
        return dense

    def init_params(self):
        raise NotImplementedError

    def extra_repr(self):
        return "shape = {}, dof = {}".format(self.shape, self.dof)


class DenseParameter(StructuredDenseParameter):
    r"""
    A trivial dense parameter tensor
    """
    def __init__(self, tensor):
        self.transform = lambda x:x
        super(DenseParameter, self).__init__(
            shape=tensor.size(),
            bank=tensor,
            transform=self.transform
        )

    def init_params(self):
        pass # TODO


class HashedParameter(StructuredDenseParameter):
    r"""
    A dense parameter tensor computed with hashing
    Chen et al. 2015 (http://arxiv.org/abs/1504.04788)
    """
    def __init__(self, shape, bank, seed=0):
        self.transform = lambda x:torch.embedding(
            bank, rand_placement(shape, bank.numel(), seed=seed)
        )
        super(HashedParameter, self).__init__(
            shape=shape,
            bank=bank,
            transform=self.transform
        )

    def init_params(self):
        pass # TODO
