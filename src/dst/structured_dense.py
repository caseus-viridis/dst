import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import rand_placement


class StructuredDenseParameter(nn.Module):
    r"""
    A structured dense parameter tensor
        size: size of the output dense tensor
        bank: contains true parameters
        transform: a function reparameterizing bank to a tensor of size size
    """
    def __init__(self, size, bank, transform):
        super(StructuredDenseParameter, self).__init__()
        self.shape = torch.Size(size)
        self.bank = nn.Parameter(data=bank)

    @property
    def dof(self):
        return self.bank.numel()

    def size(self):
        return self.shape

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
            size=tensor.size(),
            bank=tensor,
            transform=self.transform
        )

    def init_params(self):
        nn.init.kaiming_uniform_(self.bank, a=math.sqrt(5))

    def clamp_values(self, mask, value=0.):
        """
        Clamp bank values specified by mask, use with extreme caution!
        """
        self.bank.data[mask] = value


class HashedParameter(StructuredDenseParameter):
    r"""
    A dense parameter tensor computed with hashing
    Chen et al. 2015 (http://arxiv.org/abs/1504.04788)
    """
    def __init__(self, size, bank, seed=0):
        self.transform = lambda x:torch.embedding(
            bank, rand_placement(size, bank.numel(), seed=seed)
        )
        super(HashedParameter, self).__init__(
            size=size,
            bank=bank,
            transform=self.transform
        )

    def init_params(self):
        nn.init.kaiming_uniform_(self.bank, a=math.sqrt(5))


class LowDisplacementRankParameter(StructuredDenseParameter):
    pass # TODO
