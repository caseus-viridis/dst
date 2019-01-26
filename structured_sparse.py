import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import rand_placement


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


class Grouping(object):
    def __call__(self, *args, **kwargs):
        pass

class ElementGrouping(Grouping):
    pass


class DimGrouping(Grouping):
    pass


class ISSGrouping(Grouping):
    pass


class StructuredSparseParameter(nn.Module):
    def __init__(self, dense, grouping=ElementGrouping):
        super(StructuredSparseParameter, self).__init__()
        assert isinstance(dense, StructuredDenseParameter), "need a StructuredDenseParameter to wrap around"
        self.dense = dense
        self.groups = grouping(self.size)
        self.group_mask = torch.ones(num_groups)

        self.register_buffer('mask', torch.zeros(self.size).long())
        self.compute_mask()

    @property
    def size(self):
        return self.dense.size()

    def compute_mask(self):
        return union(f(self.groups)) # just a note

    def forward(self):
        return self.mask * self.dense

    def init_params(self):
        self.dense.init_params()

    def prune(self, threshold):
        pass

    def sparsity(self):
        pass


if __name__=="__main__":
    # par = DenseParameter(torch.rand(8))
    # par = HashedParameter(shape=[8, 8], bank=torch.rand(6), seed=0)
    import ipdb; ipdb.set_trace()