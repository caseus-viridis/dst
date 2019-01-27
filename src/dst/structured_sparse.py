import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
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


class Grouping(object):
    r"""
    A grouping object responsible for two methods given a tensor shape:
    - Reduction over groups, used to e.g. compute Lp-norm
    - Expansion over groups, used to e.g. set group values to a specified value given a group mask (i.e. group pruning if the value is 0.)
    """
    def __init__(self, shape):
        self.shape = torch.Size(shape)

    @property
    def num_groups(self):
        raise NotImplementedError

    # def group_reduce(self, fn):
    #     raise NotImplementedError

    # def group_expand(self, fn):
    #     raise NotImplementedError

    def group_lp_reduce(self, p=1):
        return self.group_reduce(lambda x: torch.norm(x, p=p))

    def check_group_mask_len(self, group_mask):
        assert group_mask.numel()==self.num_groups, "group mask size mismatch"

    def group_mask_expand(self, group_mask):
        self.check_group_mask_len(group_mask)
        return self.group_expand(group_mask)


class SumGrouping(Grouping):
    r"""
    Combines two groupings by union of groups
    resulting g.num_groups = ga.num_groups + gb.num_groups
    Used for e.g. channel pruning for conv parematers
    """
    def __init__(self, ga, gb):
        assert ga.shape==gb.shape, "shapes of groupings mismatch"
        super(SumGrouping, self).__init__(ga.shape)

    @property
    def num_groups(self):
        return ga.num_groups + gb.num_groups

    def group_reduce(self, fn):
        raise NotImplementedError

    def group_expand(self, fn):
        raise NotImplementedError


class ProductGrouping(Grouping):
    r"""
    Combines two groupings by group of unions
    resulting g.num_groups = ga.num_groups * gb.num_groups
    Used for e.g. ISS pruning for LSTM
    """

    def __init__(self, ga, gb):
        assert ga.shape==gb.shape, "shapes of groupings mismatch"
        super(ProductGrouping, self).__init__(ga.shape)

    @property
    def num_groups(self):
        return ga.num_groups * gb.num_groups

    def group_reduce(self, fn):
        raise NotImplementedError

    def group_expand(self, fn):
        raise NotImplementedError


class ElementGrouping(Grouping):
    r"""
    Trivial grouping, each element is a group
    """
    def __init__(self, size):
        super(ElementGrouping, self).__init__(size)
        self.size = size

    @property
    def num_groups(self):
        return np.prod(self.size)

    def group_lp_reduce(self, p=1):
        return torch.abs

    def group_mask_expand(self, group_mask):
        return group_mask # need to be reshaped


class DimGrouping(Grouping):
    r"""
    Grouping along certain dimensions
    """
    def __init__(self, size, dim=(0,)):
        super(DimsGrouping, self).__init__(size)
        self.size = size
        self.dim = dim
        self.cdim = tuple(set(range(len(size))) - set(dim))

    @property
    def num_groups(self):
        return np.prod([self.size(d) for d in self.cdim])

    def group_lp_reduce(self, p=1):
        lambda x: torch.norm(x, p=p, dim=self.cdim)

    def group_mask_expand(self, group_mask):
        return group_mask.unsqueeze().expand()


class ISSGrouping(Grouping):
    r"""
    Intrinsic sparse structure (ISS) grouping for RNN cells
    Wen et al. 2018 (http://arxiv.org/abs/1709.05027)
    """
    def __init__(self, num_hidden, num_gates=4):
        self.num_hidden = num_hidden
        self.num_gates = num_gates
        super(ISSGrouping, self).__init__((num_hidden*2, num_hidden*num_gates))

    @property
    def num_groups(self):
        return self.num_hidden ** 2

    def group_reduce(self):
        raise NotImplementedError # [TODO]

    def group_expand(self):
        raise NotImplementedError # [TODO]


class StructuredSparseParameter(nn.Module):
    def __init__(self, dense, groupings=(ElementGrouping,)):
        super(StructuredSparseParameter, self).__init__()
        assert isinstance(dense, StructuredDenseParameter), "need a StructuredDenseParameter to wrap around"
        self.dense = dense
        self.groups = grouping(self.size)
        self.group_mask = torch.ones(num_groups)

        self.register_buffer('mask', torch.Tensor(*self.size))
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

    def prune_by_threshold(self, threshold):
        pass

    def sparsity(self):
        pass


if __name__=="__main__":
    # par = DenseParameter(torch.rand(8))
    # par = HashedParameter(shape=[8, 8], bank=torch.rand(6), seed=0)
    # gg = DimGrouping((4, 2, 3, 3), dim=1)
    import ipdb; ipdb.set_trace()