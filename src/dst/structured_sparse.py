import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .structured_dense import StructuredDenseParameter, DenseParameter


class Grouping(object):
    r"""
    A grouping object responsible for two methods given a tensor shape:
    - Reduction over groups, used to e.g. compute Lp-norm
    - Expansion over groups, used to e.g. set group values to a specified value given a group mask (i.e. group pruning if the value is 0.)
    """
    def __init__(self, size):
        self.size = torch.Size(size)

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
        return group_mask.view(self.size)


class DimGrouping(Grouping):
    r"""
    Grouping along certain dimensions
    """
    def __init__(self, size, dim=(0,)):
        super(DimGrouping, self).__init__(size)
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
    def __init__(self, dense, grouping=ElementGrouping):
        super(StructuredSparseParameter, self).__init__()
        assert isinstance(dense, StructuredDenseParameter), "need a StructuredDenseParameter to wrap around"
        self.dense = dense
        self.size = self.dense.size
        self.groups = grouping(self.shape)

        self.register_buffer('group_mask', torch.ByteTensor(self.groups.num_groups))
        self.register_buffer('mask', torch.ByteTensor(size=self.shape))
        self.init_params()

    @property
    def shape(self):
        return self.dense.shape

    def compute_mask_(self):
        # NOTE: side effect!
        self.mask = self.groups.group_mask_expand(self.group_mask)

    def init_params(self):
        self.group_mask.fill_(1)
        self.compute_mask_()
        self.dense.init_params()

    @property
    def sparsity(self):
        return 1. - self.mask.sum().float() / self.mask.numel()

    def forward(self):
        return self.dense() * self.mask.float()

    def prune_by_threshold(self, threshold):
        raise NotImplementedError # TODO

    def prune_random(self):
        raise NotImplementedError # TODO


# Alias for dumbest sparse parameter tensor
SparseParameter = lambda t: StructuredSparseParameter(
    dense=DenseParameter(t), 
    grouping=ElementGrouping
)

if __name__=="__main__":
    # par = DenseParameter(torch.rand(8))
    # par = HashedParameter(shape=[8, 8], bank=torch.rand(6), seed=0)
    # gg = DimGrouping((4, 2, 3, 3), dim=1)
    # g = ElementGrouping(size=(3, 4))
    # import ipdb; ipdb.set_trace()
    pass