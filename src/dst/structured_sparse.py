import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .structured_dense import StructuredDenseParameter, DenseParameter


class Grouping(object):
    r"""
    A grouping object responsible for two methods given a tensor shape:
    - Reduction over groups, used to e.g. compute Lp-norm
    - Expansion over groups, used to e.g. set group values to a specified value given a group mask (i.e. group pruning if the value is 0.)
    """
    def __init__(self, size):
        self.shape = torch.Size(size)

    def size(self):
        return self.shape

    @property
    def num_groups(self):
        raise NotImplementedError

    @property
    def group_sizes(self):
        raise NotImplementedError

    # def group_reduce(self, fn):
    #     raise NotImplementedError

    # def group_expand(self, fn):
    #     raise NotImplementedError

    def group_lp_reduce(self, p=1):
        # return self.group_reduce(lambda x: torch.norm(x, p=p))
        raise NotImplementedError

    def check_group_mask_len(self, group_mask):
        assert group_mask.numel()==self.num_groups, "group mask size mismatch"

    def group_mask_expand(self, group_mask):
        # self.check_group_mask_len(group_mask)
        # return self.group_expand(group_mask)
        raise NotImplementedError


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


class ElementGrouping(Grouping):
    r"""
    Trivial grouping, each element is a group
    """
    def __init__(self, size):
        super(ElementGrouping, self).__init__(size)

    @property
    def num_groups(self):
        return int(np.prod(self.shape))

    @property
    def group_sizes(self):
        return torch.LongTensor(self.num_groups).fill_(1)

    def group_lp_reduce(self, p=1):
        return torch.abs

    def group_mask_expand(self, group_mask):
        return group_mask.view(self.shape)


class BlockGrouping(Grouping):
    r"""
    Block grouping
    """
    def __init__(self, size, block_size=1):
        super(BlockGrouping, self).__init__(size)
        self.block_size = block_size

    @property
    def num_groups(self):
        return np.prod(self.shape)

    @property
    def group_sizes(self):
        return torch.LongTensor(self.num_groups).fill_(1)

    def group_lp_reduce(self, p=1):
        return torch.abs

    def group_mask_expand(self, group_mask):
        return group_mask.view(self.shape)


class DimGrouping(Grouping):
    r"""
    Grouping along certain dimensions
    """
    def __init__(self, size, dim=(0,)):
        super(DimGrouping, self).__init__(size)
        self.dim = dim
        self.cdim = tuple(set(range(len(size))) - set(dim))

    @property
    def num_groups(self):
        return np.prod([self.shape(d) for d in self.cdim])

    @property
    def group_sizes(self):
        pass # TODO

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

    @property
    def group_sizes(self):
        return torch.LongTensor(self.num_groups).fill_(
            (self.num_gates + 2) * self.num_hidden
        )


class StructuredSparseParameter(nn.Module):
    r"""
    Most general sparse parameter: a structured dense tensor masked by a structured mask
    """
    def __init__(self, dense, grouping=ElementGrouping):
        super(StructuredSparseParameter, self).__init__()
        assert isinstance(dense, StructuredDenseParameter), "need a StructuredDenseParameter to wrap around"
        self.dense = dense
        self.shape = self.dense.shape
        self.groups = grouping(self.shape)

        self.register_buffer('group_mask', torch.ByteTensor(self.groups.num_groups))
        self.register_buffer('mask', torch.ByteTensor(size=self.shape))
        self.init_params()

    def size(self):
        return self.shape

    def compute_mask_(self):
        self.mask = self.groups.group_mask_expand(self.group_mask)

    def compute_group_lp_(self, p=1):
        self.group_lp = self.groups.group_lp_reduce(p=p)(self.dense()).view(-1)

    def init_params(self):
        self.group_mask.fill_(1)
        self.compute_mask_()
        self.compute_group_lp_()
        self.dense.init_params()

    def sparsity(self):
        n_nonzero, n_total = self.param_count()
        return 1. - float(n_nonzero) / n_total

    def param_count(self):
        return int(self.mask.sum()), self.mask.numel()

    def forward(self):
        return self.dense() * self.mask.float()

    def prune_by_threshold(self, threshold, p=1):
        r"""
        Prune groups based on Lp-norm compared to a threshold
        """
        sparsity_before = self.sparsity()
        self.compute_group_lp_(p=p)
        self.group_mask[self.group_lp < threshold] = 0
        self.compute_mask_()
        sparsity_after = self.sparsity()
        return sparsity_before, sparsity_after

    def prune_to_sparsity(self, sparsity, p=1):
        # Prune groups with smallest Lp-norm until at least sparsity
        self.compute_group_lp_(p=p)  # fresh computation of self.group_lp
        idx = self.group_lp.argsort()  # sort self.group_lp
        for i in idx[self.group_mask[idx]]:  # only those 1's
            if self.sparsity() < sparsity:
                self.group_mask[i].fill_(0) # prune the i-th group away
                self.compute_mask_()
            else:
                return self.sparsity()
        return 1.

    def grow_to_sparsity(self, sparsity, reset_value=0.):
        # Grow groups randomly until at most sparsity
        growth_mask = torch.zeros_like(self.group_mask)
        idx = torch.randperm(self.groups.num_groups) # randomized group order
        for i in idx[1-self.group_mask[idx]]:  # only those 0's
            if self.sparsity() > sparsity:
                self.group_mask[i].fill_(1) # grow the i-th group back
                growth_mask[i].fill_(1)
                self.compute_mask_()
            else:
                # NOTE: This is a very hacky thing: clamp parameter values to reset_value for the newly grown weights, ONLY IF the underlying dense parameter is unstructured
                if isinstance(self.dense, DenseParameter):
                    self.dense.clamp_values(
                        mask=self.groups.group_mask_expand(growth_mask),
                        value=reset_value)
                return self.sparsity()
        return 0.

    def prune_or_grow_to_sparsity(self, sparsity, p=1, reset_value=0.):
        r"""
        The main sparse reparameterization function, which
        - (if current sparsity < target sparsity) prunes by group Lp-norm until at least the target sparsity
        - (if current sparsity > target sparsity) grows randomly until at most the target sparsity
        NOTE: the per-group iteration of pruning and growth could be stupidly slow when a large number of groups are to be pruned or grown, but this ensures correctness easily
        TODO: make it more efficient
        """
        sparsity_before = self.sparsity()
        if sparsity_before < sparsity:
            sparsity_after = self.prune_to_sparsity(sparsity, p=p)
        elif sparsity_before > sparsity:
            sparsity_after = self.grow_to_sparsity(sparsity, reset_value=reset_value)
        # print("Reparameterized from sparsity {} to {}".format(
        #     sparsity_before, sparsity_after))
        return sparsity_before, sparsity_after


# Alias for the dumbest sparse parameter tensor
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