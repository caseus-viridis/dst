import tqdm
import itertools
import torch
import torch.nn as nn
from .structured_sparse import StructuredSparseParameter
from .utils import param_count


class DSModel(nn.Module):
    r"""
    A dynamic sparse network wrapper for models with dynamic sparse layers
    """
    def __init__(self, model):
        super(DSModel, self).__init__()
        self.model = model
        self.pruning_threshold = 0.
        self.update_stats(init=True)
        self.num_sparse_parameters = len(self.breakdown)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sparse_parameters(self):
        # An iterator over sparse parameters
        for n, m in self.named_sparse_parameters():
            yield m

    def named_sparse_parameters(self):
        # An iterator over sparse parameters and names
        for n, m in self.named_modules():
            if isinstance(m, StructuredSparseParameter):
                yield n, m

    def update_stats(self, init=False):
        # Update sparse parameter statistics
        if not init:
            self._breakdown = self.breakdown.copy()  # keep previous
        self.breakdown = {
            n: m.param_count()
            for n, m in self.named_sparse_parameters()
        }
        self.np_total = param_count(self.model)
        self.np_nonzero = sum([_n for _, (_n, _) in self.breakdown.items()])
        self.np_sparse = sum([_n for _, (_, _n) in self.breakdown.items()])
        self.np_dense = self.np_total - self.np_sparse
        self.sparsity = float(self.np_sparse - self.np_nonzero) / self.np_total

    def prune_by_threshold(self, threshold, p=1):
        # Prune all sparse parameters by a global threshold on Lp-norm
        changes = {
            n: m.prune_by_threshold(threshold, p=p)
            for n, m in self.named_sparse_parameters()
        }
        self.update_stats()
        return changes

    def prune_or_grow_to_sparsity(self, sparsity, p=1):
        # Prune or grow all sparse parameters to a target sparsity
        if isinstance(sparsity, float):
            sparsity = [sparsity] * self.num_sparse_parameters
        changes = {
            n: m.prune_or_grow_to_sparsity(s, p=p)
            for s, (n, m) in zip(sparsity, self.named_sparse_parameters())
        }
        self.update_stats()
        return changes

    def reallocate_free_parameters(self, target_sparsity, p=1, heuristic=None):
        # Reallocate parameters until model reaches at most a target sparsity
        if self.sparsity > target_sparsity: # only when there is free parameters to reallocate
            N = [n for _, (_, n) in self.breakdown.items()]  # total non-zero parameter count
            M = [m for _, (m, _) in self._breakdown.items()]  # old non-zero parameter count
            # K = [_m - m for (_, (_m, _)), (_, (m, _)) in zip(self._breakdown.items(), self.breakdown.items())]
            R = [r for _, (r, _) in self.breakdown.items()]  # numbers of surviving weights
            F = self.np_total * (self.sparsity - target_sparsity) # number of free parameters
            G = [F * r / sum(R) for r in R] # number of growth (TODO: make it a general heuristic)
            S = [1. - (g + r) / n for g, r, n in zip(G, R, N)] # target sparsities
            return self.prune_or_grow_to_sparsity(sparsity=S, p=p)


def set_point_control(self, old_threshold,
                                num_pruned,
                                target_num_pruned,
                                tolerance=0.1,
                                gain=2.):
    # TODO
    if (num_pruned/target_num_pruned):
        pass
    return new_threshold
