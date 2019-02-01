from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
from .structured_sparse import StructuredSparseParameter
from .utils import param_count


class DSModel(nn.Module):
    r"""
    A dynamic sparse network wrapper for models with dynamic sparse parameters
    """

    def __init__(self,
                 model,
                 target_sparsity=0.9,
                 target_fraction_to_prune=1e-2,
                 pruning_threshold=1e-3):
        super(DSModel, self).__init__()
        self.model = model
        self.target_sparsity = target_sparsity
        self.target_fraction_to_prune = target_fraction_to_prune
        self.pruning_threshold = pruning_threshold
        self.update_stats(init=True)
        self.num_sparse_parameters = len(self.breakdown)
        self.prune_or_grow_to_sparsity(sparsity=self.target_sparsity_in_sparse())
        self.shuffle_structure()
        # TODO: make this random pruning

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def target_sparsity_in_sparse(self):
        return self.target_sparsity * self.np_total / self.np_sparse

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
        if not init:  # keep previous stats
            self._breakdown = self.breakdown.copy()
            self._sparsity = self.sparsity
        self.breakdown = {
            n: m.param_count()
            for n, m in self.named_sparse_parameters()
        }
        self.np_total = param_count(self.model)
        self.np_nonzero = sum([_n for _, (_n, _) in self.breakdown.items()])
        self.np_sparse = sum([_n for _, (_, _n) in self.breakdown.items()])
        self.np_dense = self.np_total - self.np_sparse
        self.sparsity = float(self.np_sparse - self.np_nonzero) / self.np_total

    def shuffle_structure(self):
        for sp in self.sparse_parameters():
            sp.shuffle_mask()

    def adjust_pruning_threshold(self, tolerance=0.1, gain=2.):
        # Adjust pruning threshold
        pruned_fraction = self.sparsity - self._sparsity
        if pruned_fraction < self.target_fraction_to_prune * (1 - tolerance):
            self.pruning_threshold *= 2.
        elif pruned_fraction > self.target_fraction_to_prune * (1 + tolerance):
            self.pruning_threshold /= 2.
        tqdm.write("adjust_pruning_threshold: -> {:f}".format(
            self.pruning_threshold))

    def prune_by_threshold(self, p=1):
        # Prune all sparse parameters by a global threshold on Lp-norm
        changes = {
            n: m.prune_by_threshold(self.pruning_threshold, p=p)
            for n, m in self.named_sparse_parameters()
        }
        self.update_stats()
        tqdm.write("prune_by_threshold: {:6.4f} -> {:6.4f}".format(
            self._sparsity, self.sparsity
        ))
        return changes

    def prune_or_grow_to_sparsity(self, sparsity, p=1):
        # Prune or grow all sparse parameters to a target sparsity
        # NOTE: this is sparsity in all sparse parameters
        if isinstance(sparsity, float):
            sparsity = [sparsity] * self.num_sparse_parameters
        changes = {
            n: m.prune_or_grow_to_sparsity(s, p=p)
            for s, (n, m) in zip(sparsity, self.named_sparse_parameters())
        }
        self.update_stats()
        tqdm.write("prune_or_grow_to_sparsity: {:6.4f} -> {:6.4f}".format(
            self._sparsity, self.sparsity
        ))
        return changes

    def reallocate_free_parameters(self, p=1, heuristic=None):
        # Reallocate parameters until model reaches at most a target sparsity
        if self.sparsity > self.target_sparsity: # only when there is free parameters to reallocate
            N = [n for _, (_, n) in self.breakdown.items()]  # total non-zero parameter count
            M = [m for _, (m, _) in self._breakdown.items()]  # old non-zero parameter count
            # K = [_m - m for (_, (_m, _)), (_, (m, _)) in zip(self._breakdown.items(), self.breakdown.items())]
            R = [r for _, (r, _) in self.breakdown.items()]  # numbers of surviving weights
            F = self.np_total * (self.sparsity - self.target_sparsity) # number of free parameters
            G = [F * r / sum(R) for r in R] # number of growth (TODO: make it a general heuristic)
            S = [1. - (g + r) / n for g, r, n in zip(G, R, N)] # target sparsities
            return self.prune_or_grow_to_sparsity(sparsity=S, p=p)

    def reparameterize(self):
        self.prune_by_threshold()
        self.adjust_pruning_threshold()
        self.reallocate_free_parameters()