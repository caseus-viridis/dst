from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
from .structured_sparse import StructuredSparseParameter
from .utils import param_count
from prettytable import PrettyTable as pt
from .visualization import mask2braille


class ReallocationHeuristics(object):
    r"""
    A collection of free parameter reallocation heuristics
    A heuristic has the signature: gamma = heuristic(N, M, R), where
        N is a list of total parameter counts of sparse parameters,
        M is the counts of old non-zero parameters in each, and
        R is the counts of new non-zero parameters in each;
        it returns gamma, a list of fractions to reallocate outstanding free parameters, all elements in gamma should sum to 1.
    """

    @staticmethod
    def R_p_sphere(N, M, R, p=1):
        Rp = R ** p
        return Rp / Rp.sum()

    @staticmethod
    def M_p_sphere(N, M, R, p=1):
        Mp = M**p
        return Mp / Mp.sum()

    @staticmethod
    def Z_p_sphere(N, M, R, p=1):
        Zp = (N - R)**p
        return Zp / Zp.sum()

    @staticmethod
    def N_p_sphere(N, M, R, p=1):
        Np = N**p
        return Np / Np.sum()

    @staticmethod
    def K_p_sphere(N, M, R, p=1):
        Kp = (M - R) ** p
        return Kp / Kp.sum()

    @staticmethod
    def paper(N, M, R):
        # Heuristic of Mostafa & Wang 2018a,b
        # (https://openreview.net/pdf?id=S1xBioR5KX and https://openreview.net/pdf?id=BygIWTMdjX)
        return ReallocationHeuristics.R_p_sphere(N, M, R, p=1)

    @staticmethod
    def within_param(N, M, R):
        # Heuristic of no reallocation across parameter tensors, only within
        return ReallocationHeuristics.K_p_sphere(N, M, R, p=1)

    @staticmethod
    def anti_paper(N, M, R):
        # Heuristic of no reallocation across parameter tensors, using complementary weight counts as the paper
        return ReallocationHeuristics.Z_p_sphere(N, M, R, p=1)


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
        self.stats_table = pt(
            field_names=[
                'Parameter tensor', '# total', '# nonzero', 'sparsity' #, 'snapshot'
            ],
            float_format='.4',
            align='r')
        self.sum_table = pt(
            field_names=[
                '# total parameters', '# sparse parameters',
                '# dense parameters', '# nonzero parameters in sparse',
                '# free parameters', 'sparsity'
            ],
            float_format='.4',
            align='r')
        self.update_stats(init=True)
        self.num_sparse_parameters = len(self.breakdown)
        self.randomly_thin_to_group_sparsity(
            group_sparsity=self.target_sparsity_in_sparse())

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def target_sparsity_in_sparse(self):
        return self.target_sparsity * self.np_total / self.np_sparse

    def sparse_parameters(self):
        # An iterator over sparse parameters
        for _, m in self.named_sparse_parameters():
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
        self.stats_table.clear_rows()
        # for n, m in self.named_sparse_parameters():
        #     _nz, _tot = self.breakdown[n]
        #     self.stats_table.add_row([
        #         n, _tot, _nz, 1. - _nz/_tot, mask2braille(m.mask)
        #     ])
        for n, (_nz, _tot) in self.breakdown.items():
            self.stats_table.add_row([
                n, _tot, _nz, 1. - _nz/_tot
            ])

        self.np_total = param_count(self.model)
        self.np_nonzero = sum([_n for _, (_n, _) in self.breakdown.items()])
        self.np_sparse = sum([_n for _, (_, _n) in self.breakdown.items()])
        self.np_dense = self.np_total - self.np_sparse
        self.np_free = self.np_dense + self.np_nonzero
        self.sparsity = float(self.np_sparse - self.np_nonzero) / self.np_total
        self.sum_table.clear_rows()
        self.sum_table.add_row([
            self.np_total,
            self.np_sparse,
            self.np_dense,
            self.np_nonzero,
            self.np_free,
            self.sparsity
        ])

    def shuffle_structure(self):
        for sp in self.sparse_parameters():
            sp.shuffle_mask()

    def randomly_thin_to_group_sparsity(self, group_sparsity):
        for sp in self.sparse_parameters():
            sp.randomly_thin_to_group_sparsity(group_sparsity)

    def adjust_pruning_threshold(self, tolerance=0.1, gain=2.):
        # Adjust pruning threshold
        pruned_fraction = self.sparsity - self._sparsity
        if pruned_fraction < self.target_fraction_to_prune * (1 - tolerance):
            self.pruning_threshold *= gain
        elif pruned_fraction > self.target_fraction_to_prune * (1 + tolerance):
            self.pruning_threshold /= gain
        # tqdm.write("adjust_pruning_threshold: -> {:f}".format(
        # self.pruning_threshold))

    def prune_by_threshold(self, p=1):
        # Prune all sparse parameters by a global threshold on Lp-norm
        changes = {
            n: m.prune_by_threshold(self.pruning_threshold, p=p)
            for n, m in self.named_sparse_parameters()
        }
        self.update_stats()
        # tqdm.write("prune_by_threshold: {:6.4f} -> {:6.4f}".format(
        #     self._sparsity, self.sparsity
        # ))
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
        # tqdm.write("prune_or_grow_to_sparsity: {:6.4f} -> {:6.4f}".format(
        #     self._sparsity, self.sparsity
        # ))
        return changes

    def reallocate_free_parameters(self, heuristic, p=1):
        # Reallocate parameters until model reaches at most a target sparsity
        if self.sparsity > self.target_sparsity: # only when there is free parameters to reallocate
            N = torch.Tensor([n for _, (_, n) in self.breakdown.items()])  # total parameter count
            M = torch.Tensor([m for _, (m, _) in self._breakdown.items()])  # old non-zero parameter count
            # K = torch.Tensor([_m - m for (_, (_m, _)), (_, (m, _)) in zip(self._breakdown.items(), self.breakdown.items())])  # numbers of pruned weights
            R = torch.Tensor([r for _, (r, _) in self.breakdown.items()])  # numbers of surviving weights, i.e. new non-zero parameter count
            F = self.np_total * (self.sparsity - self.target_sparsity) # number of free parameters
            G = F * heuristic(N, M, R)  # number of growth
            S = 1. - (G + R) / N # [1. - (g + r) / n for g, r, n in zip(G, R, N)] # target sparsities
            return self.prune_or_grow_to_sparsity(sparsity=S, p=p)

    def reparameterize(self, heuristic=ReallocationHeuristics.paper, p=1):
        # Following our paper, this is a whole-sale package to execute every few hundred batches
        self.prune_by_threshold(p=p)
        self.adjust_pruning_threshold()
        self.reallocate_free_parameters(heuristic=heuristic, p=p)
