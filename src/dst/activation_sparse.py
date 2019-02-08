import torch
import torch.nn as nn
from .utils import _calculate_fan_in_and_fan_out_from_size, sparse_mask_2d, checker_mask_2d


class SparseActivation(nn.Module):
    r"""
    A sparse activation tensor
        mask: a spatial multiplicative mask
        dynamic: whether to reshuffle mask each time like dropout
        dense_inference: whether to use underlying dense activation at inference time
    """

    def __init__(self, mask, dynamic=False, dense_inference=False):
        super(SparseActivation, self).__init__()
        self.register_buffer('mask', mask)
        self.dynamic = dynamic
        self.dense_inference = dense_inference

    def reshuffle_mask_(self):
        self.mask = self.mask.view(-1)[torch.randperm(
            self.mask.numel())].view_as(self.mask)

    def forward(self, input):
        if self.dynamic:
            self.reshuffle_mask_()
        if not self.training and self.dense_inference:
            return input * self.mask.float().mean()
        else:
            return input * self.mask.float()

    def extra_repr(self):
        return "mask shape = {}".format(self.mask.shape)

    def sparsity(self):
        return 1. - self.mask.float().mean().item()


# aliases
Checker2d = lambda dim, quarters=1: SparseActivation(
    mask=checker_mask_2d(dim, quarters),
    dynamic=False
)
Sparse2d = lambda dim, density=0.5, dynamic=False: SparseActivation(
    mask=sparse_mask_2d(dim, density),
    dynamic=dynamic,
    dense_inference=dynamic
)


class SparseBatchNorm(nn.Module):
    r"""
    A batchnorm following a sparse activation (i.e. perform normalization on only the non-zero components)
    """

    def __init__(self, num_features, sparse_activation=None):
        super(SparseBatchNorm, self).__init__()
        assert sparse_activation is None or isinstance(sparse_activation,
                                                       SparseActivation)
        self.sparse_activation = sparse_activation
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, input):
        if self.sparse_activation is None:
            output = self.bn(input.view(*input.shape[:2], -1))
        else:
            output = self.sparse_activation(input).view(*input.shape[:2], -1)
            if self.training:
                ix = self.sparse_activation.mask.view(-1)
                output[..., ix] = self.bn(output[..., ix])
            else:
                if self.sparse_activation.dynamic: # gnarly situation of scaling back spatial dropout for inference
                    output = self.bn(input.view(*input.shape[:2], -1)) * (1. - self.sparse_activation.sparsity())
                else:
                    output = self.bn(output)
        return output.view_as(input)
