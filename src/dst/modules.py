import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .structured_sparse import StructuredSparseParameter, SparseParameter
from .utils import _calculate_fan_in_and_fan_out_from_size, sparse_mask_2d, checker_mask_2d


class _DSBase(nn.Module):
    r"""
    Base class of a dynamic sparse module
    """

    def __init__(self):
        super(_DSBase, self).__init__()


class DSLinear(_DSBase):
    r"""
    A dynamic sparse version of nn.Linear
    """

    def __init__(self, in_features, out_features, bias=True):
        super(DSLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = SparseParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_params()

    def init_params(self):
        self.weight.init_params()
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_from_size(
                self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight(), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class _DSConvNd(_DSBase):
    r"""
    A dynamic sparse version of the base convolution class like nn._ConvNd
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias):
        super(_DSConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = SparseParameter(
                torch.Tensor(in_channels, out_channels // groups,
                             *kernel_size))
        else:
            self.weight = SparseParameter(
                torch.Tensor(out_channels, in_channels // groups,
                             *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_params()

    def init_params(self):
        self.weight.init_params()
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_from_size(
                self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class DSConv2d(_DSConvNd):
    r"""
    A dynamic sparse version of nn.Conv2d
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        super(DSConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, False,
                             torch.nn.modules.utils._pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class _DSConvTransposeMixin(object):
    __constants__ = [
        'stride', 'padding', 'kernel_size', 'dim_size', 'output_padding',
        'groups', 'dilation', 'transposed', 'bias'
    ]

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)
        func = self._backend.ConvNd(self.stride, self.padding, self.dilation,
                                    self.transposed, output_padding,
                                    self.groups)
        if self.bias is None:
            return func(input, self.weight())
        else:
            return func(input, self.weight(), self.bias)

    def _output_padding(self, input, output_size, stride, padding,
                        kernel_size):
        if output_size is None:
            ret = torch.nn.modules.utils._single(
                self.output_padding)  # converting to list if was not already
        else:
            output_size = torch.jit._unwrap_optional(output_size)
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})".format(
                        k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes,
                            input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class DSConvTranspose2d(_DSConvTransposeMixin, _DSConvNd):
    r"""
    A dynamic sparse version of nn.ConvTranspose2d
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        output_padding = torch.nn.modules.utils._pair(output_padding)
        super(DSConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)
        return F.conv_transpose2d(input, self.weight(), self.bias, self.stride,
                                  self.padding, output_padding, self.groups,
                                  self.dilation)


class SpatialMask(nn.Module):
    r"""
    A spatial mask layer
        mask: a spatial multiplicative mask
        shuffle: whether to shuffle mask each time like dropout
    """

    def __init__(self, mask, shuffle=False):
        super(SpatialMask, self).__init__()
        self.register_buffer('mask', mask)
        self.shuffle = shuffle

    def reshuffle_mask_(self):
        self.mask = self.mask.view(-1)[torch.randperm(self.mask.numel())].view_as(self.mask)

    def forward(self, input):
        if self.shuffle:
            self.reshuffle_mask_()
        if not self.training and self.shuffle:
            return input * self.mask.float().mean()
        else:
            return input * self.mask.float()

    def extra_repr(self):
        return "size = {}".format(self.mask.shape)


# aliases
CheckerMask2d = lambda dim, quarters=1: SpatialMask(
    mask=checker_mask_2d(dim, quarters),
    shuffle=False
)
SparseMask2d = lambda dim, sparsity=0.25, dynamic=False: SpatialMask(
    mask=sparse_mask_2d(dim, sparsity),
    shuffle=dynamic
)