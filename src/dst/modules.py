import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .structured_sparse import StructuredSparseParameter, SparseParameter
from .utils import _calculate_fan_in_and_fan_out_from_size


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
            fan_in, _ = _calculate_fan_in_and_fan_out_from_size(self.weight.size())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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
            fan_in, _ = _calculate_fan_in_and_fan_out_from_size(self.weight.size())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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
        super(DSConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, False,
                                       torch.nn.modules.utils._pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight(), self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)