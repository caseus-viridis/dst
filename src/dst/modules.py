import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .structured_dense import StructuredDenseParameter
from .structured_sparse import StructuredSparseParameter, SparseParameter
from .utils import _calculate_fan_in_and_fan_out_from_size


class _DSBase(nn.Module):
    r"""
    Base class of a dynamic sparse module
    """

    def __init__(self):
        super(_DSBase, self).__init__()


class ConcatenatedParameters(nn.ModuleList):
    def __init__(self, *mlist, dim=0):
        super(ConcatenatedParameters, self).__init__(mlist)
        self.dim=dim

    def forward(self):
        return torch.cat([m() for m in self], dim=self.dim)


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


class _DSRNNCellBase(_DSBase):
    r"""
    A dynamic sparse version of _RNNCellBase
    """

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(_DSRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = ConcatenatedParameters(*[
            SparseParameter(torch.Tensor(hidden_size, input_size)) for _ in range(num_chunks)
        ], dim=0)
        self.weight_hh = ConcatenatedParameters(*[
            SparseParameter(torch.Tensor(hidden_size, hidden_size)) for _ in range(num_chunks)
        ], dim=0)
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".
                format(input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".
                format(input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".
                format(hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        for m in self.modules():
            if isinstance(m, StructuredDenseParameter):
                nn.init.uniform_(m.bank, -stdv, stdv)

    def zero_state(self, batch_size=1, device='cpu'):
        if isinstance(self, (DSRNNCell, DSGRUCell)):
            return torch.zeros(batch_size, self.hidden_size).to(device)
        elif isinstance(self, (DSLSTMCell)):
            return (torch.zeros(batch_size, self.hidden_size).to(device),
                    torch.zeros(batch_size, self.hidden_size).to(device))
        else:
            raise RuntimeError(
                "zero_state() not implemented for {}".format(self))


class DSRNNCell(_DSRNNCellBase):
    r"""
    A dynamic sparse version of RNNCell
    """

    def __init__(self, input_size, hidden_size, bias=True,
                 nonlinearity="tanh"):
        super(DSRNNCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(
                input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        if self.nonlinearity == "tanh":
            func = torch._C._VariableFunctions.rnn_tanh_cell
        elif self.nonlinearity == "relu":
            func = torch._C._VariableFunctions.rnn_relu_cell
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(
                self.nonlinearity))
        state = output = func(
            input,
            hx,
            self.weight_ih(),
            self.weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )
        return output, state


class DSLSTMCell(_DSRNNCellBase):
    r"""
    A dynamic sparse version of LSTMCell
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(DSLSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(
                input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        h_, c_ = torch._C._VariableFunctions.lstm_cell(
            input,
            hx,
            self.weight_ih(),
            self.weight_hh(),
            self.bias_ih,
            self.bias_hh,
        )

        output = h_
        state = (h_, c_)
        return output, state


class DSGRUCell(_DSRNNCellBase):
    r"""
    A dynamic sparse version of GRUCell
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(DSGRUCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(
                input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx)
        state = output = torch._C._VariableFunctions.gru_cell(
            input,
            hx,
            self.weight_ih(),
            self.weight_hh(),
            self.bias_ih,
            self.bias_hh,

        )
        return output, state

        
