import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from parameterized_tensors import SparseTensor,TiedTensor

class DynamicLinear(nn.Module):

    def __init__(self, in_features, out_features, initial_sparsity, bias = True , sparse = True ):
        super(DynamicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sparsity = initial_sparsity
        self.sparse = sparse
        
        if sparse:
            self.d_tensor = SparseTensor([out_features,in_features],initial_sparsity = initial_sparsity)
        else:
            self.d_tensor = TiedTensor([out_features,in_features],initial_sparsity = initial_sparsity)
            
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        self.d_tensor.init_parameters()
            
    def forward(self, input):
        return F.linear(input, self.d_tensor() , self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, initial_sparsity = {}, bias={}'.format(
            self.in_features, self.out_features, self.initial_sparsity,self.bias is not None)



class DynamicConv2d(nn.Module):

    def __init__(self,
                 n_input_maps,
                 n_output_maps,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias = True,initial_sparsity = 0.5,sub_kernel_granularity = False,sparse=True):

        super(DynamicConv2d, self).__init__()

        if n_input_maps % groups != 0:
            raise ValueError('n_input_maps must be divisible by groups')

        self.sparse = sparse
        self.n_input_maps = n_input_maps
        self.n_output_maps = n_output_maps
        self.kernel_size = kernel_size


        if sparse:
            self.d_tensor = SparseTensor([n_output_maps,n_input_maps // groups, kernel_size, kernel_size],initial_sparsity = initial_sparsity,sub_kernel_granularity = sub_kernel_granularity)
        else:
            self.d_tensor = TiedTensor([n_output_maps,n_input_maps // groups, kernel_size, kernel_size],initial_sparsity = initial_sparsity,sub_kernel_granularity = sub_kernel_granularity)

        if bias:
            self.bias = Parameter(torch.Tensor(n_output_maps))
        else:
            self.bias = None

        self.groups = groups
        self.stride = (stride,) * 2
        self.padding = (padding,) * 2
        self.dilation = (dilation,) * 2

        self.init_parameters()

        
    def init_parameters(self):
        if self.bias is not None:
            self.bias.data.zero_()
        self.d_tensor.init_parameters()

        
    def forward(self, input):
        return F.conv2d(input, self.d_tensor(), self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

    def extra_repr(self):
        s = ('{name}({n_input_maps}, {n_output_maps}, kernel_size={kernel_size}, bias = {bias_exists}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ')'
        return s.format(
            name=self.__class__.__name__,bias_exists = self.bias is not None,**self.__dict__)
    
