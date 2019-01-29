import numpy as np
import torch

def rand_placement(shape, dof, seed=0):
    np.random.seed(seed)
    return torch.LongTensor(
        np.random.randint(dof, size=shape)
    )

def _calculate_fan_in_and_fan_out_from_size(sz):
    dimensions = len(sz)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  
        fan_in = sz[1]
        fan_out = sz[0]
    else:
        num_input_fmaps = sz[1]
        num_output_fmaps = sz[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = np.prod(sz[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
