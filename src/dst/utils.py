import numpy as np
import torch

param_count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad) if isinstance(m, torch.nn.Module) else 0


def rand_placement(shape, dof, seed=0):
    r"""
    Random placement for hashed dense parameter tensor
    """
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

def sparse_mask_2d(dim, density=0.5):
    r"""
    A sparse 2D binary mask
    """
    mask = torch.ByteTensor(dim**2).zero_()
    mask[:int(density*dim**2)] = 1
    return mask[torch.randperm(dim**2)].view([dim, dim])

def checker_mask_2d(dim, quarters=1):
    r"""
    A regular sparse 2D binary mask with checkerboard pattern
    """
    mask = torch.ByteTensor(size=[dim, dim]).zero_()
    if quarters > 0:
        mask[0::2, 0::2] = 1
    if quarters > 1:
        mask[1::2, 1::2] = 1
    if quarters > 2:
        mask[0::2, 1::2] = 1
    if quarters > 3:
        mask[1::2, 0::2] = 1
    return mask