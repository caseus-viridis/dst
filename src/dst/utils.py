import numpy as np
import torch 

def rand_placement(shape, dof, seed=0):
    np.random.seed(seed)
    return torch.LongTensor(
        np.random.randint(dof, size=shape)
    )
