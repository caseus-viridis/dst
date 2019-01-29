import torch
import torch.nn as nn
# from ..dynamics import DSNetwork
from ..modules import DSLinear

def net():
    return nn.Sequential(
        DSLinear(784, 300, bias=False),
        nn.BatchNorm1d(300),
        nn.ReLU(inplace=True),
        DSLinear(300, 100, bias=False),
        nn.BatchNorm1d(100),
        nn.ReLU(inplace=True),
        DSLinear(100, 10, bias=False)
    )
