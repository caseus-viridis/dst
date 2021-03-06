import math
import torch


def mask2braille(m, dims=(0, 1)):
    # permute dimensions
    ds = list(range(m.dim()))
    ds[0], ds[dims[0]] = dims[0], ds[0]
    ds[1], ds[dims[1]] = dims[1], ds[1]
    m = m.permute(*ds)
    # reduce mask
    _h, _w = m.size(0), m.size(1)
    m = m.view(_h, _w, -1).sum(-1) > 0
    # pad mask
    h, w = math.ceil(_h / 4) * 4, math.ceil(_w / 2) * 2
    m = torch.nn.functional.pad(m, (0, w - _w, 0, h - _h))
    # compute values
    k = torch.Tensor([[0, 3], [1, 4], [2, 5], [6, 7]])
    x = torch.nn.functional.conv2d(
        m.unsqueeze(0).unsqueeze(0).float(),
        2.**k.to(m.device).unsqueeze(0).unsqueeze(0),
        stride=(4, 2)).squeeze(0).squeeze(0).int() + 10240
    x = '\n'.join([''.join([chr(_x) for _x in row]) for row in x])
    return x

if __name__ == "__main__":
    pass