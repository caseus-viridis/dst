import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class DivergentPaths(nn.ModuleList):
    def __init__(self, *mlist):
        super(DivergentPaths, self).__init__(mlist)

    def forward(self, *args, **kwargs):
        return (m(*args, **kwargs) for m in self)


class ParallelPaths(nn.ModuleList):
    def __init__(self, *mlist):
        super(ParallelPaths, self).__init__(mlist)

    def forward(self, inputs):
        return [m(x) for m, x in zip(self, inputs)]


class ConvergentPaths(nn.Module):
    def __init__(self, red_fn=lambda x, y: x+y):
        super(ConvergentPaths, self).__init__()
        self.red_fn = red_fn
    
    def forward(self, inputs):
        return reduce(self.red_fn, inputs)


class SemiResidualBlock(nn.Module):
    def __init__(self, f, g, h):
        super(SemiResidualBlock, self).__init__()
        self.f = f
        self.g = g
        self.h = h

    def forward(self, x, y):
        return self.f(x), self.g(y) + self.h(x)


class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x):
        return x.view(*x.shape[:2], -1).mean(dim=-1)


class BNReLUPoolLin(nn.Module):
    def __init__(self, ni, no):
        super(BNReLUPoolLin, self).__init__()
        self.norm = nn.BatchNorm2d(ni)
        self.nl = nn.ReLU()
        self.pool = GlobalAvgPooling()
        self.lin = nn.Linear(ni, no)

    def forward(self, x):
        return self.lin(self.pool(self.nl(self.norm(x))))


class BNReLUConv(nn.Module):
    """
    A sequential BatchNorm-ReLU-Conv3x3
        ni: number of input feature maps
        no: number of output feature maps
        dir: direction of scale change, -1, 0 or 1
    """

    def __init__(self,
                 ni,
                 no,
                 dir=0):
        super(BNReLUConv, self).__init__()
        self.norm = nn.BatchNorm2d(ni)
        self.nl = nn.ReLU(inplace=True)
        conv = nn.ConvTranspose2d if dir > 0 else nn.Conv2d
        extra = dict(output_padding=1) if dir > 0 else {}
        stride = 1 if dir == 0 else 2
        self.conv = conv(
            ni,
            no,
            kernel_size=3,
            bias=False,
            stride=stride,
            padding=1,
            **extra)

    def forward(self, x):
        return self.conv(self.nl(self.norm(x)))

def get_skip(ni, no, fg=None):
    if fg is None:
        return lambda _: 0.  # no skip connection
    else:
        if no == ni and fg == 0:
            return nn.Sequential()  # identity skip
        elif fg < 0: # downsampling 1x1 conv skip
            return nn.Conv2d(ni, no, kernel_size=1, stride=2)
        elif fg > 0: # upsampling 1x1 conv skip
            return nn.ConvTranspose2d(ni, no, kernel_size=1, stride=2, output_padding=1)
        else: # 1x1 conv skip
            return nn.Conv2d(ni, no, kernel_size=1)

def semi_res_conv(f, g, h):
    return lambda ni, no: SemiResidualBlock(
        f=get_skip(ni, no, f),
        g=get_skip(ni, no, g),
        h=BNReLUConv(ni, no, dir=h)
    )


class CResNetBlock(nn.Module):
    def __init__(self, ni, no, head, tail):
        super(CResNetBlock, self).__init__()
        self.head = head(ni, no)
        self.tail = tail(no, no)

    def forward(self, input):
        x, y = input
        u, v = self.head(x, y)
        y, x = self.tail(v, u)
        output = (x, y)
        return output

resnet_block = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, None, 0), tail=semi_res_conv(0, 0, 0))
resnet_block_trans = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(-1, None, -1), tail=semi_res_conv(0, 0, 0))
resnet_rsbn_block = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, None, -1), tail=semi_res_conv(0, 0, 1))
resnet_rsbn_block_trans = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, None, -1), tail=semi_res_conv(-1, -1, 0))
cresnet_block = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, 0, 0), tail=semi_res_conv(0, 0, 0))
cresnet_block_trans = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(-1, -1, -1), tail=semi_res_conv(0, 0, 0))
cresnet_rsbn_block = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, 0, -1), tail=semi_res_conv(0, 0, 1))
cresnet_rsbn_block_first = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, -1, -1), tail=semi_res_conv(0, 0, 1))
cresnet_rsbn_block_trans = lambda ni, no: CResNetBlock(ni, no,
    head=semi_res_conv(0, 0, -1), tail=semi_res_conv(-1, -1, 0))


def group(ni, no, first_block, later_block, depth=1):
    return nn.Sequential(*[(first_block if d==0 else later_block)(
        ni if d==0 else no, no
    ) for d in range(depth)])


resnet_group = lambda ni, no, depth=1, no_trans=False: group(
    ni, no, resnet_block if no_trans else resnet_block_trans, resnet_block, depth=depth
)
resnet_rsbn_group = lambda ni, no, depth=1, no_trans=False: group(
    ni, no, resnet_rsbn_block if no_trans else resnet_rsbn_block_trans, resnet_rsbn_block, depth=depth
)
cresnet_group = lambda ni, no, depth=1, no_trans=False: group(
    ni, no, cresnet_block if no_trans else cresnet_block_trans, cresnet_block, depth=depth
)
cresnet_rsbn_group = lambda ni, no, depth=1, no_trans=False: group(
    ni, no, cresnet_rsbn_block_first if no_trans else cresnet_rsbn_block_trans, cresnet_rsbn_block, depth=depth
)


class CResNet(nn.Module):
    """
    Coupled ResNet
    """
    def __init__(
            self,
            num_classes=10,
            num_scales=3,
            width=16,
            depth=4,  # note this is the number of blocks per group, not the total depth
            rsbn=False,
            coupled=False
        ):
        super(CResNet, self).__init__()
        widths = [int(width * 2**s) for s in range(num_scales)]
        if coupled:
            group = cresnet_rsbn_group if rsbn else cresnet_group
        else:
            group = resnet_rsbn_group if rsbn else resnet_group

        self.head = DivergentPaths(
            nn.Conv2d(
                3, width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.Conv2d(
                3, width,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ) if rsbn else nn.Conv2d(
                3, width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        self.body = nn.Sequential(*[
            group(
                ni=width if s==0 else widths[s-1],
                no=widths[s],
                depth=depth,
                no_trans=s==0
            ) for s in range(num_scales)
        ])
        self.tail = nn.Sequential(
            ParallelPaths(
                BNReLUPoolLin(widths[-1], num_classes),
                BNReLUPoolLin(widths[-1]//2 if rsbn else widths[-1], num_classes)
            ), 
            ConvergentPaths(red_fn=lambda x, y: x+y) if coupled else ConvergentPaths(red_fn=lambda x, y: x)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
