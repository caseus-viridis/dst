import torch
import torch.nn as nn
from ..modules import DSConv2d, DSConvTranspose2d


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Skip(nn.Module):
    def __init__(self, ni=1, no=1, stride=1):
        super(Skip, self).__init__()
        self.trans = nn.Conv2d(
            ni, no, kernel_size=1, stride=stride,
            bias=False) if ni != no else None

    def forward(self, x):
        return x if self.trans is None else self.trans(x)


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
    A sequential Norm-NL-Conv3x3
        ni: number of input feature maps
        no: number of output feature maps
        stride: stride of convolution layer
        padding: padding of convolution layer
    """
    def __init__(self,
                 ni,
                 no,
                 conv=DSConv2d,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 **kwargs):
        super(BNReLUConv, self).__init__()
        self.norm = nn.BatchNorm2d(ni)
        self.nl = nn.ReLU(inplace=True)
        self.conv = conv(
            ni,
            no,
            kernel_size=kernel_size,
            bias=False,
            stride=stride,
            padding=padding,
            **kwargs
        )
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.running_mean, 0.)
                nn.init.constant_(m.running_var, 1.)

    def forward(self, x):
        return self.conv(self.nl(self.norm(x)))


class WideResNetBlock(nn.Module):
    def __init__(self, ni, no, k=1, stride=1, spatial_bottleneck=False, **kwargs):
        super(WideResNetBlock, self).__init__()
        self.head = BNReLUConv(
            ni * k, no * k,
            stride=stride*2 if spatial_bottleneck else stride,
            conv=DSConv2d,
            **kwargs
        )
        self.tail = BNReLUConv(
            no * k, no * k,
            stride=2 if spatial_bottleneck else 1,
            conv=DSConvTranspose2d if spatial_bottleneck else DSConv2d,
            output_padding=1 if spatial_bottleneck else 0,
            **kwargs
        )
        self.skip = Skip(ni * k, no * k, stride=stride)

    def forward(self, x):
        return self.tail(self.head(x)) + self.skip(x)


def wide_resnet_group(ni, no, k=1, stride=1, depth=1, spatial_bottleneck=False):
    assert depth >= 1, "invalid depth"
    return [WideResNetBlock(
                ni if i == 0 else no,
                no,
                k=k,
                stride=stride if i == 0 else 1,
                spatial_bottleneck=spatial_bottleneck
            ) for i in range(depth)]


class WideResNet(nn.Module):
    """
    Wide ResNet
    Zagoruyko et al. 2016 (http://arxiv.org/abs/1605.07146)
    """
    def __init__(
            self,
            num_classes=10,
            num_features=16,
            num_scales=3,
            width=1,
            depth=4,  # note this is the number of blocks per group, not the total depth
            spatial_bottleneck=False
        ):
        super(WideResNet, self).__init__()
        widths = [int(num_features * width * 2**s) for s in range(num_scales)]
        self.head = DSConv2d(
            3, num_features, kernel_size=3, padding=1, bias=False)
        self.body = nn.Sequential(*[
            nn.Sequential(*wide_resnet_group(
                ni=num_features if s==0 else widths[s-1],
                no=widths[s],
                stride=1 if s==0 else 2,
                depth=depth,
                spatial_bottleneck=spatial_bottleneck
            )) for s in range(num_scales)
        ])
        self.tail = BNReLUPoolLin(widths[-1], num_classes)

    def forward(self, x):
        return self.tail(self.body(self.head(x)))


def net(width, depth=4, num_classes=10, num_features=16, num_scales=3, spatial_bottleneck=False):
    return WideResNet(
        num_classes=num_classes,  # 10 for cifar10, 100 for cifar100
        num_features=num_features,
        num_scales=num_scales,
        width=width,
        depth=depth,
        spatial_bottleneck=spatial_bottleneck
    )
