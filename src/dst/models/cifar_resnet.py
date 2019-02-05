import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import DSConv2d, DSConvTranspose2d, CheckerMask2d, SparseMask2d

__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
    'resnet1202'
]


def _get_spatial_mask(spatial_mask, dim):
    if spatial_mask == 'sparse_fixed_quarter':
        return SparseMask2d(dim=dim, sparsity=0.25, dynamic=False)
    elif spatial_mask == 'sparse_dynamic_quarter':
        return SparseMask2d(dim=dim, sparsity=0.25, dynamic=True)
    if spatial_mask == 'sparse_fixed_half':
        return SparseMask2d(dim=dim, sparsity=0.5, dynamic=False)
    elif spatial_mask == 'sparse_dynamic_half':
        return SparseMask2d(dim=dim, sparsity=0.5, dynamic=True)
    if spatial_mask == 'sparse_fixed_three_quarters':
        return SparseMask2d(dim=dim, sparsity=0.75, dynamic=False)
    elif spatial_mask == 'sparse_dynamic_three_quarters':
        return SparseMask2d(dim=dim, sparsity=0.75, dynamic=True)
    elif spatial_mask == 'checker_quarter':
        return CheckerMask2d(dim=dim, quarters=1)
    elif spatial_mask == 'checker_half':
        return CheckerMask2d(dim=dim, quarters=2)
    elif spatial_mask == 'checker_three_quarters':
        return CheckerMask2d(dim=dim, quarters=3)
    else:
        return nn.Sequential()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 dim,
                 in_planes,
                 planes,
                 stride=1,
                 spatial_bottleneck=False,
                 spatial_mask=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride * 2 if spatial_bottleneck else stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sm = _get_spatial_mask(
            spatial_mask, dim) if not spatial_bottleneck else nn.Sequential()
        self.conv2 = nn.ConvTranspose2d(
            planes,
            planes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        ) if spatial_bottleneck else nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.sm(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 dim=32,
                 num_classes=10,
                 spatial_bottleneck=False,
                 spatial_mask=None):
        super(ResNet, self).__init__()
        self.spatial_bottleneck = spatial_bottleneck
        self.spatial_mask = spatial_mask
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], dim=dim, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], dim=dim//2, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], dim=dim//4, stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, dim, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    dim,
                    self.in_planes,
                    planes,
                    stride,
                    spatial_bottleneck=self.spatial_bottleneck,
                    spatial_mask=self.spatial_mask))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [3, 3, 3],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)


def resnet32(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [5, 5, 5],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)


def resnet44(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [7, 7, 7],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)


def resnet56(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [9, 9, 9],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)


def resnet110(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [18, 18, 18],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)


def resnet1202(num_classes=10, spatial_bottleneck=False, spatial_mask=None):
    return ResNet(
        BasicBlock, [200, 200, 200],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        spatial_mask=spatial_mask)
