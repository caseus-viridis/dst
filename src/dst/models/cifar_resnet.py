import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import DSConv2d, DSConvTranspose2d
from ..activation_sparse import Checker2d, Sparse2d, SparseBatchNorm

__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
    'resnet1202'
]

def _spatial_bottleneck_activation(spatial_bottleneck, density, dim):
    assert density in (0.25, 0.5, 0.75)
    if spatial_bottleneck == 'none':
        return None
    elif spatial_bottleneck == 'structured':
        return Checker2d(dim=dim, quarters=int(density*4))
    elif spatial_bottleneck == 'static':
        return Sparse2d(dim=dim, density=density, dynamic=False)
    elif spatial_bottleneck == 'dynamic':
        return Sparse2d(dim=dim, density=density, dynamic=True)
    else:
        raise RuntimeError('Wrong spatial_bottleneck configuration')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 dim,
                 in_planes,
                 planes,
                 stride=1,
                 spatial_bottleneck=None,
                 density=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SparseBatchNorm(planes,
            _spatial_bottleneck_activation(spatial_bottleneck, density, dim))
        self.conv2 = nn.Conv2d(
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
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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
                 spatial_bottleneck=None,
                 density=0.5):
        super(ResNet, self).__init__()
        self.spatial_bottleneck = spatial_bottleneck
        self.density = density
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], dim=dim, stride=1)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], dim=dim // 2, stride=2)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], dim=dim // 4, stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

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
                    density=self.density))
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


def resnet20(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [3, 3, 3],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)


def resnet32(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [5, 5, 5],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)


def resnet44(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [7, 7, 7],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)


def resnet56(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [9, 9, 9],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)


def resnet110(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [18, 18, 18],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)


def resnet1202(num_classes=10, spatial_bottleneck=None, density=0.5):
    return ResNet(
        BasicBlock, [200, 200, 200],
        num_classes=num_classes,
        spatial_bottleneck=spatial_bottleneck,
        density=density)
