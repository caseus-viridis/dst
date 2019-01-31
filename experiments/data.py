import torch
from torchvision import datasets, transforms
import numpy as np
from functools import partial


class MNIST(object):
    def __init__(self,
                 data_dir='./data/mnist',
                 cuda=False,
                 num_workers=4,
                 batch_size=64,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.train = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_dir, train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.test = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)


class CIFAR(object):
    def __init__(self,
                 num_classes=10,
                 data_dir='./data/cifar',
                 cuda=False,
                 num_workers=4,
                 batch_size=64,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        # normalize = transforms.Normalize(
        #     mean=(0.4914, 0.4822, 0.4465),
        #     std=(0.2023, 0.1994, 0.2010)
        # )
        normalize = transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(), normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        self.train = torch.utils.data.DataLoader(
            eval("datasets.CIFAR{:d}".format(num_classes))(
                data_dir, train=True, download=True,
                transform=transform_train),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.test = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transform_test),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)

# aliases
CIFAR10 = partial(CIFAR, num_classes=10)
CIFAR100 = partial(CIFAR, num_classes=100)


class I1K(object):
    def __init__(self,
                 data_dir='./I1K/i1k-extracted',
                 cuda=False,
                 num_workers=4,
                 batch_size=32,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.train = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_dir + '/train',
                                 transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.val = torch.utils.data.DataLoader(
            datasets.ImageFolder(data_dir + '/val',
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)
