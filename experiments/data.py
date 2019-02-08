import torch
import torchvision as tv
import torchtext as tt
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
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.train = torch.utils.data.DataLoader(
            tv.datasets.MNIST(
                data_dir, train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.test = torch.utils.data.DataLoader(
            tv.datasets.MNIST(data_dir, train=False, transform=transform),
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
        normalize = tv.transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)
        transform_train = tv.transforms.Compose([
            tv.transforms.Pad(4, padding_mode='reflect'),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32),
            tv.transforms.ToTensor(), normalize
        ])
        transform_test = tv.transforms.Compose(
            [tv.transforms.ToTensor(), normalize])
        self.train = torch.utils.data.DataLoader(
            eval("tv.datasets.CIFAR{:d}".format(num_classes))(
                data_dir, train=True, download=True,
                transform=transform_train),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.test = torch.utils.data.DataLoader(
            eval("tv.datasets.CIFAR{:d}".format(num_classes))(
                data_dir, train=False, download=True,
                transform=transform_test),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)


# aliases
CIFAR10 = partial(CIFAR, num_classes=10)
CIFAR100 = partial(CIFAR, num_classes=100)


class I1K(object):
    def __init__(self,
                 data_dir='./data/I1K/i1k-extracted',
                 cuda=False,
                 num_workers=4,
                 batch_size=32,
                 shuffle=True):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True
        } if cuda else {}
        normalize = tv.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.train = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/train',
                tv.transforms.Compose([
                    tv.transforms.RandomResizedCrop(224),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.val = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/val',
                tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size,
            shuffle=False,
            **gpu_conf)


class PennTreebank(object):
    def __init__(self,
                 data_dir='~/data/PennTreebank',
                 cuda=False,
                 batch_size=32,
                 bptt_len=50):
        self.text = tt.data.Field(tokenize=lambda s: list(s)) # character-level tokenization
        self.train, self.val, self.test = tt.datasets.PennTreebank.splits(
            text_field=self.text,
            # batch_size=batch_size,
            # bptt_len=bptt_len,
            # device=None if cuda else -1
        )
        self.text.build_vocab(self.train)


if __name__ == "__main__":
    ptb = PennTreebank()
    import ipdb; ipdb.set_trace()