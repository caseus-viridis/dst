import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from data import CIFAR10
from dst.models import cifar10_wrn
from dst.dynamics import get_sparse_param_stats, prune_or_grow_to_sparsity
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='CIFAR10-WRN dynamic sparse training')
parser.add_argument(
    '-w', '--width', type=int, default=2, help='width of WRN (default: 2)')
parser.add_argument(
    '-z',
    '--batch-size',
    type=int,
    default=128,
    help='batch size (default: 128)')
parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=200,
    help='number of epochs (default: 200)')
parser.add_argument(
    '--gpu', default='0', type=str, help='id(s) for GPU(s) to use')
args = parser.parse_args()

# progress bar
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)

# env
from dotenv import load_dotenv
load_dotenv(verbose=True)

# data path
logger = logging.getLogger(__name__)
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    logger.warning("Dataset directory is not configured. Please set the "
                   "DATAPATH env variable or create an .env file.")
    DATAPATH = './data'  # default

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

data = CIFAR10(
    data_dir=DATAPATH + '/cifar10',
    cuda=True,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=True)
model = cifar10_wrn.net(args.width).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# scheduler = LambdaLR(
#     optimizer,
#     lambda epoch: max([
#         1e-1 if epoch < 60  else 0.,
#         2e-2 if epoch < 120 else 0.,
#         4e-3 if epoch < 160 else 0.,
#         8e-4 if epoch < 200 else 0.
#     ]),
#     last_epoch=-1)

# print the model description
print(model)

def train(epochs=args.epochs):
    


def _train(epoch):
    model.train()
    scheduler.step()
    total_loss = 0.
    with pb_wrap(data.train) as loader:
        loader.set_description("Training epoch {:3d}, lr = {:6.4f}".format(
            epoch, [pg['lr'] for pg in optimizer.param_groups][0]))
        for batch, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            loss = loss_func(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loader.set_postfix(loss="{:6.4f}".format(loss.item()))
    return total_loss / (batch + 1)


def test(epoch):
    model.eval()
    total_loss = correct = 0.
    total_size = 0
    with torch.no_grad():
        with pb_wrap(data.test) as loader:
            loader.set_description("Testing after epoch {:3d}".format(epoch))
            for batch, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                _y = model(x)
                loss = loss_func(model(x), y)
                total_loss += loss.item()
                pred = _y.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_size += x.shape[0]
    return total_loss / (batch + 1), correct / total_size


if __name__ == "__main__":
    prune_or_grow_to_sparsity(model, sparsity=0.99)
    n_total, n_dense, n_sparse, n_nonzero, sparsity, breakdown = get_sparse_param_stats(
        model)
    print("Total parameter count = {}".format(n_total))
    print("Dense parameter count = {}".format(n_dense))
    print("Sparse parameter count = {}".format(n_sparse))
    print("Nonzero sparse parameter count = {}".format(n_nonzero))
    print("Sparsity = {:6.4f}".format(sparsity))

    for epoch in range(args.epochs):
        training_loss = train(epoch)
        test_loss, correct = test(epoch)
        print(
            "Epoch {:3d}: training loss = \33[91m{:6.4f}\033[0m, test loss = \33[91m{:6.4f}\033[0m \tcorrect% = \33[92m{:5.2f}\033[0m"
            .format(epoch, training_loss, test_loss, correct * 100))
        # n_total, n_dense, n_sparse, n_nonzero, sparsity, breakdown = get_sparse_param_stats(
        #     model)
        # print("Total parameter count = {}".format(n_total))
        # print("Dense parameter count = {}".format(n_dense))
        # print("Sparse parameter count = {}".format(n_sparse))
        # print("Nonzero sparse parameter count = {}".format(n_nonzero))
        # print("Sparsity = {:6.4f}".format(sparsity))
