import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from data import CIFAR10
from dst.models import cifar10_wrn
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='CIFAR10-WRN dynamic sparse training')
parser.add_argument(
    '-w',
    '--width',
    type=int,
    default=2,
    help='width of WRN (default: 2)')
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
    data_dir=DATAPATH+'/cifar10',
    cuda=True,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=True)
model = cifar10_wrn.net(args.width).cuda()
loss = nn.CrossEntropyLoss().cuda()
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True)
scheduler = LambdaLR(
    optimizer, 
    lambda epoch: max([
        1e-1 if epoch < 60  else 0., 
        2e-2 if epoch < 120 else 0.,
        4e-3 if epoch < 160 else 0.,
        8e-4 if epoch < 200 else 0.
    ]), 
    last_epoch=-1)



# import ipdb; ipdb.set_trace()