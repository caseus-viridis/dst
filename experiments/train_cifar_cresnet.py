import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from data import CIFAR10, CIFAR100
from dst.models.cresnet import CResNet
from dst.reparameterization import DSModel
from dst.utils import param_count
from pytorch_monitor import init_experiment, monitor_module


parser = argparse.ArgumentParser(
    description='CIFAR10/100 CResNet training')
parser.add_argument(
    '-w', '--width', type=int, default=32, help='width of cresnet (default: 32)')
parser.add_argument(
    '-d', '--depth', type=int, default=4, help='depth of cresnet groups (default: 4)')
parser.add_argument(
    '-ds',
    '--dataset',
    type=str,
    default='cifar10',
    help='dataset, cifar10 or cifar100 (default: cifar10)')
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
    '-sb',
    '--spatial-bottleneck',
    action='store_true',
    help='spatial bottleneck architecture (default: False)')
parser.add_argument(
    '-c',
    '--coupled',
    action='store_true',
    help='coupled architecture (default: False)')
parser.add_argument(
    '--gpu', default='0', type=str, help='id(s) for GPU(s) to use')
parser.add_argument(
    '-l',
    '--log-file',
    type=str,
    default='log/' + os.path.splitext(os.path.split(__file__)[1])[0] + '.log',
    help='log file')
parser.add_argument(
    '-m',
    '--monitor',
    action='store_true',
    help='monitoring or not (default: False)')
args = parser.parse_args()
run_name = "{}-cresnet-w{:d}-d{:d}-{}-{}".format(
    args.dataset, args.width, args.depth,
    'rsbn' if args.spatial_bottleneck else 'res',
    'double' if args.coupled else 'single'
)

#  logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
    filename='./log/{}.log'.format(run_name),
    filemode='a')

# monitor
if args.monitor:
    writer, config = init_experiment({
        'title': "CIFAR cresnet experiments",
        'run_name': run_name,
        'log_dir': './runs',
        'random_seed': 7734
    })

# progress bar
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)

# env
load_dotenv(verbose=True)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# data path
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    logger.warning("Dataset directory is not configured. Please set the "
                   "DATAPATH env variable or create an .env file.")
    DATAPATH = './data'  # default

# data, model, loss, optimizer, lr_scheduler, rp_schedule
data = eval(args.dataset.upper())(
    data_dir=DATAPATH + '/' + args.dataset,
    cuda=True,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=True)
model = CResNet(
    num_classes=100 if args.dataset=='cifar100' else 10,
    width=args.width,
    depth=args.depth,
    rsbn=args.spatial_bottleneck,
    coupled=args.coupled
).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
logger.debug(model)  # print the model description
logger.debug("Parameter count = {}".format(param_count(model)))


def do_training(num_epochs=args.epochs):
    batch = 0
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        training_loss = 0.
        with pb_wrap(data.train) as loader:
            loader.set_description("Training epoch {:3d}, lr = {:.4f}".format(
            epoch, [pg['lr'] for pg in optimizer.param_groups][0]))
            for i, (x, y) in enumerate(loader):
                batch += 1
                loss = train(x, y)
                training_loss += loss
                loader.set_postfix(loss="\33[91m{:6.4f}\033[0m".format(loss))
        test_loss, correct = test()
        training_loss /= i + 1
        logger.debug(
            "Epoch {:3d}: training loss = {:6.4f}, test loss = {:6.4f}, correct% = {:5.2f}"
            .format(epoch, training_loss, test_loss, correct * 100))
        if args.monitor:
            writer.add_scalar('training_loss', training_loss, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('correct', correct, epoch)


def train(x, y):
    model.train()
    x, y = x.cuda(), y.cuda()
    loss = loss_func(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    total_loss = correct = 0.
    total_size = 0
    with torch.no_grad():
        with pb_wrap(data.test) as loader:
            loader.set_description("Testing")
            for batch, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                _y = model(x)
                loss = loss_func(_y, y)
                total_loss += loss.item()
                pred = _y.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_size += x.shape[0]
    return total_loss / (batch + 1), correct / total_size


if __name__ == "__main__":
    do_training()
