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
from data import I1K
from dst.models import i1k_resnet
# from dst.reparameterization import DSModel
from dst.utils import param_count
from pytorch_monitor import init_experiment, monitor_module

parser = argparse.ArgumentParser(
    description='Imagenet1K ResNet experiments')
parser.add_argument(
    '-d', '--depth', type=int, default=50, help='depth of resnet (default: 50)')
parser.add_argument(
    '-z',
    '--batch-size',
    type=int,
    default=256,
    help='batch size (default: 256)')
parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=100,
    help='number of epochs (default: 100)')
parser.add_argument(
    '-sb',
    '--spatial-bottleneck',
    type=str,
    default='none',
    help='Spatial bottleneck sparsification: structured, static, dynamic or none (default: none)')
parser.add_argument(
    '-q',
    '--quarters',
    type=int,
    default=2,
    help='In case of spatial bottleneck sparsity, density of activation in quarters (default: 2)')
parser.add_argument(
    '--gpu', default='0', type=str, help='id(s) for GPU(s) to use')
parser.add_argument(
    '-m',
    '--monitor',
    action='store_true',
    help='monitoring or not (default: False)')
args = parser.parse_args()
run_name = "{}-resnet{:d}-sb_{}{}".format(
    'i1k', args.depth,
    args.spatial_bottleneck,
    "" if args.spatial_bottleneck=='none' else "-q{:d}".format(args.quarters)
)

# env
load_dotenv(verbose=True)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# paths
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    print("Dataset directory is not configured. Please set the "
        "DATAPATH env variable or create an .env file.")
    DATAPATH = './data'  # default
MONITORPATH = os.getenv("MONITORPATH")
if MONITORPATH is None:
    print("Monitor directory is not configured. Please set the "
        "MONITORPATH env variable or create an .env file.")
    MONITORPATH = './runs'  # default


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
        'title': "I1K ResNet experiments",
        'run_name': run_name,
        'log_dir': MONITORPATH + '/i1k_resnet',
        'random_seed': 7734
    })

# progress bar
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)

# data, model, loss, optimizer, lr_scheduler, rp_schedule
data = I1K(
    data_dir=DATAPATH + '/i1k',
    cuda=True,
    num_workers=8,
    batch_size=args.batch_size,
    shuffle=True)
model = eval("i1k_resnet.resnet{:d}".format(args.depth))(
    spatial_bottleneck=args.spatial_bottleneck,
    density=0.25*args.quarters).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=1e-4,
    momentum=0.9,
    nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
print(model)  # print the model description
print("Parameter count = {}".format(param_count(model)))

def do_training(num_epochs=args.epochs):
    batch = 0
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        training_loss = 0.
        with pb_wrap(data.train) as loader:
            loader.set_description("Training epoch {:3d}".format(epoch))
            for i, (x, y) in enumerate(loader):
                batch += 1
                loss = train(x, y)
                training_loss += loss
                loader.set_postfix(
                    loss="\33[91m{:6.4f}\033[0m".format(loss))
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res 


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
                prec1, prec5 = accuracy(_y, y, 1), accuracy(_y, y, 5)
                import ipdb; ipdb.set_trace()
                pred = _y.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_size += x.shape[0]
    return total_loss / (batch + 1), correct / total_size


if __name__ == "__main__":
    do_training()
