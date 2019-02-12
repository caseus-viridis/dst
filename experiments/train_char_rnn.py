import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from data import PennTreebank
from dst.models import char_rnn
from dst.reparameterization import DSModel
from dst.utils import param_count
from pytorch_monitor import init_experiment, monitor_module


parser = argparse.ArgumentParser(
    description='Character RNN dynamic sparse training')
parser.add_argument(
    '-ds',
    '--dataset',
    type=str,
    default='ptb',
    help='dataset, ptb or wt2 (default: ptb)')
parser.add_argument(
    '-c',
    '--cell-type',
    type=str,
    default='rnn',
    help='RNN cell type, rnn, lstm or gru (default: rnn)')
parser.add_argument(
    '-d',
    '--depth',
    type=int,
    default=2,
    help='number of layers (default: 2)')
parser.add_argument(
    '-n',
    '--hidden-size',
    type=int,
    default=128,
    help='hidden dimension (default: 128)')
parser.add_argument(
    '-t',
    '--seq_len',
    type=int,
    default=50,
    help='sequence length for BPTT (default: 50)')
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
    default=50,
    help='number of epochs (default: 50)')
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

#  logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
    filename=args.log_file,
    filemode='a')

# monitor
if args.monitor:
    writer, config = init_experiment({
        'title': "Character RNN experiments",
        'run_name': "{}-{}-{:d}".format(args.dataset, args.cell, args.hidden_size),
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
device = 'cuda:0'

# data path
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    logger.warning("Dataset directory is not configured. Please set the "
                   "DATAPATH env variable or create an .env file.")
    DATAPATH = './data'  # default

# data, model, loss, optimizer, lr_scheduler, rp_schedule
data = PennTreebank(
    data_dir=DATAPATH + '/' + args.dataset,
    batch_size=args.batch_size,
    bptt_len=args.seq_len,
    device=device)
model = DSModel(
    model=char_rnn.net(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_size=data.vocab_size,
        output_size=data.vocab_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        cell_type=args.cell_type,
        device=device
    ),
    target_sparsity=0.5,
    target_fraction_to_prune=1e-2,
    pruning_threshold=1e-3
).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = RMSprop(model.parameters(), weight_decay=1e-4, lr=2e-3)
rp_schedule = lambda epoch: max([
    100 if epoch >=   0 else 0,
    200 if epoch >=  10 else 0,
    400 if epoch >=  20 else 0,
    800 if epoch >=  30 else 0,
    1600 if epoch >=  40 else 0
])/2
print(model)  # print the model description
print("Parameter count = {}".format(param_count(model)))

def do_training(num_epochs=args.epochs):
    batch = batches_since_last_rp = 0
    for epoch in range(args.epochs):
        training_loss = 0.
        with pb_wrap(data.train) as loader:
            loader.set_description("Training epoch {:2d}, theta = {:.4f}".format(epoch, model.pruning_threshold))
            for batch, seq in enumerate(loader):
                batch += 1
                loss = train(seq.text, seq.target)
                training_loss += loss
                batches_since_last_rp += 1
                if batches_since_last_rp == rp_schedule(epoch):
                    model.reparameterize()
                    batches_since_last_rp = 0
                    tqdm.write(model.stats_table.get_string())
                    tqdm.write(model.sum_table.get_string())
                loader.set_postfix(loss="\33[91m{:6.4f}\033[0m".format(loss))
        val_loss = val()
        training_loss /= batch + 1
        print(
            "Epoch {:3d}: training loss = {:6.4f}, validation loss = {:6.4f}"
            .format(epoch, training_loss, val_loss))
        if args.monitor:
            writer.add_scalar('training_loss', training_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)


def train(x, y):
    model.train()
    # x, y = x.cuda(), y.cuda()
    x, y = x.to(device).t(), y.to(device).t()  # due to bug in torchtext.data.Field.batch_first, should be fixed in the next torchtext release
    loss = sum([loss_func(_y, y[:, t]) for t, _y in enumerate(model(x))])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item() / x.shape[1]


def val():
    model.eval()
    total_loss = 0.
    total_size = 0
    with torch.no_grad():
        with pb_wrap(data.test) as loader:
            loader.set_description("Validating")
            for batch, seq in enumerate(loader):
                # x, y = x.cuda(), y.cuda()
                x, y = seq.text.to(device).t(), seq.target.to(device).t()  # due to bug in torchtext.data.Field.batch_first, should be fixed in the next torchtext release
                loss = sum(
                    [loss_func(_y, y[:, t]) for t, _y in enumerate(model(x))])
                total_loss += loss.item()
                total_size += x.shape[1]
    return total_loss / total_size


if __name__ == "__main__":
    do_training()
