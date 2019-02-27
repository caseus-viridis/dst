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
from dst.reparameterization import DSModel, ReallocationHeuristics
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
    default=1024,
    help='hidden dimension (default: 1024)')
parser.add_argument(
    '-t',
    '--seq_len',
    type=int,
    default=50,
    help='sequence length for BPTT (default: 50)')
parser.add_argument(
    '-s',
    '--sparsification',
    type=str,
    default='none',
    help='sparsification method, none, comp, or dst (default: none)')
parser.add_argument(
    '-q',
    '--pct90',
    type=float,
    default=0.1,
    help='in the case of compression, the `q` parameter in Narang et al. 2017a,b (default: 0.1)')
parser.add_argument(
    '-ts',
    '--target-sparsity',
    type=float,
    default=0.5,
    help='in the case of DST, the target overall sparsity as in Mostafa & Wang 2018a,b (default: 0.5)')
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

# run name
if args.sparsification=='comp': # compression
    method_str = "-{}-q{}".format(args.sparsification, args.pct90)
elif args.sparsification=='dst':
    method_str = "-{}-s{}".format(args.sparsification, args.target_sparsity)
else:
    method_str = ""
run_name = "{}-{}-h{:d}-d{:d}".format(
    args.dataset, args.cell_type, args.hidden_size, args.depth) + method_str

# env
load_dotenv(verbose=True)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True
device = 'cuda:0'


# path
DATAPATH = os.getenv("DATAPATH")
if DATAPATH is None:
    logger.warning("Dataset directory is not configured. Please set the "
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
    filename=args.log_file,
    filemode='a')

# monitor
if args.monitor:
    writer, config = init_experiment({
        'title': "Character RNN experiments",
        'run_name': run_name,
        'log_dir': MONITORPATH + '/char_rnn',
        'random_seed': 7743
    })

# progress bar
pb_wrap = lambda it: tqdm(it, leave=False, dynamic_ncols=True)

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
    target_sparsity=args.target_sparsity if args.sparsification=='dst' else 0.,
    target_fraction_to_prune=1e-2,
    pruning_threshold=1e-2 if args.sparsification=='dst' else -1.
).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = RMSprop(model.parameters(),
    weight_decay=1e-5,
    lr=2e-3)
if args.sparsification=='dst':
    rp_schedule = lambda epoch: max([
        100 if epoch >=   0 else 0,
        200 if epoch >=  10 else 0,
        400 if epoch >=  20 else 0,
        800 if epoch >=  30 else 0,
        1e9 if epoch >=  40 else 0
    ])
else: # elif args.sparsification=='comp':
    rp_schedule = lambda epoch: max([
        100 if epoch >= 0 else 0,
        1e9 if epoch >= args.epochs//2 else 0
    ])
print(model)  # print the model description
print("Parameter count = {}".format(param_count(model)))

# Pruning threshold schedule as in Narang et al. 2017a,b
BATCHES_PER_EPOCH = len(data.train)
FREQ = 100
def get_pruning_threshold(itr,
                          q=args.pct90,
                          freq=FREQ,
                          start_itr=BATCHES_PER_EPOCH,
                          ramp_itr=BATCHES_PER_EPOCH * args.epochs // 4,
                          end_itr=BATCHES_PER_EPOCH * args.epochs // 2):
    theta = 2. * q * freq / (2. * (ramp_itr - start_itr) + 3. * (end_itr - ramp_itr))
    phi = 1.5 * theta
    if itr >= start_itr and itr < ramp_itr:
        return theta * (itr - start_itr + 1) / freq
    elif itr >= ramp_itr and itr < end_itr:
        return (theta * (ramp_itr - start_itr + 1) + phi * (itr - ramp_itr + 1)) / freq
    else:
        return -1.

def do_training(num_epochs=args.epochs):
    batch = batches_since_last_rp = 0
    for epoch in range(args.epochs):
        training_loss = 0.
        with pb_wrap(data.train) as loader:
            loader.set_description("Training epoch {:2d}".format(epoch))
            for ix, seq in enumerate(loader):
                loss = train(seq.text, seq.target)
                training_loss += loss
                if batches_since_last_rp == rp_schedule(epoch):
                    if args.sparsification=='comp':
                        model.pruning_threshold = get_pruning_threshold(batch)
                        if model.pruning_threshold > 0:
                            model.prune_by_threshold()
                    elif args.sparsification=='dst':
                        model.reparameterize(heuristic=ReallocationHeuristics.paper)
                    tqdm.write("BATCH #{:d}".format(batch))
                    tqdm.write(model.stats_table.get_string())
                    tqdm.write(model.sum_table.get_string())
                    if args.monitor:
                        writer.add_scalar('model.sparsity', model.sparsity, batch)
                        writer.add_scalar('model.np_free', model.np_free, batch)
                    batches_since_last_rp = 0
                batch += 1
                batches_since_last_rp += 1
                loader.set_postfix(loss="\33[91m{:6.4f}\033[0m".format(loss),
                    pruning_threshold="\33[91m{:10.8f}\033[0m".format(model.pruning_threshold))
        val_loss = val()
        training_loss /= ix + 1
        print(
            "Epoch {:3d}: training loss = {:6.4f}, validation loss = {:6.4f}"
            .format(epoch, training_loss, val_loss))
        if args.monitor:
            writer.add_scalar('training_loss', training_loss, batch)
            writer.add_scalar('val_loss', val_loss, batch)
            writer.add_scalar('model.np_free', model.np_free, batch)
            writer.add_scalar('model.sparsity', model.sparsity, batch)


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
