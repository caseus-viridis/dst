import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from data import CIFAR10, CIFAR100
from dst.models import cifar_resnet
# from dst.reparameterization import DSModel
from dst.utils import param_count
from pytorch_monitor import init_experiment, monitor_module
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, LRScheduler

parser = argparse.ArgumentParser(
    description='CIFAR10/100 ResNet experiments')
parser.add_argument(
    '-d', '--depth', type=int, default=110, help='depth of resnet (default: 110)')
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
    default=64,
    help='batch size (default: 64)')
parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=400,
    help='number of epochs (default: 400)')
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
    '-r',
    '--run-id',
    type=int,
    default=0,
    help='Run ID (default: 0)')
parser.add_argument(
    '--gpu', default='0', type=str, help='id(s) for GPU(s) to use')
parser.add_argument(
    '-m',
    '--monitor',
    action='store_true',
    help='monitoring or not (default: False)')
args = parser.parse_args()

# experiment and run name
exp_name = 'cifar_resnet'
run_name = "{}-resnet{:d}-sb_{}{}-run{:d}".format(
    args.dataset, args.depth,
    args.spatial_bottleneck,
    "" if args.spatial_bottleneck=='none' else "-q_{:d}".format(args.quarters),
    args.run_id
)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = True

# env
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or './data'
MONITOR_PATH = os.getenv("MONITOR_PATH") or './monitor'
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH") or './checkpoint'

# monitor
if args.monitor:
    writer, config = init_experiment({
        'title': "CIFAR10/100 ResNet experiments",
        'run_name': run_name,
        'log_dir': '/'.join([MONITOR_PATH, exp_name]),
        'random_seed': args.run_id
    })

# checkpointer
checkpointer = ModelCheckpoint(
    dirname='/'.join([CHECKPOINT_PATH, exp_name]),
    filename_prefix=run_name,
    save_interval=1,
    require_empty=False,
    save_as_state_dict=True
)
# define checkpoint structure
class assemble_checkpoint:
    def state_dict(self):
        return dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler=scheduler,
            trainer_state=dict(
                epoch=trainer.state.epoch,
                iteration=trainer.state.iteration
            ),
            checkpointer_state=dict(
                _iteration=checkpointer._iteration
            )
        )

# data, model, loss, optimizer, lr_scheduler, rp_schedule
data = eval(args.dataset.upper())(
    data_dir='/'.join([DATA_PATH, args.dataset]),
    cuda=True,
    num_workers=0,
    batch_size=args.batch_size,
    shuffle=True)
model = eval("cifar_resnet.resnet{:d}".format(args.depth))(
    num_classes=100 if args.dataset == 'cifar100' else 10,
    spatial_bottleneck=args.spatial_bottleneck,
    density=0.25*args.quarters).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=1e-4,
    momentum=0.9,
    nesterov=True)
scheduler = LRScheduler(
    MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1))
print(model)
print("Parameter count = {}".format(param_count(model)))

# engines
trainer = create_supervised_trainer(
    model, optimizer, loss_func,
    device=device
)
evaluator = create_supervised_evaluator(
    model, metrics={
        'accuracy': Accuracy(),
        'loss': Loss(loss_func)
    },
    device=device
)
RunningAverage(alpha=0.9, output_transform=lambda x: x).attach(trainer, 'loss')
ProgressBar().attach(trainer, ['loss'])

# resume from checkpoint
@trainer.on(Events.STARTED)
def resume_from_checkpoint(engine):
    ckpt = glob('/'.join([CHECKPOINT_PATH, exp_name, run_name]) + '*.pth')
    if ckpt:
        print("Resuming from checkpoint: {}".format(*ckpt))
        _ckpt = torch.load(*ckpt)
        model.load_state_dict(_ckpt['model_state_dict'])
        optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
        scheduler = _ckpt['scheduler']
        for k, v in _ckpt['trainer_state'].items():
            setattr(trainer.state, k, v)
        for k, v in _ckpt['checkpointer_state'].items():
            setattr(checkpointer, k, v)
        os.remove(*ckpt)

# LR scheduler
trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

# log training results
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    if args.monitor:
        writer.add_scalar('train_loss', engine.state.metrics['loss'], engine.state.epoch)

# do test and log results
@trainer.on(Events.EPOCH_COMPLETED)
def log_test_results(engine):
    evaluator.run(data.test)
    loss, accuracy = evaluator.state.metrics['loss'], evaluator.state.metrics['accuracy']
    if args.monitor:
        writer.add_scalar('test_loss', loss, engine.state.epoch)
        writer.add_scalar('accuracy', accuracy, engine.state.epoch)

# print summary to console after each epoch
@trainer.on(Events.EPOCH_COMPLETED)
def print_results(engine):
    print(
        "Epoch {:3d}: train loss = {:.4f}, test loss = {:.4f}, test accuracy = {:.4f}".format(
            trainer.state.epoch,
            trainer.state.metrics['loss'],
            evaluator.state.metrics['loss'],
            evaluator.state.metrics['accuracy']
        )
    )

# save checkpoint
trainer.add_event_handler(Events.EPOCH_COMPLETED, 
    checkpointer, {'ckpt': assemble_checkpoint()})


if __name__ == "__main__":
    trainer.run(data.train, max_epochs=args.epochs)
