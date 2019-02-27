import argparse
import os
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from data import CIFAR10, CIFAR100
from dst.models import cifar_wrn
from dst.reparameterization import DSModel
from dst.utils import param_count
from pytorch_monitor import init_experiment, monitor_module
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, LRScheduler

parser = argparse.ArgumentParser(
    description='CIFAR10/100 WRN-28-W dynamic sparse training')
parser.add_argument(
    '-w', '--width', type=int, default=2, help='width of WRN (default: 2)')
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
    help='Spatial bottleneck architecture (default: False)')
parser.add_argument(
    '-r', '--run-id', type=int, default=0, help='Run ID (default: 0)')
parser.add_argument(
    '--gpu', default='0', type=str, help='id(s) for GPU(s) to use')
parser.add_argument(
    '-m',
    '--monitor',
    action='store_true',
    help='monitoring or not (default: False)')
args = parser.parse_args()

# run name
run_name = "{}-wrn_28_{:d}{}-run{:d}".format(
    args.dataset, args.width, "-sb" if args.spatial_bottleneck else "",
    args.run_id)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
torch.backends.cudnn.benchmark = True

# env
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or './data'
MONITOR_PATH = os.getenv("MONITOR_PATH") or './monitor'
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH") or './checkpoint'

# monitor
if args.monitor:
    writer, config = init_experiment({
        'title': "CIFAR10/100 WRN experiments",
        'run_name': run_name,
        'log_dir': MONITOR_PATH + '/cifar_wrn',
        'random_seed': args.run_id
    })

# checkpointer
checkpointer = ModelCheckpoint(
    dirname=CHECKPOINT_PATH + '/cifar_wrn',
    filename_prefix=run_name,
    save_interval=1,
    require_empty=False,
    save_as_state_dict=False)

# data, model, loss, optimizer, lr_scheduler, rp_schedule
data = eval(args.dataset.upper())(
    data_dir=DATAPATH + '/' + args.dataset,
    cuda=True,
    num_workers=8,
    batch_size=args.batch_size,
    shuffle=True)
model = DSModel(
    model=cifar_wrn.net(
        num_classes=100 if args.dataset == 'cifar100' else 10,
        width=args.width,
        spatial_bottleneck=args.spatial_bottleneck),
    target_sparsity=0.9).cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = SGD(
    model.parameters(),
    lr=1e-1,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True)
scheduler = LRScheduler(
    MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2))
rp_schedule = lambda epoch: max([
    100 if epoch >=   0 else 0,
    200 if epoch >=  25 else 0,
    400 if epoch >=  80 else 0,
    800 if epoch >= 140 else 0
])
print(model)  # print the model description
print("Parameter count = {}".format(param_count(model)))

trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)
evaluator = create_supervised_evaluator(
    model,
    metrics={
        'accuracy': Accuracy(),
        'loss': Loss(loss_func)
    },
    device=device)

RunningAverage(alpha=0.9, output_transform=lambda x: x).attach(trainer, 'loss')
ProgressBar().attach(trainer, ['loss'])


@trainer.on(Events.STARTED)
def init_counter(engine):
    engine.state.iterations_since_last_rp = 0


@trainer.on(Events.ITERATION_COMPLETED)
def reparameterize(engine):
    engine.state.iterations_since_last_rp += 1
    if engine.state.iterations_since_last_rp == rp_schedule(
            engine.state.epoch):
        # print("Reparameterize model at Iteration {}, Epoch {}".format(engine.state.iteration, engine.state.epoch))
        model.reparameterize()
        engine.state.iterations_since_last_rp = 0


trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    print("Epoch {:3d}: train loss = {:.4f}".format(
        engine.state.epoch, engine.state.metrics['loss']))
    if args.monitor:
        writer.add_scalar('train_loss', engine.state.metrics['loss'],
                          engine.state.epoch)


@trainer.on(Events.EPOCH_COMPLETED)
def log_test_results(engine):
    evaluator.run(data.test)
    loss, accuracy = evaluator.state.metrics['loss'], evaluator.state.metrics[
        'accuracy']
    print("Epoch {:3d}: test loss = {:.4f}, test accuracy = {:.4f}".format(
        engine.state.epoch, loss, accuracy))
    if args.monitor:
        writer.add_scalar('test_loss', loss, engine.state.epoch)
        writer.add_scalar('accuracy', accuracy, engine.state.epoch)


@trainer.on(Events.EPOCH_COMPLETED)
def save_checkpoint(engine):
    checkpointer(
        engine,
        dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=dict(
                epoch=trainer.state.epoch,
                iteration=trainer.state.iteration,
                iterations_since_last_rp=trainer.state.
                iterations_since_last_rp),
        ))


if __name__ == "__main__":
    trainer.run(data.train, max_epochs=args.epochs)
