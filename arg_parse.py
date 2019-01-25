import argparse

parser = argparse.ArgumentParser(description='PyTorch dynamic sparse training')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('--model', type=str, choices = ['mnist_mlp','cifar10_WideResNet','imagenet_resnet50'],default='mnist_mlp',  help='network name (default: mnist_mlp)')

parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='L2 weight decay coefficient (default: 1e-4)')

parser.add_argument('--L1-loss-coeff', default=0.0, type=float,
                    help='Lasso coefficient (default: 0.0)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers for wide resnet (default: 28)')

parser.add_argument('--start-pruning-after-epoch', default=20, type=int,
                    help='Epoch after which to start pruning (default: 20)')

parser.add_argument('--prune-epoch-frequency', default=2, type=int,
                    help='Interval (in epochs) between prunes (default: 2)')

parser.add_argument('--prune-target-sparsity-fc', default=0.98, type=float,
                    help='Target sparsity when pruning fully connected layers (default: 0.98)')

parser.add_argument('--prune-target-sparsity-conv', default=0.5, type=float,
                    help='Target sparsity when pruning conv layers (default: 0.5)')

parser.add_argument('--prune-iterations', default=1, type=int,
                    help='Number of prunes. Set to 1 for single prune, larger than 1 for gradual pruning (default: 1)')

parser.add_argument('--post-prune-epochs', default=10, type=int,
                    help='Epochs to train after pruning is done (default: 10)')

parser.add_argument('--n-realloc-params', default=600, type=int,
                    help='Target number of parameters to reallocate each prune/grow cycle (default: 600)')

parser.add_argument('--threshold-prune',  action='store_true',
                    help='Prune based on a global adaptive threshold and not a fixed fraction from each layer  (default: False)')

parser.add_argument('--prune', dest='prune', action='store_true',
                    help='prune mode to sparsify a dense model  (default: False)')

parser.add_argument('--validate-set',  action='store_true',
                    help='whether to use a validation set or not  (default: False)')

parser.add_argument('--grow-across-layers',  action='store_true',
                    help='Move weights between layers in the prune/grow cycle. (default: False)')


parser.add_argument('--tied',  action='store_true',
                    help='whether to use tied weights instead of sparse ones, i.e, similar to hash nets  (default: False)')

parser.add_argument('--rewire', action='store_true',
                    help='whether to run parameter re-allocation (default: False)')

parser.add_argument('--no-validate-train', action='store_true',
                    help='whether to run validation on training set (default: False)')

parser.add_argument('--DeepR', action='store_true',
                    help='DeepR mode (default: False)')

parser.add_argument('--DeepR_eta', default=0.001, type=float,
                    help='eta coefficient for DeepR (default: 0.1)')


parser.add_argument('--stop-rewire-epoch', default=1000, type=int,
                    help='Epoch after which to stop rewiring (default: 1000)')


parser.add_argument('--rewire-fraction', default=0.1, type=float,
                    help='Fraction of weight to rewire. Only effective if threshold-prune is false  (default: 0.1)')


parser.add_argument('--sub-kernel-granularity',action='store_true',
                    help='Use sub-kernel granularity while rewiring(default: False)')

parser.add_argument('--cubic-prune-schedule',action='store_true',
                    help='Use sparsity schedule following a cubic function as in Zhu et al. 2018 (instead of an exponential function). (default: False)')

parser.add_argument('--sparse-resnet-downsample',action='store_true',
                    help='Use a sparse/tied tensor for the resnet downsampling convolution(default: False)')


parser.add_argument('--conv-group-lasso',action='store_true',
                    help='Use group lasso to penalize an entire kernel patch(default: False)')

parser.add_argument('--big-new-weights',action='store_true',
                    help='Use weights initialized from the initial distribution for the new connections instead of zeros(default: False)')

parser.add_argument('--widen-factor', default=10, type=float,
                    help='widen factor for wide resnet (default: 10)')


parser.add_argument('--initial-sparsity-conv', default=0.5, type=float,
                    help=' Initial sparsity of conv layers(default: 0.5)')

parser.add_argument('--initial-sparsity-fc', default=0.98, type=float,
                    help=' Initial sparsity for fully connected layers(default: 0.98)')

parser.add_argument('--job-idx', default=0, type=int,
                    help='job index provided by the job manager')


parser.add_argument('--data', metavar='DIR',
                    help='path to imagenet dataset',default = '/dataset/imagenet/')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')


parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--schedule-file', default='./mnist_schedule.yaml', type=str,
                    help='yaml file containing learning rate schedule and rewire period schedule')

parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.set_defaults(augment=True)
